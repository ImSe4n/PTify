"""The fine-tuning loop. `python -m training.train --resume auto`.

RESUME IS THE SAME CODE PATH AS START
-------------------------------------
There is no separate resume script. `--resume auto` looks for the newest
checkpoint in the output directory and continues from it, or starts fresh if
there is none. That is deliberate: a resume path exercised only after a crash
is a path nobody has tested at the moment it matters most.

WHAT RUNS WHERE
---------------
This module is written to run on Kaggle (GPU, MAESTRO mounted read-only) and
to be rehearsed here on CPU with a handful of tracks. The only differences are
flags. Nothing about the loop knows which machine it is on beyond `--device`.

THE SMOKE RUN (Phase 14.5) IS NOT ABOUT ACCURACY
-------------------------------------------------
Its job is to prove the chain: targets decode, a checkpoint round-trips
Kaggle -> local, `torchlibrosa` works on Kaggle's torch, kill/resume works,
and the saved file survives `model.assert_deployable`. A terrible score is an
expected outcome of 500 steps and says nothing about the approach.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .checkpoint import (
    JsonlLogger,
    SaveTrigger,
    capture_rng_state,
    find_latest,
    load_training_state,
    prune,
    save_training_state,
    torch_load,
)
from .dataset import SegmentDataset, collate
from .index import read_index, segments_from_index
from .losses import compute_losses, parse_weights
from .model import load_pretrained, save_deployable, trainable_parameters

#: ~1/10 of ByteDance's from-scratch 5e-4. Fine-tuning wants to refine the
#: representation, not restructure it; too high and the pretrained weights are
#: destroyed in the first few hundred steps (the loss recovers, so this is not
#: obvious from the curve alone).
DEFAULT_LR = 5e-5

DEFAULT_BATCH_SIZE = 8
DEFAULT_WARMUP_STEPS = 200


def lr_at(step: int, base_lr: float, warmup: int = DEFAULT_WARMUP_STEPS,
          decay_every: int = 2000, decay: float = 0.9) -> float:
    """Linear warmup, then stepwise decay.

    Warmup matters more when fine-tuning than when training fresh: the
    optimiser starts with no moment estimates, and a full-rate step against
    good pretrained weights is a large, uninformed move.
    """
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr * (decay ** ((step - warmup) // decay_every))


def build_dataloader(args, split: str, *, shuffle: bool, augment=None,
                     epoch_offset: int = 0):
    import torch

    index = read_index(Path(args.index))
    segments = segments_from_index(index, split=split)
    if not segments:
        raise ValueError(
            f"No {split!r} segments in {args.index}. "
            f"Splits present: {index['splits']}"
        )
    if args.max_segments:
        segments = segments[: args.max_segments]

    dataset = SegmentDataset(segments, audio_root=args.audio_root,
                             midi_root=args.midi_root or args.audio_root,
                             augment=augment, epoch_offset=epoch_offset,
                             soft_onset_boost=getattr(
                                 args, "soft_onset_boost", 1.0))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        collate_fn=collate,
        drop_last=True,
        # Workers re-import the package per process; keeping them alive avoids
        # paying soxr's ~2-6s lazy init on every epoch boundary. MEASURED at
        # 14.8 seg/s/worker against 8.3 without — so turning this off to let a
        # `set_epoch` reach the workers would have cost 44% of throughput and
        # broken the >=15 budget. `epoch_offset` gets epoch variety instead,
        # without mutating anything a worker holds a copy of.
        persistent_workers=args.workers > 0,
    )


def resume_epoch_state(start_epoch: int) -> tuple[int, int]:
    """`(epoch, loader_epoch)` to begin a run or a resume with.

    The training loop increments `epoch` before using it, so the counter has
    to start one BELOW the epoch to run: a fresh start gives 0 -> epoch 1, and
    a checkpoint saved during epoch 5 gives 4 -> epoch 5 again, finishing the
    epoch it was interrupted in rather than skipping to 6.

    `loader_epoch` records which epoch the already-built loader belongs to, so
    the loop rebuilds on real epoch boundaries instead of comparing against a
    literal — the bug this replaces was `epoch > 1`, which silently meant
    "epoch 1's augmentation for the rest of the run" after any resume.

    Pulled out of `train()` because that function needs a model, a dataset and
    a GPU to reach this arithmetic, and the arithmetic is where the off-by-one
    lives.

    RELATED HAZARD, not guarded here: `epoch_offset` is `epoch * len(dataset)`,
    so resuming with a different `--index` or `--max-segments` changes the
    dataset length and silently re-maps every segment's augmentation. The
    checkpoint stores the full `vars(args)` for exactly this kind of question
    — compare against it before resuming with changed flags. Unreachable while
    a run stays inside epoch 1, which at 70,517 steps per epoch is every run
    this project currently does.
    """
    epoch = max(start_epoch - 1, 0)
    return epoch, epoch + 1


def build_augmenter(args, *, epoch: int = 0):
    """The training-time augmenter, or None when `--augment` is off.

    Off by default so Phase 14.5's known-good configuration stays exactly
    reproducible; a run that changes two things at once cannot attribute
    either.
    """
    if not args.augment:
        return None
    from .augment import AugmentationSampler

    return AugmentationSampler(
        seed=args.augment_seed, epoch=epoch,
        clean_prob=args.augment_clean_prob,
        max_cents=args.augment_max_cents,
        eq_prob=args.augment_eq_prob,
    )


def diagnose_nan(note_model, batch, use_amp: bool) -> str:
    """Locate a non-finite loss: the input, the forward pass, or the loss.

    These three fail identically from the outside — `loss = nan` — but need
    completely different fixes, and on free-tier compute a wrong guess costs a
    whole session. So the failure path reports which one it was rather than
    leaving the next person to bisect it by hand.
    """
    import torch

    lines = []
    finite_in = bool(torch.isfinite(batch["waveform"]).all())
    lines.append(f"  input waveform finite: {finite_in}")
    if not finite_in:
        return "\n".join(lines + ["  -> the AUDIO is corrupt, not the model."])

    with torch.no_grad():
        fp32 = note_model(batch["waveform"].float())
        fp32_ok = all(bool(torch.isfinite(v).all()) for v in fp32.values())
        lines.append(f"  forward in fp32 finite: {fp32_ok}")

        if use_amp:
            with torch.autocast(device_type="cuda", enabled=True):
                amp = note_model(batch["waveform"])
            amp_ok = all(bool(torch.isfinite(v).all()) for v in amp.values())
            lines.append(f"  forward under AMP finite: {amp_ok}")
            for key, value in amp.items():
                if not torch.isfinite(value).all():
                    share = float((~torch.isfinite(value)).float().mean())
                    lines.append(f"    {key}: {share:.1%} non-finite, "
                                 f"dtype={value.dtype}")
            if fp32_ok and not amp_ok:
                lines.append("  -> the FORWARD PASS overflows in fp16. "
                             "Re-run with --no-amp; the loss is not at fault.")
                return "\n".join(lines)

    lines.append("  -> forward is finite, so the LOSS is at fault "
                 "(see training/losses.py:_CLAMP).")
    return "\n".join(lines)


def evaluate(note_model, loader, device: str, max_batches: int = 20) -> dict:
    """Mean losses over a few validation batches.

    Deliberately bounded: this runs inside the training loop, and a full pass
    over 68,646 validation segments would cost more than the training it is
    meant to be monitoring.
    """
    import torch

    note_model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            # Validation is scored UNWEIGHTED, on purpose. `val_*` is the
            # regression guard and is compared against runs with different
            # weightings (and against the 14.5 baseline), so it has to mean the
            # same thing in every log. Only the training objective is weighted.
            losses = compute_losses(note_model(batch["waveform"]), batch)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            count += 1
    note_model.train()
    return {f"val_{k}": v / max(count, 1) for k, v in totals.items()}


def train(args) -> int:
    import torch

    device = args.device
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(out_dir / "train_log.jsonl")

    loss_weights = parse_weights(getattr(args, "loss_weights", None))

    print(f"Loading pretrained weights (device={device}) ...")
    note_model, pedal_state = load_pretrained(device)
    note_model.train()
    print(f"  {trainable_parameters(note_model):,} trainable parameters")
    print("  pedal model loaded and FROZEN (re-attached unchanged on save)")

    # Continue from OUR OWN weights rather than ByteDance's, when asked.
    #
    # `load_pretrained` always loads ByteDance's checkpoint, so before this
    # every run restarted from the baseline and threw away whatever the last
    # one learned -- fine for a single 10h session, useless for a multi-week
    # plan, where the whole point is to accumulate. `--resume` does NOT cover
    # this: it continues an *interrupted* run from its `step_N.pt`, and the
    # 260MB resumable file for 16b was never kept.
    #
    # Loaded AFTER load_pretrained so the pedal weights and the dropout patch
    # both come from the known-good path, and `strict=True` so a checkpoint
    # from a different architecture fails here rather than leaving layers
    # randomly initialised.
    if getattr(args, "init_checkpoint", None):
        init_path = Path(args.init_checkpoint)
        print(f"Initialising from {init_path} (NOT the pretrained baseline) ...")
        state = torch_load(str(init_path), device)
        note_state = (state.get("model", {}).get("note_model")
                      if "model" in state else state.get("note_model"))
        if note_state is None:
            raise ValueError(
                f"{init_path} has no note_model weights; expected a deployable "
                f"checkpoint ({{'model': {{'note_model': ...}}}}) or a training "
                f"checkpoint with a 'note_model' key. Keys: {sorted(state)}"
            )
        note_model.load_state_dict(note_state, strict=True)
        print("  loaded; training continues from these weights")

    if loss_weights != {k: 1.0 for k in loss_weights}:
        print(f"Loss weights: {loss_weights}")
        print("  (per-head logging stays UNWEIGHTED so logs remain comparable)")

    optimizer = torch.optim.Adam(note_model.parameters(), lr=args.lr)
    # AMP only helps on CUDA; on CPU it is a no-op wrapper that costs nothing
    # but must still exist so the save/resume shape is identical either way.
    use_amp = device.startswith("cuda") and not args.no_amp
    # torch.amp.GradScaler on >=2.4; torch.cuda.amp.GradScaler on 2.2 (the
    # local pin). Kaggle ships 2.10, where the old spelling warns.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_step = 0
    start_epoch = 0
    if args.resume:
        path = (find_latest(out_dir) if args.resume == "auto"
                else Path(args.resume))
        if path and Path(path).exists():
            info = load_training_state(
                path, note_model=note_model, optimizer=optimizer,
                scaler=scaler, device=device,
            )
            start_step = info["step"]
            # The epoch must come back too. The augmentation condition for a
            # segment is hashed from (seed, epoch, index), so resuming at
            # epoch 0 would re-draw epoch 1's conditions for the rest of the
            # run and never draw the later epochs' at all — silently narrowing
            # the training distribution, with no error and a loss curve that
            # looks fine. `test_training_state_round_trips` pinned that the
            # checkpoint CARRIES epoch, which made this look covered while the
            # consumer discarded it.
            start_epoch = info["epoch"]
            print(f"Resumed from {path} at step {start_step} "
                  f"(epoch {start_epoch})")
        elif args.resume != "auto":
            raise FileNotFoundError(f"--resume {args.resume} does not exist")
        else:
            print("No checkpoint found; starting fresh")

    augmenter = build_augmenter(args)
    loader = build_dataloader(args, args.train_split, shuffle=True,
                              augment=augmenter)
    # A resume lands mid-epoch, so the first loader must carry the RESUMED
    # epoch's offset rather than epoch 1's — otherwise the rest of the run
    # re-draws epoch 1's conditions. Rebuilt only when resuming into epoch >= 2
    # (epoch 1 already has offset 0), and it has to be a second call because
    # the offset needs the dataset length, which the first loader supplies.
    if augmenter is not None and start_epoch > 1:
        loader = build_dataloader(
            args, args.train_split, shuffle=True, augment=augmenter,
            epoch_offset=start_epoch * len(loader.dataset),
        )

    # Clean validation is the REGRESSION GUARD: it answers "did fine-tuning
    # damage what already worked", stays comparable to the Phase 14.5 baseline
    # and to everything in benchmarks/, and is the metric that catches the real
    # disaster of destroying the pretrained weights.
    val_loader = (build_dataloader(args, args.val_split, shuffle=False)
                  if args.validate_every else None)

    # Augmented validation is the metric this whole track is OPTIMISING. A
    # clean val curve can improve while room robustness goes nowhere, and that
    # would not surface until Phase 17 scores against MAPS.
    #
    # It is pinned to epoch 0 deliberately: if the val condition moved with the
    # training epoch, the curve would shift because the augmentation shifted
    # and no step-to-step comparison would mean anything.
    val_aug_loader = None
    if args.validate_every and augmenter is not None:
        val_aug_loader = build_dataloader(
            args, args.val_split, shuffle=False,
            augment=build_augmenter(args, epoch=0),
        )

    print(f"{len(loader.dataset):,} training segments, "
          f"batch {args.batch_size}, {args.workers} workers")
    if augmenter is not None:
        print(f"  augmentation ON (seed {args.augment_seed}, "
              f"clean {args.augment_clean_prob:.0%}, "
              f"+-{args.augment_max_cents:g} cents, "
              f"{len(augmenter.bank)} impulse responses)")

    trigger = SaveTrigger(args.save_every_steps, args.save_every_seconds)
    step = start_step
    started = time.time()
    # Continues from the checkpoint rather than restarting at 0, so a resumed
    # run keeps drawing the epoch it was actually in. See `resume_epoch_state`
    # for why the counter starts one below the epoch to run.
    epoch, loader_epoch = resume_epoch_state(start_epoch)

    accum = max(1, args.accum_steps)

    while step < args.steps:
        epoch += 1
        # A fresh condition for every segment each epoch. The loader is rebuilt
        # rather than the sampler mutated, because a persistent worker holds a
        # COPY of the sampler that a `set_epoch` would never reach — and
        # turning persistence off to fix that cost 44% of throughput (measured
        # 8.3 vs 14.8 seg/s/worker). Rebuilding costs one worker respawn per
        # epoch, against a ~40-minute epoch.
        #
        # NOTE: one epoch of the full index is 70,517 steps at effective batch
        # 8 (~72h at the measured 0.27 steps/s), so no run on a 12-hour Kaggle
        # session reaches a second epoch. This branch is correctness for a
        # future longer run, not something Phase 16b exercises.
        if augmenter is not None and epoch != loader_epoch:
            loader = build_dataloader(
                args, args.train_split, shuffle=True, augment=augmenter,
                epoch_offset=epoch * len(loader.dataset),
            )
            loader_epoch = epoch
        optimizer.zero_grad(set_to_none=True)
        micro = 0

        for batch in loader:
            if step >= args.steps:
                break

            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, args.lr, args.warmup)

            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type="cuda" if use_amp else "cpu",
                                enabled=use_amp):
                losses = compute_losses(note_model(batch["waveform"]), batch,
                                        loss_weights)

            if not torch.isfinite(losses["total"]):
                # A NaN loss poisons every weight through the backward pass,
                # and the run continues producing NaN forever with no error.
                # The first Kaggle run did exactly that for 500 steps. Fail at
                # the first occurrence, and say WHERE it came from — the loss
                # and the forward pass fail identically from the outside.
                raise RuntimeError(
                    f"Non-finite loss at step {step}: "
                    + ", ".join(f"{k}={float(v.detach()):.4g}"
                                for k, v in losses.items())
                    + "\n" + diagnose_nan(note_model, batch, use_amp)
                )

            # Scale by 1/accum so the accumulated gradient equals the mean
            # over the effective batch, not its sum — otherwise the effective
            # learning rate silently scales with --accum-steps.
            scaler.scale(losses["total"] / accum).backward()
            micro += 1

            if micro < accum:
                continue
            micro = 0

            # Unscale before clipping, or the clip threshold is applied to
            # scaled gradients and does nothing at fp16's scale factors.
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                note_model.parameters(), args.clip
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            step += 1

            if step % args.log_every == 0:
                elapsed = time.time() - started
                record = {
                    "step": step, "epoch": epoch,
                    "lr": lr_at(step, args.lr, args.warmup),
                    "elapsed_s": round(elapsed, 1),
                    "steps_per_s": round(
                        (step - start_step) / max(elapsed, 1e-6), 3),
                    "grad_norm": float(grad_norm),
                    # .detach() before float(): torch warns that converting a
                    # grad-tracking tensor to a scalar can behave unexpectedly.
                    **{k: float(v.detach()) for k, v in losses.items()},
                }
                if device.startswith("cuda"):
                    record["gpu_mem_gb"] = round(
                        torch.cuda.max_memory_allocated() / 1e9, 2)
                logger.log(record)
                print(f"  step {step:>6} loss {record['total']:.4f} "
                      f"(onset {record['onset']:.4f} "
                      f"frame {record['frame']:.4f}) "
                      f"{record['steps_per_s']:.2f} steps/s")

            if val_loader is not None and step % args.validate_every == 0:
                metrics = evaluate(note_model, val_loader, device,
                                   args.val_batches)
                if val_aug_loader is not None:
                    metrics |= {
                        k.replace("val_", "val_aug_"): v
                        for k, v in evaluate(note_model, val_aug_loader,
                                             device, args.val_batches).items()
                    }
                logger.log({"step": step, **metrics})
                line = f"  step {step:>6} VAL total {metrics['val_total']:.4f}"
                if "val_aug_total" in metrics:
                    line += f"  AUG {metrics['val_aug_total']:.4f}"
                print(line)

            if trigger.should_save(step):
                path = save_training_state(
                    out_dir / f"step_{step}.pt",
                    note_model=note_model, optimizer=optimizer, step=step,
                    epoch=epoch, scaler=scaler if use_amp else None,
                    config=vars(args) | {"device": device},
                    rng_state=capture_rng_state(),
                )
                trigger.mark(step)
                removed = prune(out_dir, keep=args.keep_checkpoints)
                print(f"  saved {path.name}"
                      + (f" (pruned {len(removed)})" if removed else ""))

    # Always save at the end, even if the trigger just fired: the deployable
    # file is the artifact everything downstream consumes.
    save_training_state(
        out_dir / f"step_{step}.pt", note_model=note_model,
        optimizer=optimizer, step=step, epoch=epoch,
        scaler=scaler if use_amp else None,
        config=vars(args) | {"device": device}, rng_state=capture_rng_state(),
    )
    deployable = save_deployable(note_model, pedal_state,
                                 out_dir / args.deployable_name)
    size_mb = deployable.stat().st_size / 1e6
    print(f"\nWrote {deployable} ({size_mb:.1f} MB) — verified deployable")
    print(f"Finished at step {step} in {(time.time() - started) / 60:.1f} min")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="training.train",
        description="Fine-tune the ByteDance piano CRNN for room robustness.",
    )
    ap.add_argument("--index", default="benchmarks/maestro_segments.json",
                    help="segment index from `python -m training.index`")
    ap.add_argument("--audio-root", required=True,
                    help="where MAESTRO audio lives (a Kaggle mount, or local)")
    ap.add_argument("--midi-root", default=None,
                    help="defaults to --audio-root, which is how MAESTRO ships")
    ap.add_argument("--out", default="checkpoints",
                    help="checkpoints and train_log.jsonl land here")
    ap.add_argument("--deployable-name", default="ptify-note-pedal.pth",
                    help="name of the inference-loadable checkpoint")

    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help="segments per forward pass. MEMORY-BOUND: the model "
                         "runs four parallel CRNN branches over 1001x229 "
                         "features, so a T4 (16GB) OOMs above ~2 in fp32")
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="gradient accumulation; effective batch is "
                         "batch-size x accum-steps. Use this to keep a large "
                         "effective batch on a small GPU")
    ap.add_argument("--init-checkpoint", default=None, metavar="PATH",
                    help="start from THESE weights instead of ByteDance's "
                         "pretrained baseline. Use it to continue from a "
                         "previous PTify checkpoint -- without it every run "
                         "restarts from the baseline and discards what the "
                         "last one learned. NOT the same as --resume, which "
                         "continues an interrupted run from its step_N.pt")
    ap.add_argument("--soft-onset-boost", type=float, default=1.0,
                    metavar="X",
                    help="weight the ONSET loss per note by how quiet it is: "
                         "a silent note scores X, a maximal one 1.0, and loud "
                         "notes are never downweighted. 1.0 (default) is "
                         "bit-identical to every published run. Phase 27 "
                         "measured the onset head missing 38.3%% of pp notes "
                         "against 2.4%% of f notes, with pp+p making up 66.6%% "
                         "of all misses. NOT --loss-weights velocity=, which "
                         "scales the VELOCITY head (Phase 23, a negative "
                         "result) rather than reweighting onset detection")
    ap.add_argument("--loss-weights", default=None, metavar="SPEC",
                    help="per-head coefficients, e.g. 'velocity=0.1'. "
                         "Unnamed heads stay at 1.0, and the default is all "
                         "ones -- bit-identical to the unweighted sum every "
                         "published checkpoint was trained with. Velocity was "
                         "92.5%% of 16b's total loss and moved +0.1%%, so it "
                         "sets the gradient scale while contributing almost "
                         "no signal")
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_STEPS)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--no-amp", action="store_true",
                    help="disable mixed precision even on CUDA")

    # --- augmentation (Phase 16a) ---
    # Off by default: Phase 14.5's known-good configuration must stay exactly
    # reproducible, and a run that changes two things at once cannot attribute
    # either. Every one of these lands in the checkpoint via `vars(args)`.
    ap.add_argument("--augment", action="store_true",
                    help="continuous room/detune augmentation. Targets the "
                         "measured 18.3-point MAESTRO->MAPS generalisation "
                         "gap, of which 12.9 is room acoustics alone")
    ap.add_argument("--augment-seed", type=int, default=0,
                    help="per-segment seeds are hashed from this, the epoch "
                         "and the segment index; resume reproduces them "
                         "exactly without needing the RNG stream")
    ap.add_argument("--augment-clean-prob", type=float, default=0.2,
                    help="share of segments left untouched, so clean-audio "
                         "accuracy is not traded away for robustness")
    ap.add_argument("--augment-max-cents", type=float, default=50.0,
                    help="detune half-range. A 25-cent detune cost 14.1 F1 "
                         "points in the measured degradation curve, the "
                         "worst single factor")
    ap.add_argument("--augment-eq-prob", type=float, default=0.0,
                    help="mic-colouration probability. Default 0: it costs "
                         "22.6ms per segment, more than the rest of the chain "
                         "combined, for a factor absent from the curve")

    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="validation")
    ap.add_argument("--max-segments", type=int, default=None,
                    help="cap segments per split; for smoke runs")

    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--validate-every", type=int, default=0,
                    help="0 disables periodic validation")
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--save-every-steps", type=int, default=2000)
    ap.add_argument("--save-every-seconds", type=float, default=20 * 60,
                    help="wall-clock save interval; Kaggle kills at 12h, so "
                         "a step-only trigger can miss the deadline")
    ap.add_argument("--keep-checkpoints", type=int, default=2)
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="'auto' takes the newest in --out, or give a path")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return train(args)
    except Exception as exc:  # noqa: BLE001
        # torch.OutOfMemoryError only exists on newer torch, so match by name
        # rather than importing a symbol the local pin does not have.
        if "OutOfMemory" not in type(exc).__name__:
            raise
        effective = args.batch_size * max(1, args.accum_steps)
        halved = max(1, args.batch_size // 2)
        raise SystemExit(
            f"\nCUDA out of memory at batch-size {args.batch_size}.\n\n"
            f"This model is unusually memory-hungry for its parameter count: "
            f"it runs FOUR parallel CRNN branches (frame, onset, offset, "
            f"velocity), each holding activations over 1001 frames x 229 mel "
            f"bins for the backward pass.\n\n"
            f"Keep the effective batch and halve the memory:\n"
            f"    --batch-size {halved} --accum-steps "
            f"{max(1, effective // halved)}\n"
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
