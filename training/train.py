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
)
from .dataset import SegmentDataset, collate
from .index import read_index, segments_from_index
from .losses import compute_losses
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


def build_dataloader(args, split: str, *, shuffle: bool):
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
                             midi_root=args.midi_root or args.audio_root)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        collate_fn=collate,
        drop_last=True,
        # Workers re-import the package per process; keeping them alive avoids
        # paying soxr's ~2-6s lazy init on every epoch boundary.
        persistent_workers=args.workers > 0,
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

    print(f"Loading pretrained weights (device={device}) ...")
    note_model, pedal_state = load_pretrained(device)
    note_model.train()
    print(f"  {trainable_parameters(note_model):,} trainable parameters")
    print("  pedal model loaded and FROZEN (re-attached unchanged on save)")

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
    if args.resume:
        path = (find_latest(out_dir) if args.resume == "auto"
                else Path(args.resume))
        if path and Path(path).exists():
            info = load_training_state(
                path, note_model=note_model, optimizer=optimizer,
                scaler=scaler, device=device,
            )
            start_step = info["step"]
            print(f"Resumed from {path} at step {start_step}")
        elif args.resume != "auto":
            raise FileNotFoundError(f"--resume {args.resume} does not exist")
        else:
            print("No checkpoint found; starting fresh")

    loader = build_dataloader(args, args.train_split, shuffle=True)
    val_loader = (build_dataloader(args, args.val_split, shuffle=False)
                  if args.validate_every else None)
    print(f"{len(loader.dataset):,} training segments, "
          f"batch {args.batch_size}, {args.workers} workers")

    trigger = SaveTrigger(args.save_every_steps, args.save_every_seconds)
    step = start_step
    started = time.time()
    epoch = 0

    accum = max(1, args.accum_steps)

    while step < args.steps:
        epoch += 1
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
                losses = compute_losses(note_model(batch["waveform"]), batch)

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
                logger.log({"step": step, **metrics})
                print(f"  step {step:>6} VAL total "
                      f"{metrics['val_total']:.4f}")

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
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_STEPS)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--no-amp", action="store_true",
                    help="disable mixed precision even on CUDA")

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
