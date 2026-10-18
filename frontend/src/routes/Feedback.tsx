/**
 * "Something wrong?" on the result page (plan A, 2026-10-18).
 *
 * The point is real data: every accuracy number PTify has is measured on
 * datasets, not on what people actually upload. Keeping the recording is an
 * explicit, unticked-by-default opt-in -- without it only the comment is
 * stored, and the audio is deleted on the normal one-hour timer.
 */

import { useState, type FormEvent } from "react";

import { ApiError, sendFeedback } from "../api/client";

export function Feedback({ jobId }: { jobId: string }) {
  const [open, setOpen] = useState(false);
  const [comment, setComment] = useState("");
  const [keep, setKeep] = useState(false);
  const [pending, setPending] = useState(false);
  const [sent, setSent] = useState<null | boolean>(null);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(ev: FormEvent) {
    ev.preventDefault();
    if (pending) return;
    setPending(true);
    setError(null);
    try {
      const res = await sendFeedback(jobId, comment, keep);
      setSent(res.kept);
    } catch (err) {
      setError((err as ApiError).message || "could not send feedback");
    } finally {
      setPending(false);
    }
  }

  if (sent !== null) {
    return (
      <p className="aside-note">
        Thanks — sent{sent ? ", with the recording" : ""}.
      </p>
    );
  }

  if (!open) {
    return (
      <button className="btn-link" onClick={() => setOpen(true)}>
        Report a problem with this transcription
      </button>
    );
  }

  return (
    <form className="feedback-form" onSubmit={onSubmit}>
      <label className="field">
        <span className="field-label">What went wrong?</span>
        <textarea
          className="field-input"
          rows={3}
          maxLength={2000}
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          placeholder="e.g. the left hand is missing in bar 12"
        />
      </label>
      <label className="feedback-keep">
        <input
          type="checkbox"
          checked={keep}
          onChange={(e) => setKeep(e.target.checked)}
        />{" "}
        Let PTify keep this recording and result to improve transcription.
        Otherwise it is deleted about an hour after the job finished.
      </label>
      {error && (
        <p className="form-error" role="alert">
          {error}
        </p>
      )}
      <button
        className="btn"
        type="submit"
        disabled={pending || (!comment.trim() && !keep)}
      >
        {pending ? "Sending…" : "Send"}
      </button>
    </form>
  );
}
