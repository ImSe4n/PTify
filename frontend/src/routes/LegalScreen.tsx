/**
 * Privacy policy and terms.
 *
 * WHY THESE EXIST AND WHY THEY ARE SHORT
 *
 * A site that takes accounts and file uploads with no privacy policy and no
 * terms is a tell in itself, and more to the point it leaves a real user with
 * no way to find out what happens to a recording they upload.
 *
 * WHY EVERY CLAIM HERE IS CHECKABLE
 *
 * The rest of this project refuses to publish a number it has not measured.
 * The same rule applies to prose about data handling: every statement below
 * names the code that implements it, so it can be verified rather than
 * believed, and so it breaks visibly if the behaviour changes.
 *
 *   retention        api/app.py janitor + PTIFY_JOB_TTL_SECONDS (1h default)
 *   passwords        api/users.py PBKDF2_ROUNDS = 600_000
 *   job ownership    api/routes/jobs.py -- another principal's job is a 404
 *   remote inference transcriber/remote.py -- audio leaves the server ONLY
 *                    when the remote engine is chosen
 *
 * The remote-engine paragraph is the one that matters most and is the reason
 * this page was written during Phase 9 rather than later: as of that phase a
 * recording can be sent to a third-party GPU host, and a user choosing that
 * engine is entitled to know before they choose it.
 */

import { navigate } from "../router";

type Which = "privacy" | "terms";

export default function LegalScreen({ which }: { which: Which }) {
  return (
    <div className="legal-screen">
      <article className="legal">
        <p className="eyebrow mono">
          {which === "privacy" ? "PRIVACY" : "TERMS"}
        </p>
        <h1 className="legal-title">
          {which === "privacy" ? "What happens to your audio" : "Terms of use"}
        </h1>

        {which === "privacy" ? <Privacy /> : <Terms />}

        <p className="legal-foot">
          <a
            href={which === "privacy" ? "#/terms" : "#/privacy"}
            onClick={(e) => {
              e.preventDefault();
              navigate({
                screen: "legal",
                which: which === "privacy" ? "terms" : "privacy",
              });
            }}
          >
            {which === "privacy" ? "Terms of use" : "Privacy"}
          </a>
          <span aria-hidden="true"> · </span>
          <a
            href="#/"
            onClick={(e) => {
              e.preventDefault();
              navigate({ screen: "upload", step: "file" });
            }}
          >
            Back to the app
          </a>
        </p>
      </article>
    </div>
  );
}

function Privacy() {
  return (
    <>
      <p className="legal-lede">
        Short version: your recordings are deleted about an hour after they are
        transcribed, nobody else can read them, and they are never used to train
        anything. The only exception is a recording you explicitly choose to
        share when reporting a problem.
      </p>

      <h2>What is stored</h2>
      <p>
        The audio file you upload, the transcription produced from it, and the
        artifacts you asked for (MIDI, MusicXML, PDF, SVG). Alongside those: the
        account you created, and the job&rsquo;s metadata such as which engine
        ran and when. A short log of how each job ended (finished or failed,
        an error code, how long it took) is kept after the job is deleted so
        failures can be fixed; it contains no audio and no notes.
      </p>

      <h2>How long it is kept</h2>
      <p>
        Uploads and artifacts are deleted on a timer, one hour by default. This
        is not a promise made in prose: a background task sweeps expired jobs
        and removes their files from disk, and the same expiry is why a
        transcription link stops working after a while. Your account and its
        job list persist until you ask for them to be removed.
      </p>

      <h2>Who can read it</h2>
      <p>
        Only you. Jobs are owned by the account that created them, and a request
        for someone else&rsquo;s job returns &ldquo;not found&rdquo; rather than
        &ldquo;forbidden&rdquo;, so job identifiers cannot be used to discover
        that other people&rsquo;s work exists. Passwords are stored as PBKDF2
        hashes at 600,000 rounds with a per-user salt, never as recoverable
        text.
      </p>

      <h2>Where the audio is processed</h2>
      <p>
        Your audio is sent to PTify&rsquo;s own transcription server, a GPU
        host run on Modal, and the notes are sent back. That server processes
        the file to produce the transcription and does not keep it. The site
        and its storage also run on Modal; the web pages are served by Vercel,
        which never receives your audio.
      </p>

      <h2>Training</h2>
      <p>
        Your recordings are not used to train or fine-tune any model. The model
        was trained on the MAESTRO research dataset, and the published accuracy
        figures come from public test sets, not from user uploads.
      </p>

      <h2>Reporting a problem</h2>
      <p>
        Under a result you can report what went wrong. Your comment is kept.
        The recording and its result are kept <em>only</em> if you tick the box
        to share them; they are then used to test and improve transcription,
        and are removed if you delete your account.
      </p>

      <h2>Analytics</h2>
      <p>
        There is no analytics script, no advertising network, and no third-party
        tracker on this site. The sound samples used for playback are fetched
        from a public CDN; if that host is unreachable the app falls back to a
        synthesised piano.
      </p>

      <h2>Removal</h2>
      <p>
        Uploads and artifacts remove themselves on the retention timer above, so
        nothing needs to be done to have a recording deleted. To remove
        everything at once, use &ldquo;Delete account&rdquo; at the top of the
        page: it deletes your account, every transcription, and any recordings
        you shared with a problem report, immediately and permanently.
      </p>
    </>
  );
}

function Terms() {
  return (
    <>
      <p className="legal-lede">
        Short version: upload music you have the right to upload, expect the
        output to be imperfect, and do not treat this as a guaranteed service.
      </p>

      <h2>What this is</h2>
      <p>
        PTify transcribes piano recordings into notes and engraved sheet music.
        It is a research project shipped as a working product, not a commercial
        service with a support contract.
      </p>

      <h2>What you upload</h2>
      <p>
        Upload only recordings you own or otherwise have the right to process.
        You keep whatever rights you had in the recording; uploading it grants
        no ownership here beyond what is needed to transcribe it and give you
        the result back.
      </p>

      <h2>Accuracy</h2>
      <p>
        Transcription is probabilistic and the output contains mistakes. On
        clean studio recordings the current model scores about 0.956 onset F1
        (roughly one note in twenty wrong); on unfamiliar pianos and rooms it
        is 0.859 on MAPS (roughly one in seven), and note durations are
        less reliable than note starts, especially in heavily pedalled music.
        The app marks estimated note lengths rather than hiding them. Check the
        result before relying on it for anything that matters, such as
        performance or publication.
      </p>

      <h2>Availability</h2>
      <p>
        There is no uptime guarantee. Jobs can fail, and artifacts expire on the
        retention timer described in the privacy page. Keep your own copy of
        anything you want to keep.
      </p>

      <h2>Liability</h2>
      <p>
        The service is provided as is, without warranty. It is not liable for
        losses arising from use of the output, including an incorrect
        transcription.
      </p>
    </>
  );
}
