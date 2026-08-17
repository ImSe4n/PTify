/**
 * Sign in / create account.
 *
 * The editorial half is deliberately asymmetric (the mockup used a symmetric
 * 1fr/1fr split, which is part of what read as machine-made) and carries the
 * display-scale type. The form half stays quiet: a form is an instrument.
 *
 * The pending state is not decoration -- PBKDF2 at 600,000 rounds costs ~700ms,
 * measured against a live server. Without it the button feels broken.
 */

import { useState, type FormEvent } from "react";

import { ApiError } from "../api/client";
import { Reveal } from "../ui/Reveal";
import { useAuth } from "../auth/AuthContext";

export function AuthScreen() {
  const { signIn, signUp } = useAuth();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isLogin = mode === "login";

  async function onSubmit(ev: FormEvent) {
    ev.preventDefault();
    if (pending) return;
    setPending(true);
    setError(null);
    try {
      if (isLogin) await signIn(email, password);
      else await signUp(email, password);
    } catch (err) {
      const e = err as ApiError;
      // The API never says which half of a login was wrong -- that is a
      // deliberate user-enumeration defence, so we do not embellish it either.
      setError(e.message || "something went wrong");
      setPending(false);
    }
  }

  return (
    <div className="auth enter-stagger">
      <section className="auth-pitch" style={{ "--i": 0 } as React.CSSProperties}>
        <p className="eyebrow">recording → midi → engraved score</p>

        {/* Pre-split, because this line carries markup: a naive whitespace
            split would lose both the non-breaking space and the emphasis. */}
        <Reveal
          as="h1"
          className="display"
          label="Notes you can read, edit, and trust."
        >
          {[
            "Notes",
            "you",
            "can",
            "read,",
            "edit,",
            <span key="trust">
              and&nbsp;<em>trust</em>.
            </span>,
          ]}
        </Reveal>

        <div className="auth-pitch-body">
          <p className="prose">
            Transcription is probabilistic. PTify shows you not just the result,
            but how much of it was measured rather than estimated — so you always
            know which rhythms to believe.
          </p>

          <dl className="auth-facts">
            <div>
              <dt className="mono">0.840</dt>
              <dd>onset F1 on MAPS — a piano and room the model never trained on</dd>
            </div>
            <div>
              <dt className="mono">+5.3</dt>
              <dd>points over the ByteDance baseline, on 14 of 14 tracks</dd>
            </div>
          </dl>
        </div>
      </section>

      <section className="auth-form-side" style={{ "--i": 1 } as React.CSSProperties}>
        <form className="auth-form" onSubmit={onSubmit}>
          <div className="brand brand-lg">
            <span className="serif brand-word">PTify</span>
            <span className="brand-dot" aria-hidden="true" />
          </div>

          <h2 className="h1 auth-title">{isLogin ? "Sign in" : "Create account"}</h2>
          <p className="auth-sub">
            {isLogin
              ? "Your transcriptions are waiting."
              : "Free while in beta. Jobs are private to your account."}
          </p>

          <label className="field">
            <span className="field-label">Email</span>
            <input
              className="field-input"
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
            />
          </label>

          <label className="field">
            <span className="field-label">Password</span>
            <input
              className="field-input"
              type="password"
              autoComplete={isLogin ? "current-password" : "new-password"}
              required
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={isLogin ? "" : "at least 8 characters"}
            />
          </label>

          {error && (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}

          <button className="btn auth-submit" type="submit" disabled={pending}>
            {pending
              ? isLogin
                ? "Signing in…"
                : "Creating account…"
              : isLogin
                ? "Sign in"
                : "Create account"}
          </button>

          {pending && (
            <p className="auth-note mono">
              hashing your password — 600,000 rounds, by design
            </p>
          )}

          <p className="auth-switch">
            {isLogin ? "No account yet?" : "Already have one?"}{" "}
            <button
              type="button"
              className="btn-link"
              onClick={() => {
                setMode(isLogin ? "signup" : "login");
                setError(null);
              }}
            >
              {isLogin ? "Create one" : "Sign in"}
            </button>
          </p>
        </form>
      </section>
    </div>
  );
}
