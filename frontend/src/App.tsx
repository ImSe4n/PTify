/**
 * The app shell: theme, navigation, and which screen is showing.
 *
 * Phase 7 replaced `useState<Screen>` with a hash router (see router.ts), so a
 * transcription now has a shareable URL and survives a refresh. This file is
 * back to being a pure route -> component switch.
 *
 * THE AUTH GUARD IS A RENDER-TIME CHECK, NOT AN EFFECT.
 *
 * Phase 6 force-navigated to the auth screen from an effect that had `screen`
 * in its own dependencies. Under a router that would also destroy the URL the
 * user was trying to reach. Returning <AuthScreen/> from the render path
 * instead preserves the hash, so signing in drops you exactly where you were
 * headed -- with no `?next=` parameter and no redirect dance.
 */

import { useEffect, useState } from "react";

import { useAuth } from "./auth/AuthContext";
import { AuthScreen } from "./routes/AuthScreen";
import { UploadScreen } from "./routes/UploadScreen";
import { JobScreen } from "./routes/JobScreen";
import { HistoryScreen } from "./routes/HistoryScreen";
import { SheetScreen } from "./routes/SheetScreen";
import { navigate, useRoute, type Route } from "./router";

type Theme = "light" | "dark";

const THEME_KEY = "ptify.theme";

export function App() {
  const { me, accountsEnabled, loading } = useAuth();
  const route = useRoute();
  const [theme, setTheme] = useState<Theme>(
    () => (localStorage.getItem(THEME_KEY) as Theme) || "light",
  );

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  // Signed out on a server that HAS accounts -> the auth screen. On a server
  // without accounts everything runs anonymously and there is nothing to show.
  const needsAuth = accountsEnabled && !me;

  // The one remaining navigation: someone who signed in while sitting ON the
  // sign-in URL has nowhere to be sent back to.
  useEffect(() => {
    if (!loading && !needsAuth && route.screen === "auth") {
      navigate({ screen: "upload", step: "file" }, { replace: true });
    }
  }, [loading, needsAuth, route.screen]);

  // The curtain stays mounted for one transition after the probe resolves, so
  // it can slide away rather than vanishing. Unmounting on `loading` alone is
  // why this used to be a flash of nothing.
  const [curtain, setCurtain] = useState(true);
  useEffect(() => {
    if (loading) return;
    const t = setTimeout(() => setCurtain(false), 700);
    return () => clearTimeout(t);
  }, [loading]);

  if (needsAuth) {
    return (
      <>
        <div className="grain" />
        {curtain && <Curtain lifting={!loading} />}
        <div className="app">
          <main className="app-main">
            <AuthScreen />
          </main>
        </div>
      </>
    );
  }

  return (
    <>
      <div className="grain" />
      {curtain && <Curtain lifting={!loading} />}
      <div className="app">
        <Header
          route={route}
          theme={theme}
          onToggleTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
        />

        <main className="app-main">
          {route.screen === "auth" && <AuthScreen />}
          {route.screen === "upload" && <UploadScreen step={route.step} />}
          {route.screen === "job" && <JobScreen jobId={route.jobId} />}
          {route.screen === "sheet" && (
            <SheetScreen jobId={route.jobId} page={route.page} />
          )}
          {route.screen === "history" && <HistoryScreen />}
        </main>
      </div>
    </>
  );
}

/** M1: the panel that covers the app until the auth probe resolves. */
function Curtain({ lifting }: { lifting: boolean }) {
  return (
    <div className={`boot${lifting ? " is-lifting" : ""}`} role="status">
      <span className="sr-only">Loading</span>
      <span className="boot-mark" aria-hidden="true">
        <span className="serif brand-word">PTify</span>
        <span className="brand-dot" />
      </span>
    </div>
  );
}

interface HeaderProps {
  route: Route;
  theme: Theme;
  onToggleTheme: () => void;
}

function Header({ route, theme, onToggleTheme }: HeaderProps) {
  const { me, signOut, accountsEnabled } = useAuth();
  const initials = me?.email ? me.email.slice(0, 2).toLowerCase() : null;

  return (
    <header className="app-header">
      <button className="brand" onClick={() => navigate({ screen: "upload", step: "file" })}>
        <span className="serif brand-word">PTify</span>
        <span className="brand-dot" aria-hidden="true" />
      </button>

      <nav className="app-nav">
        {(["upload", "history"] as const).map((s) => (
          <button
            key={s}
            className={`nav-link${route.screen === s ? " is-active" : ""}`}
            onClick={() => navigate(s === "upload" ? { screen: "upload", step: "file" } : { screen: "history" })}
          >
            {s === "upload" ? "New" : "Transcriptions"}
          </button>
        ))}
      </nav>

      <div className="app-header-end">
        <button
          className="theme-toggle"
          onClick={onToggleTheme}
          aria-label={`Switch to ${theme === "dark" ? "paper" : "night"} theme`}
        >
          <span aria-hidden="true">{theme === "dark" ? "☾" : "☀"}</span>
          <span>{theme === "dark" ? "Night" : "Paper"}</span>
        </button>

        {accountsEnabled && me && (
          <>
            <span className="mono avatar" title={me.email ?? undefined}>
              {initials}
            </span>
            <button className="btn-link" onClick={signOut}>
              Sign out
            </button>
          </>
        )}
      </div>
    </header>
  );
}
