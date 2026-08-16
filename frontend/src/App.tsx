/**
 * The app shell: theme, navigation, and which screen is showing.
 *
 * Screen state is held here rather than in a router because the flow is a
 * pipeline (upload -> waiting -> result) whose steps carry a job id, and
 * Phase 6 has no deep-linking requirement. A router is a Phase 7 concern once
 * a transcription has a shareable URL.
 */

import { useCallback, useEffect, useState } from "react";

import { useAuth } from "./auth/AuthContext";
import { AuthScreen } from "./routes/AuthScreen";
import { UploadScreen } from "./routes/UploadScreen";
import { WaitingScreen } from "./routes/WaitingScreen";
import { ResultScreen } from "./routes/ResultScreen";
import { HistoryScreen } from "./routes/HistoryScreen";
import { SheetScreen } from "./routes/SheetScreen";

export type Screen = "auth" | "upload" | "waiting" | "result" | "sheet" | "history";
type Theme = "light" | "dark";

const THEME_KEY = "ptify.theme";

export function App() {
  const { me, accountsEnabled, loading } = useAuth();
  const [screen, setScreen] = useState<Screen>("upload");
  const [jobId, setJobId] = useState<string | null>(null);
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

  useEffect(() => {
    if (loading) return;
    if (needsAuth) setScreen("auth");
    else if (screen === "auth") setScreen("upload");
  }, [loading, needsAuth, screen]);

  const openJob = useCallback((id: string, target: Screen = "waiting") => {
    setJobId(id);
    setScreen(target);
  }, []);

  if (loading) {
    return (
      <>
        <div className="grain" />
        <div className="boot" role="status">
          <span className="sr-only">Loading</span>
        </div>
      </>
    );
  }

  return (
    <>
      <div className="grain" />
      <div className="app">
        {screen !== "auth" && (
          <Header
            screen={screen}
            theme={theme}
            onToggleTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
            onNavigate={setScreen}
          />
        )}

        <main className="app-main">
          {screen === "auth" && <AuthScreen />}
          {screen === "upload" && <UploadScreen onSubmitted={(id) => openJob(id)} />}
          {screen === "waiting" && jobId && (
            <WaitingScreen
              jobId={jobId}
              onDone={() => setScreen("result")}
              onLeave={() => setScreen("history")}
            />
          )}
          {screen === "result" && jobId && (
            <ResultScreen jobId={jobId} onOpenSheet={() => setScreen("sheet")} />
          )}
          {screen === "sheet" && jobId && (
            <SheetScreen jobId={jobId} onBack={() => setScreen("result")} />
          )}
          {screen === "history" && (
            <HistoryScreen
              onOpen={(id, state) =>
                openJob(id, state === "succeeded" ? "result" : "waiting")
              }
              onNew={() => setScreen("upload")}
            />
          )}
        </main>
      </div>
    </>
  );
}

interface HeaderProps {
  screen: Screen;
  theme: Theme;
  onToggleTheme: () => void;
  onNavigate: (s: Screen) => void;
}

function Header({ screen, theme, onToggleTheme, onNavigate }: HeaderProps) {
  const { me, signOut, accountsEnabled } = useAuth();
  const initials = me?.email ? me.email.slice(0, 2).toLowerCase() : null;

  return (
    <header className="app-header">
      <button className="brand" onClick={() => onNavigate("upload")}>
        <span className="serif brand-word">PTify</span>
        <span className="brand-dot" aria-hidden="true" />
      </button>

      <nav className="app-nav">
        {(["upload", "history"] as const).map((s) => (
          <button
            key={s}
            className={`nav-link${screen === s ? " is-active" : ""}`}
            onClick={() => onNavigate(s)}
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
