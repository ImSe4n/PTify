/**
 * Authentication state.
 *
 * THREE BEHAVIOURS THAT COME FROM THE API AND ARE NOT NEGOTIABLE
 *
 * 1. Accounts may not exist. `/v1/auth/*` is registered only when the server
 *    has both PTIFY_JWT_SECRET and PTIFY_DB_PATH, otherwise every auth path is
 *    a plain 404 (api/app.py:74-77). That is an honest "this server does not do
 *    accounts", so we probe once and run anonymously rather than showing a
 *    login form that cannot work.
 *
 * 2. There is NO refresh endpoint and no revocation. A token is valid until
 *    `exp` and then it is not. On a 401 the only correct move is to clear the
 *    token and send the user back to sign in.
 *
 * 3. Signup and login cost ~600ms of PBKDF2 (600,000 rounds, measured at 0.74s
 *    against a live server in Phase 5.5). That is deliberate work, not slowness
 *    to optimise away -- but it does mean the form needs a pending state or it
 *    feels broken.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import * as api from "../api/client";
import type { MeOut } from "../api/types";

const TOKEN_KEY = "ptify.token";

interface AuthValue {
  token: string | null;
  me: MeOut | null;
  /** False when this deployment has no accounts at all. */
  accountsEnabled: boolean;
  /** True until the initial probe settles, so screens can avoid flashing. */
  loading: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  signOut: () => void;
}

const AuthContext = createContext<AuthValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() =>
    localStorage.getItem(TOKEN_KEY),
  );
  const [me, setMe] = useState<MeOut | null>(null);
  const [accountsEnabled, setAccountsEnabled] = useState(true);
  const [loading, setLoading] = useState(true);

  // Registered before any request goes out so the client picks up the token.
  useEffect(() => {
    api.setTokenGetter(() => token);
  }, [token]);

  const signOut = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    setToken(null);
    setMe(null);
  }, []);

  // One probe on boot: does this server do accounts, and is our token still
  // good? A 404 means no accounts; a 401 means the stored token expired.
  useEffect(() => {
    let cancelled = false;
    api.setTokenGetter(() => token);

    (async () => {
      try {
        const who = await api.getMe();
        if (!cancelled) {
          // A 200 is NOT proof of being signed in. When the server has no
          // PTIFY_API_KEY, an unauthenticated request is a valid ANONYMOUS
          // principal and /me answers 200 with kind:"anonymous". Treating that
          // as signed-in showed the app to someone who owns no jobs, so the
          // job list came back empty with nothing explaining why.
          //
          // The route EXISTING at all is what proves accounts are configured
          // (it is only registered with a secret + a database), so a 200 sets
          // accountsEnabled regardless of which principal answered.
          setAccountsEnabled(true);
          setMe(who.kind === "user" ? who : null);
        }
      } catch (err) {
        if (cancelled) return;
        const e = err as api.ApiError;
        if (e.status === 404) {
          setAccountsEnabled(false);
          setMe(null);
        } else if (e.status === 401) {
          localStorage.removeItem(TOKEN_KEY);
          setToken(null);
          setMe(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
    // Deliberately once on mount: re-probing on every token change would fire
    // a second /me straight after sign-in, which already returns the identity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const adopt = useCallback((accessToken: string, email: string, userId: string) => {
    localStorage.setItem(TOKEN_KEY, accessToken);
    api.setTokenGetter(() => accessToken);
    setToken(accessToken);
    setMe({ id: `user:${userId}`, kind: "user", email });
    setAccountsEnabled(true);
  }, []);

  const signIn = useCallback(
    async (email: string, password: string) => {
      const res = await api.login(email, password);
      adopt(res.access_token, res.email, res.user_id);
    },
    [adopt],
  );

  const signUp = useCallback(
    async (email: string, password: string) => {
      // Signup returns a token, so there is no second round trip.
      const res = await api.signup(email, password);
      adopt(res.access_token, res.email, res.user_id);
    },
    [adopt],
  );

  const value = useMemo<AuthValue>(
    () => ({ token, me, accountsEnabled, loading, signIn, signUp, signOut }),
    [token, me, accountsEnabled, loading, signIn, signUp, signOut],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
