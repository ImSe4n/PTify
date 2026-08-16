import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies the API so the browser sees one origin. That is not
// only convenience: it means the app runs in dev exactly as it will when the
// SPA is served from the same host in Phase 10, and it keeps CORS out of the
// picture during development.
//
// api/settings.py already allows http://localhost:5173 explicitly, so calling
// the backend cross-origin also works if you prefer to skip the proxy -- but
// then the SSE stream and downloads have to carry the origin too, and a 429's
// Retry-After header stays unreadable (no expose_headers). The proxy avoids
// all of it.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/v1": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/healthz": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
