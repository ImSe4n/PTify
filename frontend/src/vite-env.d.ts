/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * Absolute base for API calls. Normally UNSET: the Vite dev server proxies
   * /v1 to the backend, and in production the SPA is served from the same
   * origin. Set it only when pointing a local build at a remote API -- and
   * remember `api/settings.py: cors_origins` must then list this app's origin.
   */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
