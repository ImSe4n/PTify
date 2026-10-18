
## 2026-10-09 Deploys
**Mistake**: Redeployed the Modal host (`modal deploy hosting/modal/app.py`) to pick up a
changed token secret without exporting `PTIFY_HOST_ENGINE=ptify`. The engine is baked in
at deploy time and defaults to `bytedance`, so the live host silently switched models, and
a smoke result was reported as PTify's when it was ByteDance's.
**Correction**: Redeploy with `PTIFY_HOST_ENGINE=ptify`, then run `tools.crosscheck_remote`
(writing `--json` under `var/`, never over the committed benchmark) before trusting output.
**Rule**: Before redeploying anything, read the deploy-time environment it depends on
(HANDOFF records it). After any deploy, verify what is served before using its output.
