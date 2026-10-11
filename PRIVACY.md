# Privacy & Local Data Policy — EcoLens AI

EcoLens AI is designed to run **entirely on your machine**. There is no account,
no cloud upload, and no third-party analytics in the core workflow.

## What is stored, and where

| Data | Location | Notes |
|---|---|---|
| Uploaded images | `storage/uploads/<YYYY/MM/DD>/<random>.ext` | Generated filenames; never public. |
| Grad-CAM overlays | `storage/explanations/<YYYY/MM/DD>/<random>.png` | Optional; only if generated. |
| Analysis metadata | `storage/ecolens.db` (SQLite) | Prediction, scores, uncertainty, timing, model info. |
| Guidance snapshot | `storage/ecolens.db` | The guidance shown at analysis time. |
| Logs | `storage/ecolens.log` | Structured JSON lines. |

Overriding locations: `ECOLENS_STORAGE_DIR`, `ECOLENS_DB_PATH` (see `.env.example`).

## What is **not** collected

- No names, accounts, emails, or login.
- No IP addresses or device identifiers stored by us.
- No images or metadata sent to any external service by the core workflow.
- No telemetry. Streamlit usage stats are disabled in `.streamlit/config.toml`.

## Network access

The only network access is the **one-time download of pretrained weights** from
`download.pytorch.org`. Set `ECOLENS_ALLOW_DOWNLOAD=0` and pre-cache weights for a
fully offline deployment, after which the app makes no outbound requests.

## Your controls

- **Delete** any analysis from the History page (removes DB row + stored files).
- **Reset all data:** `python scripts/reset_database.py --yes`.
- Use a custom storage directory for isolation or ephemeral containers.

## Security posture (honest)

Implemented baseline controls:
- Extension + size checks and **real image decoding** before processing.
- Generated filenames; **path-traversal protection** on all file reads/deletes.
- Uploaded files are never executed or served from a public directory.
- Secrets are not committed; `.gitignore` excludes `.env`, DB, uploads, weights, caches.
- Graceful exception handling; broad errors are caught so the UI can recover.

We do **not** claim the application is "fully secure" or production-certified.
These are sensible safeguards for a local, single-user tool, not a hardened
multi-tenant service. If you deploy it publicly, add authentication, per-user
isolation, TLS, rate limiting, and storage quotas.

## Retention

Data persists until you delete it. There is no automatic expiry or cleanup of
user data during normal startup. `init_db()` only creates/syncs schema.
