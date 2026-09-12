# Security Policy

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's private reporting:
**Security tab, then "Report a vulnerability"**, or open a draft security advisory on this repository.

You will get an acknowledgement within 7 days and a plan or fix within 30 days for confirmed issues.

## Supported versions

Pre-1.0: only the latest tagged release and `main` receive fixes.

## Security design notes

- Secrets come from environment variables only. Production refuses to boot with a placeholder JWT secret
  or the shipped bootstrap admin credentials.
- Access control is enforced inside retrieval so restricted chunks never reach the model context or a
  citation.
- Uploaded files are size and type limited and parsed with libraries that do not execute embedded content.
- API keys are stored hashed, scoped to collections and strategies, and rate limited.
- Telemetry is off by default. Nothing leaves your deployment unless you configure an export.
