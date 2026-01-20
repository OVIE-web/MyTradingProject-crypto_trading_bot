# 🔐 Security

This project follows security-first development principles:

- **No secrets are committed** to the repository
- All credentials are loaded via environment variables
- JWT authentication is enforced for protected endpoints
- Strong typing and validation reduce runtime vulnerabilities
- Binance **Testnet-only** usage by default
- Database connections are validated before startup
- Logging replaces print statements to avoid leaking sensitive data

## Reporting Security Issues

If you discover a security vulnerability, please **do not open a public issue**.  
Instead, report it responsibly by contacting the repository owner directly.

Security issues are addressed with priority.
