# Security Policy

[English](./SECURITY.md) | [日本語](./SECURITY_ja.md)

## Supported Versions

IXV-LM has no release yet. Only the latest `main` is supported.

## Reporting a Vulnerability

If you discover a security vulnerability, **do NOT create a public GitHub Issue**.

### How to Report

1. **GitHub Security Advisory** (recommended):
   [open a private advisory](https://github.com/elvezjp/ixv-lm/security/advisories/new)
2. **Email**: info@elvez.co.jp

### What to Include

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact and severity
- Suggested fix or mitigation (optional)
- Contact information (optional)

## Notes on Use

- **Conversion server.** The input method's server (`ime/server.py`) listens on `127.0.0.1:18190` only and has no
  authentication. Do not expose it to other machines.
- **What the input method keeps.** Learned conversions (`history.tsv`) and the user dictionary are stored in
  `~/Library/Application Support/IXVKanji/`. They are plain text on your machine and contain what you typed. They are
  never sent anywhere. Delete them from the settings window or by removing the folder.
- **Model files.** Load weights only from sources you trust, such as
  [elvezjp/IXV-LM-Kanji-0.1B](https://huggingface.co/elvezjp/IXV-LM-Kanji-0.1B). The code loads safetensors and NumPy
  arrays without pickle.
- **Training data.** Data downloaded by `corpus/` scripts is untrusted input. Keep it outside the repository.
