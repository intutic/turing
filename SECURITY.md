# Security Policy & Vulnerability Disclosure

This policy covers the Turing Engine Python package, its native C++ extension and `turing-cli`, and the HTTP servers they run.

---

## 🛡️ Supported Versions

We provide active security patches, bug fixes, and vulnerability remediation for the following release series:

| Version Series | Release Date | Supported Status |
| :--- | :--- | :---: |
| **1.7.x and later** | November 2026 – Present | 🟢 **Active Security Support** |
| **< 1.7.0** | August – November 2026 | 🔴 **End of Life** |

---

## 🔑 Deploying the Server

Authentication is off by default, and `turing serve` binds `0.0.0.0` by default: it answers every request that
reaches its port unless it is started with `--api-key` (or the `TURING_API_KEY` environment variable), which then
requires `Authorization: Bearer <key>` or `x-api-key: <key>` on every endpoint except `/health`. Bind it to
`127.0.0.1` (`--host 127.0.0.1`), or put it behind a gateway that authenticates callers, before exposing it to a
network. `/api/pull` downloads any HuggingFace repository a caller names into the server's HuggingFace cache, and
`--prefill-worker` accepts unauthenticated TCP connections on `--kv-port`; keep both off untrusted networks.
`turing-cli serve` binds `127.0.0.1` by default and has no authentication.

---

## 🔒 Reporting a Vulnerability

If you discover a security vulnerability, side-channel attack, memory safety defect in our C++20 AVX2 native extensions (`csrc/`), or an authentication/authorization bypass in the serving endpoints:

### 1. Private Vulnerability Reporting (Preferred)
Please submit a report directly via **[GitHub Private Vulnerability Reporting](https://github.com/intutic/turing/security/advisories/new)**. This keeps the disclosure private between you and our security maintainers until a fix is released.

### 2. Direct Security Contact
Alternatively, you can email our security response team directly:
- **Email**: `security@intutic.ai` / `support@intutic.ai`
- **Lead Maintainer**: Ishan Gupta (`ishangupta.ds@gmail.com`)

### 📋 What to Include in Your Report
To help us triage and resolve the issue quickly, please provide:
1. **Description**: A clear overview of the potential vulnerability.
2. **Impact**: Which components are affected (e.g. `csrc/`, `/v1/chat/completions`, the `.tgate` or GGUF readers).
3. **Reproduction Steps / PoC**: Minimal reproducible Python/C++ script or payload demonstrating the issue.
4. **Environment**: OS (Linux, macOS, Windows), Python version, and CUDA/AVX2 hardware configuration.

---

## ⏱️ Response Timelines & Policy

- **Initial Response**: Within **24 hours** acknowledging receipt of your report.
- **Triage & Assessment**: Within **48 hours** with a severity rating (CVSS score) and reproduction confirmation.
- **Remediation & Patch**: Target fix released within **7 to 14 business days** depending on severity.
- **Public Disclosure**: Coordinated disclosure after the patch has been published and validated.

---

## 🚫 Non-Eligible Issues
- Social engineering, phishing, or physical attacks.
- Denial of Service (DoS) via resource exhaustion on unauthenticated local debug endpoints without production gateway deployment.
- Issues in third-party upstream dependencies (please report directly to the respective upstream project, e.g., PyTorch, FastAPI, Triton).
