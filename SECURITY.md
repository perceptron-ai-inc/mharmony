<!-- Modified by Perceptron AI, Inc.; derived from OpenAI Harmony. -->

# Security Policy

## Supported versions

Security fixes are made on the latest `0.x` release line. Because mharmony is
pre-1.0, older minor releases do not receive backports unless a release note
explicitly says otherwise.

| Version | Supported |
| --- | --- |
| Latest `0.x` release | Yes |
| Older releases | No |

## Reporting a vulnerability

Please do not open a public issue or pull request for a suspected
vulnerability. Use GitHub's private
[Report a vulnerability](https://github.com/perceptron-ai-inc/mharmony/security/advisories/new)
form instead.

Include the affected version or commit, affected API, realistic impact,
preconditions, and a minimal reproducer. Maintainers will coordinate a fix and
disclosure through the private advisory. Please allow a reasonable remediation
window before publishing details.

## Security boundary

The following are in scope when they cross a real trust boundary:

- memory-safety, code-execution, path-traversal, or unintended network-access
  flaws reachable through the published Rust or Python API;
- parser or tokenizer bugs that allow untrusted input to bypass a documented
  validation or isolation guarantee;
- integrity failures in downloaded tokenizer data or published release
  artifacts; and
- vulnerabilities in this repository's release or mirror automation.

Host-supplied media processors execute application code and are trusted by
design. Model behavior, prompt injection, intentionally loading untrusted
processor code, local resource exhaustion without a remote or multi-tenant
impact, and vulnerabilities in third-party dependencies are normally out of
scope and should be reported to their owners.

Applications must still bound input sizes. Base64 media, token streams, and
deeply nested JSON can consume significant memory before a host processor or
model sees them.

Treat tokenizer vocabulary files as executable model configuration: obtain
them from an authenticated source and pin their SHA-256 digest. The Qwen3.5
loader authenticates the official vocabulary by default and supports an
explicit `QWEN35_VOCAB_SHA256` pin for alternate checkpoints. Set
`MHARMONY_REQUIRE_QWEN_VOCAB_SHA256=1` in strict deployments so other Qwen
vocabularies also require an explicit digest.
