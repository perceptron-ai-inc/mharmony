<!-- Modified by Perceptron AI, Inc.; derived from OpenAI Harmony. -->

# Stability and versioning

mharmony follows Semantic Versioning while it is in the `0.x` development
series. A release is a snapshot of a validated source commit, not a promise
that every internal experiment will be published.

## What `0.x` means

- A patch release, such as `0.1.1`, should preserve documented behavior and
  public signatures. It may fix output only where the previous behavior was a
  bug or a security issue.
- A minor release, such as `0.2.0`, may contain breaking API, schema, or token
  mapping changes.
- New media kinds and token groups are additive in the source API, but can
  still require a matching model tokenizer. They therefore ship in minor
  releases.

Pin an exact version when byte-identical rendering or checkpoint compatibility
matters. Applications should upgrade only after running their own golden
rendering fixtures.

## Compatibility-sensitive behavior

The following are part of a published release's compatibility surface:

- serialized `Conversation`, `Message`, and documented content envelopes;
- rendered token sequences and formatting-token mappings for published
  encodings;
- parser behavior for valid Harmony streams;
- documented Python and Rust entry points; and
- the generic media envelope and callback-selection contracts.

Token IDs are a model interface, not merely an implementation detail. A new
crate version is not automatically compatible with an existing model or
checkpoint even when the source API is unchanged. Consult
[Compatibility](compatibility.md) and record the mharmony version with model
artifacts.

## Public and implementation APIs

Names documented in this repository are public. Python names prefixed with
`_`, the compiled extension module, undocumented Rust modules or fields, and
serialized implementation details of callback results may change without a
deprecation period.

The `0.1.0` Python package includes PEP 561 metadata and a native-extension
stub. These describe the published Python API, but they do not imply that the
Python and Rust surfaces or their test suites are one-to-one mirrors.

## Deprecation

When practical, a public API scheduled for removal is documented for at least
one minor release. Security, correctness, format-integrity, and unpublished
experimental surfaces may require immediate change.
