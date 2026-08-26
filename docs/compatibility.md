<!-- Modified by Perceptron AI, Inc.; derived from OpenAI Harmony. -->

# Compatibility

mharmony connects application messages to a model tokenizer. Compatibility
therefore has three independent parts: package/runtime support, encoding data,
and the model or checkpoint that consumes the resulting tokens.

## Runtime support

| Surface | `0.1.x` support |
| --- | --- |
| Python | Python 3.10+; install a wheel matching the operating system and architecture, or build from source with Rust |
| Rust | Rust 1.87+ using the 2021 edition |
| WebAssembly | `HarmonyGptOss` core rendering/parsing only; npm packaging is not part of `0.1.0` |

Published release assets are the authority for the available wheel platforms.
Do not assume an unsupported platform will have a binary wheel.

## Encodings

| Encoding | Native Rust/Python | WebAssembly | Vocabulary requirement |
| --- | --- | --- | --- |
| `HarmonyGptOss` | Yes | Yes | The pinned `o200k_base` vocabulary is downloaded and hash-verified, or read from the configured local vocabulary directory |
| `Qwen3Harmony` | Yes | No | The pinned `Qwen/Qwen3-0.6B` vocabulary at revision `c1899de289a04d12100db370d81485cdf75e47ca`, or an explicitly authenticated alternative |
| `Qwen35Harmony` | Yes | No | The pinned `Qwen/Qwen3.5-0.8B` vocabulary at revision `2fc06364715b967f1860aea9cf38778875588b17`, or an explicitly authenticated alternative |

`HF_HOME` controls Hugging Face cache discovery. `TIKTOKEN_ENCODINGS_BASE`
selects a local directory for tiktoken vocabulary files, and
`TIKTOKEN_RS_CACHE_DIR` selects the download cache.

Qwen vocabulary files are not bundled. For Qwen3, set `QWEN3_VOCAB_PATH` to
the file, set `HF_HOME` to a populated Hugging Face cache, or provide `HOME`
for standard cache discovery. The loader does not fall back to a shared
temporary directory when none of those location sources exists. Qwen3.5 uses
`QWEN35_VOCAB_PATH` or the supported Hugging Face cache layout.

Both official Qwen vocabularies are pinned to their published SHA-256 digests
by the native loader. Qwen3 uses
`ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910`;
Qwen3.5 uses
`ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003`.
Set `QWEN3_VOCAB_SHA256` or `QWEN35_VOCAB_SHA256` only when intentionally
using a different authenticated checkpoint. Pointing coordinate groups may
reserve additional token IDs, so the model vocabulary and checkpoint
embedding size must match the reserved-token configuration used at training
time.

## Model compatibility

There is no universal claim that an mharmony release works with every model in
an encoding family. A model publisher must record all of the following:

- exact mharmony version;
- encoding name;
- tokenizer source and immutable revision;
- reserved token groups, sizes, and offsets; and
- any media-kind registrations used to construct training or inference
  streams; and
- the exact NumPy, Pillow, PyTorch, torchvision, and Transformers versions
  when the optional Qwen3.5 image adapter is used.

Training and inference must use the same values. A mismatch can produce valid
integers with the wrong meaning and may not fail loudly.

## Content and media support

Text, token, vector, pointing, system, developer, channel, and tool structures
are handled by the core. Binary media uses a generic envelope and host-side
registration: the core does not ship codecs, model-specific patchification, or
tensor preprocessing.

Image is the named public processor integration in `0.1.x`. Previously
published non-image wire mappings remain only for compatibility; this release
does not define corresponding source, processor-result, or tensor APIs. The
generic envelope can carry other kind names, but a host processor mapping alone
does not register their wire format. See the language guides for the concrete
surface.

## Source builds

A Python source build requires a Rust toolchain and maturin. Installing a wheel
does not. The optional `qwen35` extra installs the dependencies for the bundled
Qwen3.5 image adapter. Other host processors must declare their own media and
model dependencies.
