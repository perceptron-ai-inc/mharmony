<!-- Modified by Perceptron AI, Inc.; derived from OpenAI Harmony. -->

# Rust guide

The `mharmony` crate is the canonical implementation of Harmony rendering,
parsing, tokenization, pointing, and the generic media envelope.

```toml
[dependencies]
mharmony = "0.1"
```

The crate uses Rust 2021 and requires Rust 1.87 or newer for `0.1.x`; see
[Compatibility](compatibility.md).

## Render a conversation

```rust
use mharmony::chat::{Conversation, Message, Role, SystemContent};
use mharmony::{load_harmony_encoding, HarmonyEncodingName};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;
    let conversation = Conversation::from_messages([
        Message::from_role_and_content(Role::System, SystemContent::new()),
        Message::from_role_and_content(Role::User, "What is 2 + 2?"),
    ]);

    let tokens = encoding.render_conversation_for_completion(
        &conversation,
        Role::Assistant,
        None,
    )?;
    println!("{tokens:?}");
    Ok(())
}
```

Use `render_conversation_for_training` for supervised targets,
`render_conversation` without a next-role prefix, and `render` for one message.

## Public modules

| Module | Purpose |
| --- | --- |
| `chat` | Conversation, message, content, tool, and pointing structures |
| `encoding` | Rendering, parsing, completion streaming, and configuration |
| `registry` | Named encodings and reserved-token-group configuration |
| `channels` | Canonical channel names and validation |
| `stream` | Generic event streams and deterministic scheduling |
| `tiktoken_ext` | Vocabulary loading and tokenizer integration |

Common encoding types and loaders are re-exported at the crate root.

## Content model

`Content` contains the public, cross-language conversation representation:

```rust
pub enum Content {
    Text(TextContent),
    Tokens(TokensContent),
    Vector(VectorContent),
    Media(MediaContent),
    Pointing(PointingContent),
    PointCollection(PointCollectionContent),
    SystemContent(SystemContent),
    DeveloperContent(DeveloperContent),
}
```

`Message::from_role_and_content` constructs a single-content message.
`Message::from_role_and_contents` accepts multiple `Content` values. Builder
methods add channels, recipients, and content types.

## Generic media boundary

The Rust type system does not expose separate modality variants. All binary
media uses one envelope:

```rust
use mharmony::chat::{MediaContent, MediaRef};

let media = MediaContent {
    kind: "image".to_owned(),
    media_ref: MediaRef {
        id: "camera.front".to_owned(),
        mime: Some("image/jpeg".to_owned()),
        bytes_b64: Some(encoded_bytes),
        metadata: None,
    },
    metadata: None,
    source: None,
};
```

The serialized representation is:

```json
{
  "type": "media",
  "kind": "image",
  "media_ref": {"id": "camera.front", "mime": "image/jpeg", "bytes_b64": "..."}
}
```

The core owns only modality-neutral values: `kind`, `MediaRef`, opaque JSON
`metadata`, and optional opaque JSON `source`. Kind-specific validation,
decoding, preprocessing, and tensor contracts belong to the host.

The Python binding supplies the `media_processors` callback registry used by
applications that render model input tensors. Native Rust applications should
perform those transforms in their host layer. `image` is the processor
integration shipped in `0.1.x`. A legacy `video` token/event wire mapping is
retained without a typed source, result, or tensor contract. The generic
envelope can represent other names; register their formatting token and stream
modality through `HarmonyEncodingOptions.media_kinds`. A processor callback
alone does not create a wire registration.

## Pointing

`PointingContent` represents point, box, polygon, and temporal clip markup.
`PointCollectionContent` represents collections and tracks. Their HTML-like
wire format is rendered and parsed by the Rust core, including stable streaming
views that do not expose incomplete point payloads.

Use `format_markup_open_tag` only when constructing a split pointing payload;
normal applications should construct the typed content values and let the
encoding render them.

## Encodings

```rust
use mharmony::{
    load_harmony_encoding_with_options,
    HarmonyEncodingName,
    HarmonyEncodingOptions,
    MediaKindRegistrationConfig,
    ReservedTokenGroupConfig,
};

let options = HarmonyEncodingOptions {
    reserved_token_groups: vec![ReservedTokenGroupConfig {
        name: Some("coord".to_owned()),
        tokenizer: None,
        size: 4096,
        offset: None,
    }],
    media_kinds: vec![MediaKindRegistrationConfig {
        kind: "sensor-grid".to_owned(),
        formatting_token: "<|vector|>".to_owned(),
        modality: "sensor_frame".to_owned(),
        empty_payload: serde_json::json!({}),
    }],
};
let encoding = load_harmony_encoding_with_options(
    HarmonyEncodingName::Qwen35Harmony,
    options,
)?;
```

The loader rejects zero-sized, overlapping, or out-of-pool ranges. Qwen
encodings require a local `vocab.json` authenticated against the official
release pin or an explicit alternative digest; see
[Compatibility](compatibility.md). Tokenizer configuration must match the
model or checkpoint consuming the output.

## Parsing and streaming

`parse_messages_from_completion_tokens` validates and parses a complete token
suffix. Use `parse_messages_from_completion_tokens_with_options` with
`ParseOptions { strict: false }` only for deliberate malformed-output
recovery.

`completion_stream_parser(CompletionStreamOptions)` is the higher-level
incremental interface. Its default parses an assistant completion suffix after
the already-rendered assistant prefix. Set `role: None` to parse a full message
stream. `finish(absolute_start)` flushes an open message when a provider ends
without emitting a Harmony stop token.

`StreamableParser` is the lower-level token-at-a-time parser. The completion
view API additionally provides append-safe visible text for incremental
pointing output.

## Features

- `python-binding`: builds the PyO3 extension used by the Python package.
- `wasm-binding`: builds the wasm-bindgen surface.

WebAssembly supports `HarmonyGptOss`. Qwen encodings currently require native
filesystem vocabulary loading and are rejected in wasm builds. An npm package
is not part of the `0.1.0` release.

## Errors

Most fallible APIs return `anyhow::Result`. Loading can fail because a
vocabulary is absent or invalid; rendering can fail on unsupported formatting;
parsing can fail on malformed token streams.

Applications should bound untrusted JSON, base64, and token-array sizes before
deserialization. See [SECURITY.md](../SECURITY.md) for the reporting policy.
