<!-- Modified by Perceptron AI, Inc.; derived from OpenAI Harmony. -->

# Python guide

The `mharmony` Python package combines a Rust extension with a small Python
facade for constructing conversations. It supports Python 3.10 and later.

```bash
pip install mharmony==0.1.0
```

The distribution includes a `py.typed` marker and a stub for the native
extension, so type checkers can consume the packaged PEP 561 metadata. The
hand-written Python facade also carries runtime annotations.

## Render a conversation

```python
from mharmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    load_harmony_encoding,
)

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
conversation = Conversation.from_messages(
    [
        Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions("Be concise."),
        ),
        Message.from_role_and_content(Role.USER, "What is 2 + 2?"),
    ]
)

prompt_tokens = encoding.render_conversation_for_completion(
    conversation,
    Role.ASSISTANT,
)
```

Use `render_conversation_for_training()` for supervised targets,
`render_conversation()` when no next-role prefix is wanted, and `render()` for
a single message. `RenderConversationConfig(auto_drop_analysis=True)` controls
whether earlier analysis is removed during conversation rendering.

## Conversation data model

| Type | Purpose |
| --- | --- |
| `Role`, `Author` | Message identity |
| `Message`, `Conversation` | Ordered conversation structure and JSON round trips |
| `TextContent`, `TokensContent`, `VectorContent` | Generic non-media payloads |
| `MediaRef`, `MediaContent` | Modality-neutral media identity and payload envelope |
| `ImageContent` | Convenience helper for `MediaContent(kind="image", ...)` |
| `PointingContent`, `PointCollectionContent` | Point, box, polygon, clip, collection, and track markup |
| `SystemContent`, `DeveloperContent` | Harmony system metadata, instructions, and tool definitions |

`Message.from_role_and_content()` accepts text or a content object.
`Message.from_role_and_contents()` accepts multiple content objects.
`with_channel()`, `with_recipient()`, and `with_content_type()` add Harmony
header fields.

## Generic media envelopes

Every public binary-media value has the same serialized shape:

```json
{
  "type": "media",
  "kind": "image",
  "media_ref": {
    "id": "camera.front",
    "mime": "image/jpeg",
    "bytes_b64": "..."
  },
  "metadata": {}
}
```

The fields are:

- `kind`: a non-empty registered name. `image` is the processor integration
  published in `0.1.x`; the core also preserves the legacy `video` wire
  token/event mapping without defining its processor-result contract.
- `media_ref.id`: stable application identity for the media item.
- `media_ref.mime`: optional MIME hint.
- `media_ref.bytes_b64`: optional inline bytes. The binding base64-decodes them
  before calling a processor.
- `media_ref.metadata` and `metadata`: opaque JSON metadata.
- `source`: optional opaque, JSON-compatible host descriptor. When present it
  takes precedence over inline bytes as the processor input.

Construct an image with the convenience helper:

```python
import base64

from mharmony import ImageContent, MediaRef

content = ImageContent(
    media_ref=MediaRef(
        id="camera.front",
        mime="image/png",
        bytes_b64=base64.b64encode(png_bytes).decode("ascii"),
    ),
    metadata={"camera": "front"},
)
```

Register and construct a host-defined transport envelope without adding
another core content variant:

```python
from mharmony import (
    HarmonyEncodingName,
    MediaContent,
    MediaKindRegistration,
    MediaRef,
    load_harmony_encoding,
)

encoding = load_harmony_encoding(
    HarmonyEncodingName.HARMONY_GPT_OSS,
    media_kinds=[
        MediaKindRegistration(
            kind="sensor-grid",
            formatting_token="<|vector|>",
            modality="sensor_frame",
        )
    ],
)

content = MediaContent(
    kind="sensor-grid",
    media_ref=MediaRef(id="observation-7"),
    source={"asset_id": "observation-7"},
)
```

The loader requires the formatting token to be one tokenizer special token and
rejects duplicate kinds. Adding a callable to `media_processors` selects
processing for an already-registered kind; it does not define the wire format.
`0.1.x` does not publish kind-specific processor or tensor contracts for
application-defined kinds.

## Media processor callbacks

Model-specific media work is callback-delegated. Prefer the per-kind registry:

```python
rendered = encoding.render_conversation_multimodal_with_processors(
    conversation,
    preprocess_config={"patch_size": 16, "max_num_patches": 3600},
    media_processors={"image": process_image},
)
```

The high-level signature is:

```python
def render_conversation_multimodal_with_processors(
    conversation,
    preprocess_config=None,
    media_processors=None,
    *,
    media_validators=None,
    image_processor=None,
    timestamp_processor=None,
) -> dict: ...
```

`media_validators[kind](media_envelope, config_dict)` runs before rendering and
processing. It may raise or return exactly `False` to reject an envelope.
`image_processor` is a compatibility fallback; new integrations should use
`media_processors={"image": ...}`. The core itself contains no media codec.

Each registered callable receives `(source, config_dict)`:

- `source` is the envelope's opaque `source` value when present; otherwise it
  is decoded `bytes` from `media_ref.bytes_b64`. When neither is present, the
  renderer retains a placeholder and does not invoke the processor.
- `config_dict` contains normalized preprocessing fields and includes the
  media item's `metadata` when present.

An image processor returns an object or mapping exposing `data` and `shape`.
The shape must be `[height_patches, width_patches, patch_dimension]`.
`ProcessedImage` is a convenient result object:

```python
from mharmony import ProcessedImage

def process_image(source: bytes, config: dict) -> ProcessedImage:
    patches = application_image_preprocessor(source, config)
    return ProcessedImage(patches, list(patches.shape))
```

The rendered stream contains an `image_frame` event with
`{"Image": {"data": ..., "shape": [...]}}`. Non-image processor payload
semantics are host-defined; a preserved wire discriminant is not a public
tensor contract.

`timestamp_processor(text, role, context)` is optional. It can rewrite
pointing markup before tokenization using host-provided temporal context.

### Qwen3.5 image adapter

The optional `qwen35` extra provides the standalone Qwen3.5 image transform
needed by LeRobot and other Qwen3.5 consumers:

```bash
pip install "mharmony[qwen35]==0.1.0"
```

```python
from mharmony import create_qwen35_image_processor

image_processor = create_qwen35_image_processor()
rendered = encoding.render_conversation_multimodal_with_processors(
    conversation,
    preprocess_config={
        "patch_size": 16,
        "max_num_patches": 576,
        "pixel_shuffle_scale": 1,
    },
    media_processors={"image": image_processor},
)
```

The extra installs NumPy, Pillow, PyTorch, torchvision, and Transformers. The
factory decodes image bytes, applies EXIF orientation, uses Qwen-aligned smart
resizing, normalizes pixels, and returns the required three-dimensional patch
layout. Factory options provide defaults; keys supplied explicitly through
`preprocess_config` take precedence. Because preprocessing is checkpoint
critical, record and pin the resolved NumPy, Pillow, PyTorch, torchvision, and
Transformers versions with each checkpoint. `create_qwen35_mharmony_image_processor`
is retained as a compatibility alias for existing integrations.

The adapter accepts inline image bytes, not an opaque `source` descriptor. It
also enforces a pre-decode pixel ceiling (`max_decode_pixels`, 64 million by
default) before allocating an RGB array; choose a lower bound for constrained
workers. Native multimodal rendering separately limits decoded inline payloads
to 64 MiB by default. Set a positive `max_inline_media_bytes` value in
`preprocess_config` to use a lower application-specific ceiling.

## Encodings and reserved token groups

Available names are:

- `HarmonyEncodingName.HARMONY_GPT_OSS`
- `HarmonyEncodingName.QWEN3_HARMONY`
- `HarmonyEncodingName.QWEN35_HARMONY`

Qwen encodings require a local vocabulary authenticated against the official
release pin or an explicit alternative digest; see
[Compatibility](compatibility.md).

Reserve application token ranges while loading an encoding:

```python
encoding = load_harmony_encoding(
    HarmonyEncodingName.QWEN35_HARMONY,
    reserved_token_groups=[
        {"name": "coord", "size": 4096},
    ],
)
```

Use `encoding_vocab_info()` and `resolve_reserved_token_group_offsets()` when
an application needs to validate a model vocabulary layout before loading a
checkpoint.

## Parse completions

For a complete generated suffix:

```python
messages = encoding.parse_messages_from_completion_tokens(
    completion_tokens,
    role=Role.ASSISTANT,
    strict=True,
)
```

Set `strict=False` only when recovery from malformed model output is desired.
Pass `role=None` to parse a full stream that includes each message's start and
role header.

For incremental generation:

```python
parser = encoding.completion_stream_parser(
    role=Role.ASSISTANT,
    strict=False,
    stop_on_tool_call=True,
)

offset = 0
for delta in token_deltas:
    update = parser.push_tokens(delta, absolute_start=offset)
    offset += len(delta)
    if update.stop is not None:
        break
else:
    update = parser.finish(absolute_start=offset)
```

`completion_stream_parser()` is the higher-level streaming API.
`StreamableParser` exposes the lower-level token-by-token state machine.
`parse_completion_view_from_tokens()` returns parsed messages plus stable,
append-safe display text for pointing spans.

## Errors

`HarmonyError` represents rendering, parsing, tokenization, and vocabulary
failures. Argument and schema validation may raise `ValueError` or Pydantic
validation errors. Media callbacks can also propagate their own exceptions.

Treat media bytes and token streams as untrusted input and apply application
limits before constructing large base64 or token arrays.
