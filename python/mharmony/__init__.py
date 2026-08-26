# Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
"""Python wrapper around the Rust implementation of *harmony*.

The heavy lifting (tokenisation, rendering, parsing, …) is implemented in
Rust.  The thin bindings are available through the private ``mharmony``
extension module which is compiled via *maturin* / *PyO3*.

This package provides a small, typed convenience layer that mirrors the public
API of the Rust crate so that it can be used from Python code in an
idiomatic way (``dataclasses``, ``Enum``s, …).
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import (
    AbstractSet,
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Pattern,
    Sequence,
    Union,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

# Re-export the low-level Rust bindings under a private name so that we can
# keep the *public* namespace clean and purely Pythonic.
from .mharmony import (
    CanonicalChannel as CanonicalChannel,
    HarmonyError as HarmonyError,
    ProcessedImage as ProcessedImage,
    PyHarmonyEncoding as _PyHarmonyEncoding,
    PyStreamableParser as _PyStreamableParser,
    encoding_vocab_info as encoding_vocab_info,
    format_markup_open_tag as format_markup_open_tag,
    format_markup_open_tag_display_t as format_markup_open_tag_display_t,
    get_tool_namespace_config as _get_tool_namespace_config,
    load_harmony_encoding as _load_harmony_encoding,
    resolve_reserved_token_group_offsets as resolve_reserved_token_group_offsets,
)
from .processors import (
    create_qwen35_image_processor as create_qwen35_image_processor,
    create_qwen35_mharmony_image_processor as create_qwen35_mharmony_image_processor,
)


def _json_dumps(value: Any) -> str:
    """Serialize only JSON that the Rust ``serde_json`` boundary accepts."""
    return json.dumps(value, allow_nan=False)


@functools.lru_cache(maxsize=128)
def _special_token_regex(tokens: frozenset[str]) -> Pattern[str]:
    if not tokens:
        return re.compile(r"(?!x)x")
    inner = "|".join(re.escape(token) for token in tokens)
    return re.compile(f"({inner})")


@functools.lru_cache(maxsize=128)
def _special_token_initials(tokens: frozenset[str]) -> frozenset[str]:
    """Return the tiny prefix set used to skip impossible regex searches."""
    return frozenset(token[:1] for token in tokens)


def _validate_nonempty_unpadded(value: str, label: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{label} must be non-empty")
    if value != stripped:
        raise ValueError(f"{label} must not contain surrounding whitespace")
    return value


def raise_disallowed_special_token(token: str) -> None:
    raise HarmonyError(
        "Encountered text corresponding to disallowed special token "
        f"{token!r}.\n"
        "If you want this text to be encoded as a special token, "
        f"pass it to `allowed_special`, e.g. `allowed_special={{'{token}', ...}}`.\n"
        "If you want this text to be encoded as normal text, disable the check for this token "
        f"by passing `disallowed_special=(enc.special_tokens_set - {{'{token}'}})`.\n"
        "To disable this check for all special tokens, pass `disallowed_special=()`.\n"
    )


# ---------------------------------------------------------------------------
# Chat-related data-structures (mirroring ``src/chat.rs``)
# ---------------------------------------------------------------------------


class Role(str, Enum):
    """The role of a message author (mirrors ``chat::Role``)."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    DEVELOPER = "developer"
    TOOL = "tool"

    @classmethod
    def _missing_(cls, value: object) -> "Role":  # type: ignore[override]
        raise ValueError(f"Unknown role: {value!r}")


class Author(BaseModel):
    role: Role
    name: Optional[str] = None

    @classmethod
    def new(cls, role: Role, name: str) -> "Author":  # noqa: D401 – keep parity with Rust API
        return cls(role=role, name=name)


# Content hierarchy ---------------------------------------------------------


class Content(BaseModel):  # noqa: D101 – simple wrapper
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


class TextContent(Content):
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "text", "text": self.text}


class TokensContent(Content):
    tokens: List[int]
    tags: Optional[dict[str, JsonValue]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "tokens", **self.model_dump(exclude_none=True)}


class VectorContent(Content):
    values: List[float]
    shape: List[int]
    dtype: Optional[str] = None
    metadata: Optional[JsonValue] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "vector", **self.model_dump(exclude_none=True)}


class MediaRef(BaseModel):
    """A stable media identity with an optional inline byte payload."""

    id: str
    mime: Optional[str] = None
    bytes_b64: Optional[str] = None
    metadata: Optional[JsonValue] = None
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_nonempty_unpadded(value, "media ref id")


class MediaKindRegistration(BaseModel):
    """Runtime token and stream registration for one media kind.

    Processor and validator callbacks remain host-side and are supplied when
    rendering. ``formatting_token`` must be one special token in the selected
    tokenizer.
    """

    kind: str
    formatting_token: str
    modality: str
    empty_payload: JsonValue = Field(default_factory=dict)

    @field_validator("kind", "formatting_token", "modality")
    @classmethod
    def validate_nonempty_string(cls, value: str) -> str:
        return _validate_nonempty_unpadded(value, "registration strings")


class MediaContent(Content):
    """Generic media envelope shared by every registered media kind.

    ``source`` is an opaque JSON value for host-defined, non-byte sources. The
    core does not publish kind-specific source or processor result schemas.
    """

    kind: str
    media_ref: MediaRef
    model_config = ConfigDict(validate_assignment=True)

    metadata: Optional[JsonValue] = None
    source: Optional[JsonValue] = None

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        return _validate_nonempty_unpadded(value, "media kind")

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "media", **self.model_dump(exclude_none=True)}


class ImageContent(MediaContent):
    """Convenience media envelope for the public ``image`` registration."""

    kind: Literal["image"] = "image"


class VideoContent(MediaContent):
    """Typed envelope helper for the public ``video`` registration.

    Processing and tensor layout remain application-defined; this helper only
    constructs the generic public media envelope.
    """

    kind: Literal["video"] = "video"


class PointingKind(str, Enum):
    POINT = "point"
    BOX = "box"
    POLYGON = "polygon"
    CLIP = "clip"


class PointCollectionKind(str, Enum):
    COLLECTION = "collection"
    TRACK = "track"


class PointingContent(Content):
    kind: PointingKind
    coords: List[tuple[int, int]] = Field(default_factory=list)
    target: Optional[int] = None
    t: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    mention: Optional[str] = None
    label: Optional[str] = None
    index: Optional[str] = None

    @model_validator(mode="after")
    def validate_kind_payload(self) -> "PointingContent":
        if self.kind == PointingKind.CLIP:
            if self.start_time is None:
                raise ValueError("Clip pointing requires start_time")
            return self
        if not self.coords:
            raise ValueError(f"{self.kind.value} pointing requires coords")
        return self

    def to_dict(self) -> Dict[str, Any]:
        out = self.model_dump(exclude_none=True)
        out["type"] = "pointing"
        return out


class PointCollectionContent(Content):
    points: List[Union[PointingContent, "PointCollectionContent"]]
    kind: PointCollectionKind = PointCollectionKind.COLLECTION
    t: Optional[float] = None
    mention: Optional[str] = None
    label: Optional[str] = None

    @model_validator(mode="after")
    def validate_children(self) -> "PointCollectionContent":
        if self.kind == PointCollectionKind.TRACK:
            if any(isinstance(point, PointCollectionContent) for point in self.points):
                raise ValueError(
                    "Track point collection can only contain pointing children"
                )
            leaf_kinds = [
                point.kind
                for point in self.points
                if isinstance(point, PointingContent)
            ]
            if any(kind == PointingKind.CLIP for kind in leaf_kinds):
                raise ValueError("Track points must be point, point_box, or polygon")
            if leaf_kinds and any(kind != leaf_kinds[0] for kind in leaf_kinds):
                raise ValueError("Track points must all be the same type")
        elif any(
            isinstance(point, PointCollectionContent)
            and point.kind != PointCollectionKind.TRACK
            for point in self.points
        ):
            raise ValueError(
                "Collection point collection can contain track children, but not nested collections"
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        out = self.model_dump(exclude_none=True)
        if self.kind == PointCollectionKind.COLLECTION:
            out.pop("kind", None)
        out["points"] = [point.to_dict() for point in self.points]
        out["type"] = "point_collection"
        return out


PointCollectionContent.model_rebuild()


class ToolDescription(BaseModel):
    name: str
    description: str
    parameters: Optional[JsonValue] = None

    @classmethod
    def new(
        cls, name: str, description: str, parameters: Optional[JsonValue] = None
    ) -> "ToolDescription":  # noqa: D401
        return cls(name=name, description=description, parameters=parameters)


class ReasoningEffort(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class SystemHint(str, Enum):
    POINT = "POINT"
    BOX = "BOX"
    POLYGON = "POLYGON"
    CLIP = "CLIP"
    FOCUS = "FOCUS"
    THINK = "THINK"
    TOOLS = "TOOLS"


class ChannelConfig(BaseModel):
    valid_channels: List[str]
    channel_required: bool

    @classmethod
    def require_channels(
        cls, channels: Sequence[str | CanonicalChannel]
    ) -> "ChannelConfig":  # noqa: D401
        return cls(
            valid_channels=[str(channel) for channel in channels], channel_required=True
        )

    @classmethod
    def require_canonical_channels(cls) -> "ChannelConfig":
        return cls.require_channels(CanonicalChannel.canonical())


class ToolNamespaceConfig(BaseModel):
    name: str
    description: Optional[str] = None
    tools: List[ToolDescription]

    @staticmethod
    def browser() -> "ToolNamespaceConfig":
        return ToolNamespaceConfig(**_get_tool_namespace_config("browser"))

    @staticmethod
    def python() -> "ToolNamespaceConfig":
        return ToolNamespaceConfig(**_get_tool_namespace_config("python"))


class SystemContent(Content):
    model_identity: Optional[str] = "You are ChatGPT, a large language model."
    reasoning_effort: Optional[ReasoningEffort] = ReasoningEffort.MEDIUM
    hints: Optional[List[SystemHint]] = None
    conversation_start_date: Optional[str] = None
    knowledge_cutoff: Optional[str] = "2024-06"
    channel_config: Optional[ChannelConfig] = Field(
        default_factory=ChannelConfig.require_canonical_channels
    )
    tools: Optional[dict[str, ToolNamespaceConfig]] = None

    @classmethod
    def new(cls) -> "SystemContent":
        return cls()

    @classmethod
    def empty(cls) -> "SystemContent":
        return cls(
            model_identity=None,
            reasoning_effort=None,
            hints=None,
            conversation_start_date=None,
            knowledge_cutoff=None,
            channel_config=None,
            tools=None,
        )

    # Fluent interface ------------------------------------------------------

    def with_model_identity(self, model_identity: str) -> "SystemContent":
        self.model_identity = model_identity
        return self

    def with_reasoning_effort(
        self, reasoning_effort: ReasoningEffort
    ) -> "SystemContent":
        self.reasoning_effort = reasoning_effort
        return self

    def with_hints(self, hints: List[SystemHint]) -> "SystemContent":
        self.hints = hints
        return self

    def with_hint(self, hint: SystemHint) -> "SystemContent":
        if self.hints is None:
            self.hints = []
        self.hints.append(hint)
        return self

    def with_conversation_start_date(
        self, conversation_start_date: str
    ) -> "SystemContent":
        self.conversation_start_date = conversation_start_date
        return self

    def with_knowledge_cutoff(self, knowledge_cutoff: str) -> "SystemContent":
        self.knowledge_cutoff = knowledge_cutoff
        return self

    def with_channel_config(self, channel_config: ChannelConfig) -> "SystemContent":
        self.channel_config = channel_config
        return self

    def with_required_channels(self, channels: Sequence[str]) -> "SystemContent":
        self.channel_config = ChannelConfig.require_channels(channels)
        return self

    def with_tools(self, ns_config: ToolNamespaceConfig) -> "SystemContent":
        if self.tools is None:
            self.tools = {}
        self.tools[ns_config.name] = ns_config
        return self

    def with_browser_tool(self) -> "SystemContent":
        return self.with_tools(ToolNamespaceConfig.browser())

    def with_python_tool(self) -> "SystemContent":
        return self.with_tools(ToolNamespaceConfig.python())

    def to_dict(self) -> dict:
        out = self.model_dump(exclude_none=True)
        out["type"] = "system_content"
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> "SystemContent":
        return cls(**raw)


class DeveloperContent(Content):
    instructions: Optional[str] = None
    tools: Optional[dict[str, ToolNamespaceConfig]] = None

    @classmethod
    def new(cls) -> "DeveloperContent":
        return cls()

    def with_instructions(self, instructions: str) -> "DeveloperContent":
        self.instructions = instructions
        return self

    def with_tools(self, ns_config: ToolNamespaceConfig) -> "DeveloperContent":
        if self.tools is None:
            self.tools = {}
        self.tools[ns_config.name] = ns_config
        return self

    def with_function_tools(
        self, tools: Sequence[ToolDescription]
    ) -> "DeveloperContent":
        return self.with_tools(
            ToolNamespaceConfig(name="functions", description=None, tools=list(tools))
        )

    def to_dict(self) -> dict:
        out = self.model_dump(exclude_none=True)
        out["type"] = "developer_content"
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> "DeveloperContent":
        return cls(**raw)


# Message & Conversation -----------------------------------------------------


class Message(BaseModel):
    author: Author
    content: List[Content] = Field(default_factory=list)
    channel: Optional[str] = None
    recipient: Optional[str] = None
    content_type: Optional[str] = None

    # ------------------------------------------------------------------
    # Convenience constructors (mirroring the Rust API)
    # ------------------------------------------------------------------

    @classmethod
    def from_author_and_content(
        cls, author: Author, content: Union[str, Content]
    ) -> "Message":
        if isinstance(content, str):
            content = TextContent(text=content)
        return cls(author=author, content=[content])

    @classmethod
    def from_role_and_content(
        cls, role: Role, content: Union[str, Content]
    ) -> "Message":  # noqa: D401 – parity with Rust API
        return cls.from_author_and_content(Author(role=role), content)

    @classmethod
    def from_role_and_contents(
        cls, role: Role, contents: Sequence[Content]
    ) -> "Message":
        return cls(author=Author(role=role), content=list(contents))

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def adding_content(self, content: Union[str, Content]) -> "Message":
        if isinstance(content, str):
            content = TextContent(text=content)
        self.content.append(content)
        return self

    def with_channel(self, channel: str) -> "Message":
        self.channel = str(channel)
        return self

    def with_recipient(self, recipient: str) -> "Message":
        self.recipient = recipient
        return self

    def with_content_type(self, content_type: str) -> "Message":
        self.content_type = content_type
        return self

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:  # noqa: D401 – simple mapper
        out: Dict[str, Any] = {
            **self.author.model_dump(),
            "content": [c.to_dict() for c in self.content],
        }
        if self.channel is not None:
            out["channel"] = self.channel
        if self.recipient is not None:
            out["recipient"] = self.recipient
        if self.content_type is not None:
            out["content_type"] = self.content_type
        return out

    def to_json(self) -> str:  # noqa: D401
        return _json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        # Simple, sufficient implementation for test-roundtrip purposes.
        role = Role(data["role"])
        author = Author(role=role, name=data.get("name"))

        contents: List[Content] = []

        raw_content = data["content"]

        # The Rust side serialises *single* text contents as a **plain string**
        # for convenience.  Detect this shortcut and normalise it to the list
        # representation that the rest of the Python code expects.
        if isinstance(raw_content, str):
            raw_content = [{"type": "text", "text": raw_content}]

        for raw in raw_content:
            if raw.get("type") == "text":
                contents.append(TextContent(**raw))
            elif raw.get("type") == "tokens":
                contents.append(TokensContent(**raw))
            elif raw.get("type") == "vector":
                contents.append(VectorContent(**raw))
            elif raw.get("type") == "system_content":
                contents.append(SystemContent(**raw))
            elif raw.get("type") == "developer_content":
                contents.append(DeveloperContent(**raw))
            elif raw.get("type") == "media":
                media_type: type[MediaContent]
                if raw.get("kind") == "image":
                    media_type = ImageContent
                elif raw.get("kind") == "video":
                    media_type = VideoContent
                else:
                    media_type = MediaContent
                contents.append(media_type.model_validate(raw))
            elif raw.get("type") == "pointing":
                contents.append(PointingContent(**raw))
            elif raw.get("type") == "point_collection":
                contents.append(PointCollectionContent(**raw))
            else:  # pragma: no cover – unknown variant
                raise ValueError(f"Unknown content variant: {raw}")

        msg = cls(author=author, content=contents)
        msg.channel = data.get("channel")
        msg.recipient = data.get("recipient")
        msg.content_type = data.get("content_type")
        return msg


class CompletionParseView(BaseModel):
    """Structured messages plus per-message append-safe stable display text.

    ``stable_message_texts`` is index-aligned with ``messages``. Each stable
    text string is suitable for progressive display of the corresponding message:
    it never contains mHarmony point sentinels, may include completed point
    children before a point span closes, and leaves channel/role visibility
    filtering to callers.
    """

    messages: List[Message]
    stable_message_texts: List[str]

    def model_post_init(self, __context: Any) -> None:
        if len(self.messages) != len(self.stable_message_texts):
            raise HarmonyError("messages and stable_message_texts must align")


@dataclass(slots=True)
class StreamTextUpdate:
    visible_delta: str
    visible_full: str
    final_delta: str
    final_full: str
    reasoning_delta: str
    reasoning_full: str
    visible_delta_parts: list[str]
    final_delta_parts: list[str]
    reasoning_delta_parts: list[str]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StreamTextUpdate":
        return cls(
            visible_delta=str(raw.get("visible_delta", "")),
            visible_full=str(raw.get("visible_full", "")),
            final_delta=str(raw.get("final_delta", "")),
            final_full=str(raw.get("final_full", "")),
            reasoning_delta=str(raw.get("reasoning_delta", "")),
            reasoning_full=str(raw.get("reasoning_full", "")),
            visible_delta_parts=list(raw.get("visible_delta_parts", [])),
            final_delta_parts=list(raw.get("final_delta_parts", [])),
            reasoning_delta_parts=list(raw.get("reasoning_delta_parts", [])),
        )


@dataclass(slots=True)
class StreamStop:
    kind: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StreamStop":
        return cls(kind=str(raw["kind"]))


@dataclass(slots=True)
class CompletionStreamUpdate:
    token_count: int
    completed_messages: list[Message]
    stop: StreamStop | None
    text: StreamTextUpdate

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CompletionStreamUpdate":
        stop = raw.get("stop")
        return cls(
            token_count=int(raw["token_count"]),
            completed_messages=[
                Message.from_dict(message)
                for message in raw.get("completed_messages", [])
            ],
            stop=StreamStop.from_dict(stop) if isinstance(stop, dict) else None,
            text=StreamTextUpdate.from_dict(raw["text"]),
        )


class Conversation(BaseModel):
    messages: List[Message] = Field(default_factory=list)

    @classmethod
    def from_messages(cls, messages: Sequence[Message]) -> "Conversation":  # noqa: D401
        return cls(messages=list(messages))

    def __iter__(self):
        return iter(self.messages)

    # Serialisation helpers -------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:  # noqa: D401
        return {"messages": [m.to_dict() for m in self.messages]}

    def to_json(self) -> str:  # noqa: D401
        return _json_dumps(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "Conversation":  # noqa: D401
        data = json.loads(payload)
        return cls(messages=[Message.from_dict(m) for m in data["messages"]])


# ---------------------------------------------------------------------------
# Encoding interaction (thin wrappers around the Rust bindings)
# ---------------------------------------------------------------------------


class RenderConversationConfig(BaseModel):
    auto_drop_analysis: bool = True


class RenderOptions(BaseModel):
    conversation_has_function_tools: bool = False


class MessageHeader(BaseModel):
    author: Author
    recipient: Optional[str] = None
    channel: Optional[str] = None
    content_type: Optional[str] = None

    @classmethod
    def from_author(cls, author: Author) -> "MessageHeader":
        return cls(author=author)

    @classmethod
    def from_role(cls, role: Role) -> "MessageHeader":
        return cls.from_author(Author(role=role))

    @classmethod
    def assistant(cls) -> "MessageHeader":
        return cls.from_role(Role.ASSISTANT)

    @classmethod
    def from_message(cls, message: Message) -> "MessageHeader":
        return cls(
            author=message.author,
            recipient=message.recipient,
            channel=message.channel,
            content_type=message.content_type,
        )

    def with_recipient(self, recipient: str) -> "MessageHeader":
        self.recipient = recipient
        return self

    def with_channel(self, channel: str) -> "MessageHeader":
        self.channel = str(channel)
        return self

    def with_content_type(self, content_type: str) -> "MessageHeader":
        self.content_type = content_type
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "author": {"role": self.author.role.value, "name": self.author.name},
            "recipient": self.recipient,
            "channel": self.channel,
            "content_type": self.content_type,
        }

    def to_json(self) -> str:
        return _json_dumps(self.to_dict())


class HeaderRenderOptions(BaseModel):
    include_start_token: bool = True
    include_author: bool = True
    include_recipient: bool = True
    include_channel_marker: bool = True
    include_channel_value: bool = True
    include_content_type: bool = True
    include_message_token: bool = True

    @classmethod
    def none(cls) -> "HeaderRenderOptions":
        return cls(
            include_start_token=False,
            include_author=False,
            include_recipient=False,
            include_channel_marker=False,
            include_channel_value=False,
            include_content_type=False,
            include_message_token=False,
        )

    def with_start_token(self, include: bool) -> "HeaderRenderOptions":
        self.include_start_token = include
        return self

    def with_author(self, include: bool) -> "HeaderRenderOptions":
        self.include_author = include
        return self

    def with_recipient(self, include: bool) -> "HeaderRenderOptions":
        self.include_recipient = include
        return self

    def with_channel_marker(self, include: bool) -> "HeaderRenderOptions":
        self.include_channel_marker = include
        return self

    def with_channel_value(self, include: bool) -> "HeaderRenderOptions":
        self.include_channel_value = include
        return self

    def with_content_type(self, include: bool) -> "HeaderRenderOptions":
        self.include_content_type = include
        return self

    def with_message_token(self, include: bool) -> "HeaderRenderOptions":
        self.include_message_token = include
        return self

    def to_json(self) -> str:
        return _json_dumps(self.model_dump())


class HarmonyEncoding:
    """High-level wrapper around the Rust ``PyHarmonyEncoding`` class."""

    def __init__(self, inner: _PyHarmonyEncoding):
        self._inner = inner

    # ------------------------------------------------------------------
    # Delegated helpers
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        return self._inner.name  # type: ignore[attr-defined]

    @functools.cached_property
    def special_tokens_set(self) -> frozenset[str]:
        return frozenset(self._inner.special_tokens())

    # -- Rendering -----------------------------------------------------

    def render_message_header_text(
        self, header: MessageHeader, options: Optional[HeaderRenderOptions] = None
    ) -> str:
        if options is None:
            options = HeaderRenderOptions()
        return self._inner.render_message_header_text(
            header_json=header.to_json(), options_json=options.to_json()
        )

    def render_message_header_tokens(
        self, header: MessageHeader, options: Optional[HeaderRenderOptions] = None
    ) -> List[int]:
        if options is None:
            options = HeaderRenderOptions()
        return self._inner.render_message_header_tokens(
            header_json=header.to_json(), options_json=options.to_json()
        )

    def render_conversation_for_completion(
        self,
        conversation: Conversation,
        next_turn_role: Role,
        config: Optional[RenderConversationConfig] = None,
    ) -> List[int]:
        """
        Render a conversation for completion.
        Args:
            conversation: Conversation object
            next_turn_role: Role for the next turn
            config: Optional RenderConversationConfig (default auto_drop_analysis=True)
        """
        config_dict = (config or RenderConversationConfig()).model_dump()
        return self._inner.render_conversation_for_completion(
            conversation_json=conversation.to_json(),
            next_turn_role=str(next_turn_role.value),
            config=config_dict,
        )

    def render_conversation(
        self,
        conversation: Conversation,
        config: Optional[RenderConversationConfig] = None,
    ) -> List[int]:
        """Render a conversation without appending a new role."""
        config_dict = (config or RenderConversationConfig()).model_dump()
        return self._inner.render_conversation(
            conversation_json=conversation.to_json(),
            config=config_dict,
        )

    def render_conversation_for_training(
        self,
        conversation: Conversation,
        config: Optional[RenderConversationConfig] = None,
    ) -> List[int]:
        """Render a conversation for training."""
        config_dict = (config or RenderConversationConfig()).model_dump()
        return self._inner.render_conversation_for_training(
            conversation_json=conversation.to_json(),
            config=config_dict,
        )

    def render(
        self, message: Message, render_options: Optional[RenderOptions] = None
    ) -> List[int]:
        """Render a single message into tokens."""
        render_options_dict = (render_options or RenderOptions()).model_dump()
        return self._inner.render(
            message_json=message.to_json(), render_options=render_options_dict
        )

    def render_conversation_multimodal(
        self,
        conversation: Conversation,
        preprocess_config: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        Render a conversation and return a multimodal RenderedStream (JSON dict).

        Args:
            conversation: The conversation to render.
            preprocess_config: Optional preprocessing config with keys:
                - patch_size: int (default 16)
                - max_num_patches: int or None
                - min_num_patches: int or None
                - pixel_shuffle_scale: int (default 1)
        """
        raw = self._inner.render_conversation_multimodal(
            conversation.to_json(),
            preprocess_config,
        )
        return json.loads(raw)

    def render_conversation_multimodal_with_processors(
        self,
        conversation: Conversation,
        preprocess_config: Optional[Dict[str, Any]] = None,
        media_processors: Optional[
            Mapping[str, Callable[[Any, Dict[str, Any]], Any]]
        ] = None,
        *,
        media_validators: Optional[
            Mapping[
                str,
                Callable[[dict[str, Any], Dict[str, Any]], object],
            ]
        ] = None,
        image_processor: Optional[Callable[[Any, Dict[str, Any]], Any]] = None,
        timestamp_processor: Optional[Callable[[str, str, Dict[str, Any]], str]] = None,
    ) -> dict:
        """Render a conversation with host-defined processors for media kinds.

        Args:
            conversation: The conversation to render.
            preprocess_config: Opaque configuration forwarded to each processor.
            media_processors: Mapping from media ``kind`` to a callable taking
                ``(source, config_dict)``. The callback's result contract belongs to
                that registered media kind.
            media_validators: Mapping from media ``kind`` to a callable taking
                ``(media_envelope_dict, config_dict)``. Validators run before
                rendering and before processors. They may raise to reject an
                envelope or return exactly ``False`` for a ``ValueError``.
            image_processor: Compatibility fallback for the public ``image`` kind.
                A processor in ``media_processors["image"]`` takes precedence.
            timestamp_processor: A Python callable that takes (text, role, context)
                and returns rewritten text before tokenization.

        Returns:
            A dict containing the rendered stream with events.
        """
        processors = None if media_processors is None else dict(media_processors)
        processor = processors.get("image") if processors is not None else None
        if processor is None:
            processor = image_processor
        defaults = getattr(processor, "_mharmony_preprocess_defaults", None)
        if isinstance(defaults, Mapping):
            preprocess_config = {**defaults, **(preprocess_config or {})}

        return self._inner.render_conversation_multimodal_with_processors(
            conversation.to_json(),
            preprocess_config,
            processors,
            image_processor,
            timestamp_processor,
            None if media_validators is None else dict(media_validators),
        )

    # -- Parsing -------------------------------------------------------

    def parse_messages_from_completion_tokens(
        self,
        tokens: Sequence[int],
        role: Optional[Role] | None = None,
        *,
        strict: bool = True,
    ) -> List[Message]:
        raw_json: str = self._inner.parse_messages_from_completion_tokens(
            list(tokens), None if role is None else str(role.value), strict
        )
        return [Message.from_dict(m) for m in json.loads(raw_json)]

    def parse_completion_view_from_tokens(
        self,
        tokens: Sequence[int],
        role: Optional[Role] | None = None,
        *,
        strict: bool = True,
    ) -> CompletionParseView:
        """Parse completion tokens once into messages plus aligned stable text.

        Stable text is per-message display text that does not include mHarmony
        point sentinels and can expose completed point children before their
        enclosing point span closes. Visibility filtering remains outside this
        parser view.
        """

        raw_json: str = self._inner.parse_completion_view_from_tokens(
            list(tokens), None if role is None else str(role.value), strict
        )
        raw = json.loads(raw_json)
        return CompletionParseView(
            messages=[Message.from_dict(m) for m in raw["messages"]],
            stable_message_texts=list(raw["stable_message_texts"]),
        )

    def stable_visible_text_from_point_spans(self, text: str) -> str:
        return self._inner.stable_visible_text_from_point_spans(text)

    def completion_stream_parser(
        self,
        *,
        role: Optional[Role] | None = Role.ASSISTANT,
        strict: bool = False,
        stop_on_tool_call: bool = False,
        include_thinking_channels: bool = False,
    ) -> "CompletionStreamParser":
        """Create a parser for streamed completion token suffixes.

        By default this matches Rust ``CompletionStreamOptions::default()``:
        ``role=Role.ASSISTANT`` and ``strict=False``. In that mode the parser
        expects an assistant completion suffix beginning after the already-rendered
        ``<|start|>assistant`` prefix, for example
        ``<|channel|>final<|message|>...<|return|>``.

        Pass ``role=None`` explicitly to parse a full mHarmony message stream that
        includes ``<|start|>`` and the role in each message header.

        ``push_tokens`` reports ``stop`` when it consumes an implemented terminal
        stop token (final ``<|return|>``/``<|end|>``, or tool-call ``<|call|>`` when
        ``stop_on_tool_call=True``). If the provider reports completion without
        emitting a stop token, call ``finish(absolute_start=...)`` to flush and
        finalize the open partial message.
        """
        return CompletionStreamParser(
            self,
            role=role,
            strict=strict,
            stop_on_tool_call=stop_on_tool_call,
            include_thinking_channels=include_thinking_channels,
        )

    # -- Token decoding ------------------------------------------------

    def decode_utf8(self, tokens: Sequence[int]) -> str:
        """Decode a list of tokens into a UTF-8 string. Will raise an error if the tokens result in invalid UTF-8. Use decode if you want to replace invalid UTF-8 with the unicode replacement character."""
        return self._inner.decode_utf8(list(tokens))

    def encode_timestamps_qwen(
        self,
        timestamps: Sequence[float],
        *,
        precision: int = 1,
    ) -> list[list[int]]:
        """Encode Qwen-style timestamps using Python-equivalent rounding."""
        return self._inner.encode_timestamps_qwen(list(timestamps), precision)

    def encode(
        self,
        text: str,
        *,
        allowed_special: Literal["all"] | AbstractSet[str] = frozenset(),
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> list[int]:
        """Encodes a string into tokens.

        Special tokens are artificial tokens used to unlock capabilities from a model,
        such as fill-in-the-middle. So we want to be careful about accidentally encoding special
        tokens, since they can be used to trick a model into doing something we don't want it to do.

        Hence, by default, encode will raise an error if it encounters text that corresponds
        to a special token. This can be controlled on a per-token level using the `allowed_special`
        and `disallowed_special` parameters. In particular:
        - Setting `disallowed_special` to () will prevent this function from raising errors and
          cause all text corresponding to special tokens to be encoded as natural text.
        - Setting `allowed_special` to "all" will cause this function to treat all text
          corresponding to special tokens to be encoded as special tokens.

        ```
        >>> enc.encode("hello world")
        [31373, 995]
        >>> enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
        [50256]
        >>> enc.encode("<|endoftext|>", allowed_special="all")
        [50256]
        >>> enc.encode("<|endoftext|>")
        # Raises ValueError
        >>> enc.encode("<|endoftext|>", disallowed_special=())
        [27, 91, 437, 1659, 5239, 91, 29]
        ```
        """
        allowed = (
            self.special_tokens_set
            if allowed_special == "all"
            else frozenset(allowed_special)
        )
        ignored_matches: AbstractSet[str] = frozenset()
        if disallowed_special == "all":
            disallowed = self.special_tokens_set
            ignored_matches = allowed
        else:
            disallowed = frozenset(disallowed_special)
        if disallowed and disallowed != ignored_matches and text not in ignored_matches:
            initials = _special_token_initials(disallowed)
            if "" in initials or any(initial in text for initial in initials):
                pattern = _special_token_regex(disallowed)
                if match := next(
                    (
                        match
                        for match in pattern.finditer(text)
                        if match.group() not in ignored_matches
                    ),
                    None,
                ):
                    raise_disallowed_special_token(match.group())

        try:
            return self._inner.encode(text, list(allowed))
        except UnicodeEncodeError:
            text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
            return self._inner.encode(text, list(allowed))

    def decode(self, tokens: Sequence[int], errors: str = "replace") -> str:
        """Decodes a list of tokens into a string.

        WARNING: the default behaviour of this function is lossy, since decoded bytes are not
        guaranteed to be valid UTF-8. You can use `decode_utf8` if you want to raise an error on invalid UTF-8.

        ```
        >>> enc.decode([31373, 995])
        'hello world'
        ```
        """
        data = bytes(self._inner.decode_bytes(list(tokens)))
        return data.decode("utf-8", errors=errors)

    def is_special_token(self, token: int) -> bool:
        """Returns if an individual token is a special token"""
        return self._inner.is_special_token(token)

    # -- Stop tokens --------------------------------------------------

    def stop_tokens(self) -> List[int]:
        return self._inner.stop_tokens()

    def stop_tokens_for_assistant_actions(self) -> List[int]:
        return self._inner.stop_tokens_for_assistant_actions()


class CompletionStreamParser:
    def __init__(
        self,
        encoding: HarmonyEncoding,
        *,
        role: Role | None,
        strict: bool,
        stop_on_tool_call: bool,
        include_thinking_channels: bool,
    ) -> None:
        self._inner = encoding._inner.completion_stream_parser(
            None if role is None else str(role.value),
            strict,
            stop_on_tool_call,
            include_thinking_channels,
        )

    @property
    def current_recipient(self) -> Optional[str]:
        return self._inner.current_recipient

    @property
    def current_channel(self) -> Optional[str]:
        return self._inner.current_channel

    def push_tokens(
        self, token_ids_delta: Sequence[int], *, absolute_start: int
    ) -> CompletionStreamUpdate:
        return CompletionStreamUpdate.from_dict(
            self._inner.push_tokens(list(token_ids_delta), absolute_start)
        )

    def finish(self, *, absolute_start: int) -> CompletionStreamUpdate:
        """Finalize an unterminated stream after the provider reports completion out of band."""
        return CompletionStreamUpdate.from_dict(self._inner.finish(absolute_start))


class StreamState(Enum):
    EXPECT_START = "ExpectStart"
    HEADER = "Header"
    CONTENT = "Content"


class StreamableParser:
    """Incremental parser over completion tokens."""

    def __init__(
        self,
        encoding: HarmonyEncoding,
        role: Role | None,
        *,
        strict: bool = True,
    ) -> None:
        role_str = str(role.value) if role is not None else None
        self._inner = _PyStreamableParser(encoding._inner, role_str, strict)

    def process(self, token: int) -> "StreamableParser":
        self._inner.process(token)
        return self

    def process_eos(self) -> "StreamableParser":
        self._inner.process_eos()
        return self

    @property
    def current_content(self) -> str:
        return self._inner.current_content

    @property
    def current_role(self) -> Optional[Role]:
        raw = self._inner.current_role
        return Role(raw) if raw is not None else None

    @property
    def current_content_type(self) -> Optional[str]:
        return self._inner.current_content_type

    @property
    def last_content_delta(self) -> Optional[str]:
        return self._inner.last_content_delta

    @property
    def messages(self) -> List[Message]:
        raw = self._inner.messages
        return [Message.from_dict(m) for m in json.loads(raw)]

    @property
    def tokens(self) -> List[int]:
        return self._inner.tokens

    @property
    def state_data(self) -> Dict[str, Any]:
        """Return the parsed mapping representing the parser's internal state."""
        return json.loads(self._inner.state)

    @property
    def state(self) -> StreamState:
        data = self.state_data
        return StreamState(data["state"])

    @property
    def current_recipient(self) -> Optional[str]:
        return self._inner.current_recipient

    @property
    def current_channel(self) -> Optional[str]:
        return self._inner.current_channel


# Public helper --------------------------------------------------------------


def load_harmony_encoding(
    name: str | "HarmonyEncodingName",  # type: ignore[name-defined]
    *,
    reserved_token_groups: Optional[Sequence[dict[str, JsonValue]]] = None,
    media_kinds: Optional[
        Sequence[MediaKindRegistration | Mapping[str, JsonValue]]
    ] = None,
) -> HarmonyEncoding:
    """Load an encoding, optionally registering application-defined media kinds."""

    # Allow both strings and enum values.
    if not isinstance(name, str):
        name = str(name)

    registrations = (
        None
        if media_kinds is None
        else [
            MediaKindRegistration.model_validate(registration).model_dump(mode="json")
            for registration in media_kinds
        ]
    )
    inner: _PyHarmonyEncoding = _load_harmony_encoding(
        name,
        None if reserved_token_groups is None else list(reserved_token_groups),
        registrations,
    )
    return HarmonyEncoding(inner)


class HarmonyEncodingName(str, Enum):
    """Encoding names accepted by :func:`load_harmony_encoding`."""

    HARMONY_GPT_OSS = "HarmonyGptOss"
    QWEN3_HARMONY = "Qwen3Harmony"
    QWEN35_HARMONY = "Qwen35Harmony"

    def __str__(self) -> str:  # noqa: D401
        return str(self.value)


# What should be re-exported when the user does ``from mharmony import *``?
__all__ = [
    "Role",
    "Author",
    "Content",
    "TextContent",
    "TokensContent",
    "VectorContent",
    "MediaKindRegistration",
    "MediaRef",
    "MediaContent",
    "ImageContent",
    "VideoContent",
    "PointingContent",
    "PointCollectionContent",
    "PointCollectionKind",
    "PointingKind",
    "DeveloperContent",
    "ToolDescription",
    "ReasoningEffort",
    "SystemContent",
    "SystemHint",
    "ChannelConfig",
    "ToolNamespaceConfig",
    "CanonicalChannel",
    "Message",
    "MessageHeader",
    "HeaderRenderOptions",
    "CompletionParseView",
    "CompletionStreamParser",
    "CompletionStreamUpdate",
    "StreamTextUpdate",
    "StreamStop",
    "Conversation",
    "RenderConversationConfig",
    "RenderOptions",
    "HarmonyEncoding",
    "HarmonyEncodingName",
    "load_harmony_encoding",
    "StreamableParser",
    "StreamState",
    "HarmonyError",
    "ProcessedImage",
    "format_markup_open_tag",
    "format_markup_open_tag_display_t",
    "encoding_vocab_info",
    "resolve_reserved_token_group_offsets",
    "create_qwen35_image_processor",
    "create_qwen35_mharmony_image_processor",
]
