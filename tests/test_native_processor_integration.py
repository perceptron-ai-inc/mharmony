# Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
from __future__ import annotations

import base64
from typing import Any

import pytest
from mharmony import (
    Conversation,
    HarmonyEncoding,
    HarmonyEncodingName,
    HarmonyError,
    ImageContent,
    MediaContent,
    MediaKindRegistration,
    MediaRef,
    Message,
    Role,
    SystemContent,
    load_harmony_encoding,
    resolve_reserved_token_group_offsets,
)


@pytest.fixture(scope="module")
def encoding() -> HarmonyEncoding:
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def _conversation(content: MediaContent) -> Conversation:
    return Conversation.from_messages(
        [Message.from_role_and_content(Role.USER, content)]
    )


def _image_event(rendered: dict[str, Any]) -> dict[str, Any]:
    return next(
        event
        for event in rendered["stream"]["events"]
        if event["modality"] == "image_frame"
    )


def test_native_mapping_prefers_source_and_mapping_over_alias(
    encoding: HarmonyEncoding,
) -> None:
    mapped_calls: list[tuple[Any, dict[str, Any]]] = []
    alias_calls: list[tuple[Any, dict[str, Any]]] = []

    def mapped(source: Any, config: dict[str, Any]) -> dict[str, Any]:
        mapped_calls.append((source, config))
        return {"data": [0.25, 0.75], "shape": [1, 1, 2]}

    def alias(source: Any, config: dict[str, Any]) -> dict[str, Any]:
        alias_calls.append((source, config))
        return {"data": [1.0], "shape": [1, 1, 1]}

    content = ImageContent(
        media_ref=MediaRef(
            id="camera-1",
            bytes_b64=base64.b64encode(b"ignored-inline-bytes").decode(),
        ),
        metadata={"camera": "front"},
        source={"asset_id": "frame-7"},
    )
    rendered = encoding.render_conversation_multimodal_with_processors(
        _conversation(content),
        media_processors={"image": mapped},
        image_processor=alias,
    )

    assert len(mapped_calls) == 1
    assert not alias_calls
    source, config = mapped_calls[0]
    assert source == {"asset_id": "frame-7"}
    assert config["metadata"] == {"camera": "front"}
    assert config["patch_size"] == 16
    event = _image_event(rendered)
    assert event["data"] == {"Image": {"data": [0.25, 0.75], "shape": [1, 1, 2]}}
    assert event["tags"] == {"media_ref_id": "camera-1", "media_ref_type": "image"}


def test_native_mapping_decodes_inline_base64(encoding: HarmonyEncoding) -> None:
    sources: list[Any] = []

    def mapped(source: Any, config: dict[str, Any]) -> dict[str, Any]:
        sources.append(source)
        return {"data": [1.0], "shape": [1, 1, 1]}

    content = ImageContent(
        media_ref=MediaRef(
            id="camera-2", bytes_b64=base64.b64encode(b"\x00\xffframe").decode()
        )
    )
    encoding.render_conversation_multimodal_with_processors(
        _conversation(content), media_processors={"image": mapped}
    )

    assert sources == [b"\x00\xffframe"]


def test_inline_media_limit_rejects_before_processor(encoding: HarmonyEncoding) -> None:
    processor_called = False

    def mapped(source: Any, config: dict[str, Any]) -> dict[str, Any]:
        nonlocal processor_called
        processor_called = True
        return {"data": [1.0], "shape": [1, 1, 1]}

    content = ImageContent(
        media_ref=MediaRef(
            id="camera-limited", bytes_b64=base64.b64encode(b"frame").decode()
        )
    )
    with pytest.raises(ValueError, match="inline media exceeds the 4-byte limit"):
        encoding.render_conversation_multimodal_with_processors(
            _conversation(content),
            {"max_inline_media_bytes": 4},
            media_processors={"image": mapped},
        )

    assert not processor_called


def test_native_mapping_binds_duplicate_media_ids_by_occurrence(
    encoding: HarmonyEncoding,
) -> None:
    sources: list[Any] = []

    def mapped(source: Any, config: dict[str, Any]) -> dict[str, Any]:
        sources.append(source)
        return {"data": [float(source["value"])], "shape": [1, 1, 1]}

    conversation = Conversation.from_messages(
        [
            Message.from_role_and_contents(
                Role.USER,
                [
                    ImageContent(
                        media_ref=MediaRef(id="reused-id"),
                        source={"value": 1},
                    ),
                    ImageContent(
                        media_ref=MediaRef(id="reused-id"),
                        source={"value": 2},
                    ),
                ],
            )
        ]
    )
    rendered = encoding.render_conversation_multimodal_with_processors(
        conversation,
        media_processors={"image": mapped},
    )

    assert sources == [{"value": 1}, {"value": 2}]
    image_events = [
        event
        for event in rendered["stream"]["events"]
        if event["modality"] == "image_frame"
    ]
    assert [event["data"]["Image"]["data"] for event in image_events] == [
        [1.0],
        [2.0],
    ]


def test_duplicate_media_ids_bind_after_default_analysis_filtering(
    encoding: HarmonyEncoding,
) -> None:
    sources: list[Any] = []

    def mapped(source: Any, config: dict[str, Any]) -> dict[str, Any]:
        sources.append(source)
        return {"data": [float(source["value"])], "shape": [1, 1, 1]}

    conversation = Conversation.from_messages(
        [
            Message.from_role_and_content(
                Role.ASSISTANT,
                ImageContent(
                    media_ref=MediaRef(id="reused-id"),
                    source={"value": 1},
                ),
            ).with_channel("analysis"),
            Message.from_role_and_content(
                Role.ASSISTANT,
                ImageContent(
                    media_ref=MediaRef(id="reused-id"),
                    source={"value": 2},
                ),
            ).with_channel("final"),
        ]
    )
    rendered = encoding.render_conversation_multimodal_with_processors(
        conversation,
        media_processors={"image": mapped},
    )

    assert sources == [{"value": 2}]
    assert _image_event(rendered)["data"]["Image"]["data"] == [2.0]


def test_preprocess_config_none_and_empty_use_the_same_defaults(
    encoding: HarmonyEncoding,
) -> None:
    content = ImageContent(
        media_ref=MediaRef(id="config-defaults"),
        source={"asset_id": "frame-defaults"},
    )
    conversation = _conversation(content)

    assert encoding.render_conversation_multimodal(
        conversation, None
    ) == encoding.render_conversation_multimodal(conversation, {})

    configs: list[dict[str, Any]] = []

    def process(source: Any, config: dict[str, Any]) -> dict[str, Any]:
        configs.append(config)
        return {"data": [1.0], "shape": [1, 1, 1]}

    for config in (None, {}):
        encoding.render_conversation_multimodal_with_processors(
            conversation,
            preprocess_config=config,
            media_processors={"image": process},
        )

    assert configs[0] == configs[1]
    assert configs[0]["max_num_patches"] == 3600


def test_preprocess_config_preserves_explicit_optional_none(
    encoding: HarmonyEncoding,
) -> None:
    configs: list[dict[str, Any]] = []
    content = ImageContent(
        media_ref=MediaRef(id="config-none"),
        source={"asset_id": "frame-none"},
    )

    def process(source: Any, config: dict[str, Any]) -> dict[str, Any]:
        configs.append(config)
        return {"data": [1.0], "shape": [1, 1, 1]}

    encoding.render_conversation_multimodal_with_processors(
        _conversation(content),
        preprocess_config={"max_num_patches": None, "min_num_patches": None},
        media_processors={"image": process},
    )

    assert configs[0]["max_num_patches"] is None
    assert configs[0]["min_num_patches"] is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("patch_size", "16"),
        ("patch_size", True),
        ("max_num_patches", "3600"),
        ("min_num_patches", 1.5),
        ("pixel_shuffle_scale", None),
    ],
)
def test_preprocess_config_rejects_bad_types(
    encoding: HarmonyEncoding, name: str, value: Any
) -> None:
    content = ImageContent(media_ref=MediaRef(id="config-invalid"))
    conversation = _conversation(content)
    config = {name: value}

    with pytest.raises(ValueError, match=rf"{name}.*integer"):
        encoding.render_conversation_multimodal(conversation, config)
    with pytest.raises(ValueError, match=rf"{name}.*integer"):
        encoding.render_conversation_multimodal_with_processors(
            conversation, preprocess_config=config
        )


def test_image_processor_prefers_consuming_numpy_conversion(
    encoding: HarmonyEncoding,
) -> None:
    conversions: list[str] = []

    class Processed:
        data = [1.0]
        shape = [1, 1, 1]

        def into_numpy(self) -> list[float]:
            conversions.append("into")
            return self.data

        def to_numpy(self) -> list[float]:
            conversions.append("to")
            return self.data

    content = ImageContent(
        media_ref=MediaRef(id="camera-3"),
        source={"asset_id": "frame-9"},
    )
    rendered = encoding.render_conversation_multimodal_with_processors(
        _conversation(content),
        media_processors={"image": lambda source, config: Processed()},
    )

    assert conversions == ["into"]
    assert _image_event(rendered)["data"] == {
        "Image": {"data": [1.0], "shape": [1, 1, 1]}
    }


@pytest.mark.parametrize(
    ("data", "shape", "error"),
    [
        ([], [-1, 1, 1], "dimensions must be non-negative"),
        ([], [0, 1, 1], "all positive or \\[0, 0, 0\\]"),
        ([], [2**62, 4, 2], "element count overflows"),
        (["not-a-number"], [1, 1, 1], "numeric array or sequence"),
        (object(), [1, 1, 1], "numeric array or sequence"),
        ([1.0], [1, 1, 2], "data length 1.*shape product 2"),
    ],
)
def test_image_processor_rejects_malformed_mapping_results(
    encoding: HarmonyEncoding,
    data: Any,
    shape: list[int],
    error: str,
) -> None:
    content = ImageContent(
        media_ref=MediaRef(id="malformed-result"),
        source={"asset_id": "frame-invalid"},
    )

    with pytest.raises(ValueError, match=error):
        encoding.render_conversation_multimodal_with_processors(
            _conversation(content),
            media_processors={
                "image": lambda source, config: {"data": data, "shape": shape}
            },
        )


def test_image_processor_preserves_all_zero_empty_sentinel(
    encoding: HarmonyEncoding,
) -> None:
    content = ImageContent(
        media_ref=MediaRef(id="empty-result"),
        source={"asset_id": "invalid-image"},
    )

    rendered = encoding.render_conversation_multimodal_with_processors(
        _conversation(content),
        media_processors={
            "image": lambda source, config: {"data": [], "shape": [0, 0, 0]}
        },
    )

    assert _image_event(rendered)["data"] == {"Image": {"data": [], "shape": [0, 0, 0]}}


def test_native_mapping_rejects_missing_and_noncallable_callbacks(
    encoding: HarmonyEncoding,
) -> None:
    content = ImageContent(
        media_ref=MediaRef(id="camera-4", bytes_b64=base64.b64encode(b"frame").decode())
    )
    conversation = _conversation(content)

    with pytest.raises(ValueError, match="no processor callback"):
        encoding.render_conversation_multimodal_with_processors(conversation)
    with pytest.raises(ValueError, match="must be callable"):
        encoding.render_conversation_multimodal_with_processors(
            conversation,
            media_processors={"unused": object()},  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError, match="image processor must be callable"):
        encoding.render_conversation_multimodal_with_processors(
            conversation,
            image_processor=object(),  # type: ignore[arg-type]
        )


def test_runtime_media_registration_validates_before_processing() -> None:
    encoding = load_harmony_encoding(
        HarmonyEncodingName.HARMONY_GPT_OSS,
        media_kinds=[
            MediaKindRegistration(
                kind="depth_map",
                formatting_token="<|vector|>",
                modality="depth_frame",
                empty_payload={"values": []},
            )
        ],
    )
    calls: list[tuple[str, Any]] = []

    def validate(media: dict[str, Any], config: dict[str, Any]) -> None:
        calls.append(("validate", media))
        assert media == {
            "type": "media",
            "kind": "depth_map",
            "media_ref": {"id": "depth-1"},
            "metadata": {"units": "meters"},
            "source": {"frame": 7},
        }
        assert config["metadata"] == {"units": "meters"}

    def process(source: Any, config: dict[str, Any]) -> dict[str, Any]:
        calls.append(("process", source))
        assert config["metadata"] == {"units": "meters"}
        return {"values": [1.0, 2.0]}

    content = MediaContent(
        kind="depth_map",
        media_ref=MediaRef(id="depth-1"),
        metadata={"units": "meters"},
        source={"frame": 7},
    )
    rendered = encoding.render_conversation_multimodal_with_processors(
        _conversation(content),
        media_processors={"depth_map": process},
        media_validators={"depth_map": validate},
    )

    assert [kind for kind, _ in calls] == ["validate", "process"]
    event = next(
        event
        for event in rendered["stream"]["events"]
        if event["modality"] == "depth_frame"
    )
    assert event["data"] == {
        "Media": {
            "kind": "depth_map",
            "payload": {"values": [1.0, 2.0]},
        }
    }
    assert event["tags"] == {
        "media_ref_id": "depth-1",
        "media_ref_type": "depth_map",
    }


def test_media_validator_can_reject_without_calling_processor() -> None:
    encoding = load_harmony_encoding(
        HarmonyEncodingName.HARMONY_GPT_OSS,
        media_kinds=[
            MediaKindRegistration(
                kind="depth_map",
                formatting_token="<|vector|>",
                modality="depth_frame",
            )
        ],
    )
    processor_called = False

    def reject(media: dict[str, Any], config: dict[str, Any]) -> bool:
        return False

    def process(source: Any, config: dict[str, Any]) -> dict[str, Any]:
        nonlocal processor_called
        processor_called = True
        return {}

    content = MediaContent(
        kind="depth_map",
        media_ref=MediaRef(id="depth-2"),
        source={"frame": 8},
    )
    with pytest.raises(ValueError, match="rejected media ref"):
        encoding.render_conversation_multimodal_with_processors(
            _conversation(content),
            media_processors={"depth_map": process},
            media_validators={"depth_map": reject},
        )

    assert not processor_called


def test_media_validator_mapping_rejects_noncallable_values(
    encoding: HarmonyEncoding,
) -> None:
    content = ImageContent(media_ref=MediaRef(id="camera-5"))

    with pytest.raises(ValueError, match="validator.*must be callable"):
        encoding.render_conversation_multimodal_with_processors(
            _conversation(content),
            media_validators={"image": object()},  # type: ignore[dict-item]
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("name", 1, "name must be a string or None"),
        ("tokenizer", [], "tokenizer must be a string or None"),
        ("size", "1", "size must be a non-negative integer"),
        ("size", None, "size must be a non-negative integer"),
        ("size", True, "size must be a non-negative integer"),
        ("offset", "200200", "offset must be an integer or None"),
        ("offset", False, "offset must be an integer or None"),
    ],
)
def test_reserved_token_groups_reject_present_wrong_typed_fields(
    field: str, value: Any, error: str
) -> None:
    group: dict[str, Any] = {"name": "coordinates", "size": 1}
    group[field] = value

    with pytest.raises(ValueError, match=error):
        resolve_reserved_token_group_offsets(
            HarmonyEncodingName.HARMONY_GPT_OSS.value,
            [group],
        )


def test_reserved_token_groups_preserve_optional_missing_and_none() -> None:
    assert resolve_reserved_token_group_offsets(
        HarmonyEncodingName.HARMONY_GPT_OSS.value,
        [
            {"name": "first", "tokenizer": None, "size": 1, "offset": None},
            {"name": None, "tokenizer": "second", "size": 1},
        ],
    ) == [(200200, 1), (200201, 1)]


def test_render_errors_use_the_documented_exception_boundary(
    encoding: HarmonyEncoding,
) -> None:
    native = encoding._inner
    valid_message = Message.from_role_and_content(Role.USER, "hello")
    valid_conversation = Conversation.from_messages([valid_message])
    invalid_role_message = Message.from_role_and_content(
        Role.USER, SystemContent.empty()
    )
    invalid_role_conversation = Conversation.from_messages([invalid_role_message])

    with pytest.raises(ValueError, match="invalid message JSON"):
        native.render("not JSON", {})
    with pytest.raises(ValueError, match="invalid conversation JSON"):
        native.render_conversation("not JSON", {})
    with pytest.raises(ValueError, match="conversation_has_function_tools.*boolean"):
        native.render(
            valid_message.to_json(),
            {"conversation_has_function_tools": "yes"},
        )
    with pytest.raises(ValueError, match="auto_drop_analysis.*boolean"):
        native.render_conversation(
            valid_conversation.to_json(),
            {"auto_drop_analysis": "yes"},
        )

    with pytest.raises(HarmonyError, match="SystemContent may only appear"):
        native.render(invalid_role_message.to_json(), {})
    with pytest.raises(HarmonyError, match="SystemContent may only appear"):
        native.render_conversation(invalid_role_conversation.to_json(), {})
