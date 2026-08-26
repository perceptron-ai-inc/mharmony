# Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
from __future__ import annotations

import json
from typing import Any

import mharmony
import pytest
from mharmony import (
    Conversation,
    HarmonyEncoding,
    HarmonyError,
    ImageContent,
    MediaContent,
    MediaKindRegistration,
    MediaRef,
    Message,
    Role,
    VideoContent,
    create_qwen35_image_processor,
)


def test_media_content_uses_the_generic_wire_envelope() -> None:
    content = MediaContent(
        kind="depth_map",
        media_ref=MediaRef(
            id="depth-1",
            mime="application/x-depth",
            bytes_b64="AAE=",
            metadata={"sensor": "front"},
        ),
        metadata={"units": "meters"},
        source={"frame": 7},
    )

    assert content.to_dict() == {
        "type": "media",
        "kind": "depth_map",
        "media_ref": {
            "id": "depth-1",
            "mime": "application/x-depth",
            "bytes_b64": "AAE=",
            "metadata": {"sensor": "front"},
        },
        "metadata": {"units": "meters"},
        "source": {"frame": 7},
    }


def test_image_helper_serializes_as_generic_media() -> None:
    image = ImageContent(media_ref=MediaRef(id="camera-1", mime="image/png"))

    assert image.kind == "image"
    assert image.to_dict() == {
        "type": "media",
        "kind": "image",
        "media_ref": {"id": "camera-1", "mime": "image/png"},
    }
    with pytest.raises(ValueError):
        ImageContent(kind="depth_map", media_ref=MediaRef(id="depth-1"))
    with pytest.raises(ValueError):
        image.kind = "depth_map"  # type: ignore[assignment]
    assert image.kind == "image"
    with pytest.raises(ValueError):
        MediaContent(kind="image", media_ref=MediaRef(id="image-2"), source=object())
    with pytest.raises(ValueError, match="surrounding whitespace"):
        MediaContent(kind=" image ", media_ref=MediaRef(id="image-3"))


def test_video_helper_serializes_as_generic_media() -> None:
    video = VideoContent(media_ref=MediaRef(id="camera-2", mime="video/mp4"))

    assert video.kind == "video"
    assert video.to_dict() == {
        "type": "media",
        "kind": "video",
        "media_ref": {"id": "camera-2", "mime": "video/mp4"},
    }
    with pytest.raises(ValueError):
        VideoContent(kind="depth_map", media_ref=MediaRef(id="depth-1"))


def test_media_ref_id_and_runtime_registration_are_validated() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        MediaRef(id="")
    with pytest.raises(ValueError, match="surrounding whitespace"):
        MediaRef(id=" camera-1 ")

    registration = MediaKindRegistration(
        kind="depth_map",
        formatting_token="<|vector|>",
        modality="depth_frame",
    )
    assert registration.empty_payload == {}
    with pytest.raises(ValueError, match="surrounding whitespace"):
        MediaKindRegistration(
            kind="depth_map ",
            formatting_token="<|vector|>",
            modality="depth_frame",
        )


def test_opaque_json_values_round_trip_and_non_finite_values_fail_serialization() -> (
    None
):
    content = MediaContent(
        kind="image",
        media_ref=MediaRef(id="image-1", metadata=["camera", 1]),
        metadata="opaque",
        source={"score": float("nan")},
    )
    conversation = Conversation.from_messages(
        [Message.from_role_and_content(Role.USER, content)]
    )

    assert content.media_ref.metadata == ["camera", 1]
    assert content.metadata == "opaque"
    with pytest.raises(ValueError, match="Out of range float values"):
        conversation.to_json()


def test_conversation_round_trip_preserves_generic_media() -> None:
    conversation = Conversation.from_messages(
        [
            Message.from_role_and_contents(
                Role.USER,
                [
                    ImageContent(
                        media_ref=MediaRef(id="image-1"), source={"path": "one.png"}
                    ),
                    VideoContent(media_ref=MediaRef(id="video-1")),
                    MediaContent(kind="depth_map", media_ref=MediaRef(id="depth-1")),
                ],
            )
        ]
    )

    payload = conversation.to_json()
    restored = Conversation.from_json(payload)

    assert isinstance(payload, str)
    assert json.loads(payload) == conversation.to_dict()
    assert isinstance(restored.messages[0].content[0], ImageContent)
    assert isinstance(restored.messages[0].content[1], VideoContent)
    assert type(restored.messages[0].content[2]) is MediaContent
    assert restored.to_dict() == conversation.to_dict()


def test_public_media_exports_are_present_and_resolvable() -> None:
    media_exports = {
        "MediaKindRegistration",
        "MediaRef",
        "MediaContent",
        "ImageContent",
        "VideoContent",
    }

    assert media_exports <= set(mharmony.__all__)
    assert all(hasattr(mharmony, name) for name in media_exports)
    assert len(mharmony.__all__) == len(set(mharmony.__all__))


def test_runtime_registration_is_validated_and_forwarded(monkeypatch: Any) -> None:
    calls: list[tuple[Any, ...]] = []

    class FakeNativeEncoding:
        pass

    def fake_load(*args: Any) -> FakeNativeEncoding:
        calls.append(args)
        return FakeNativeEncoding()

    monkeypatch.setattr(mharmony, "_load_harmony_encoding", fake_load)
    registration = MediaKindRegistration(
        kind="depth_map",
        formatting_token="<|vector|>",
        modality="depth_frame",
        empty_payload={"values": []},
    )

    mharmony.load_harmony_encoding(
        "HarmonyGptOss",
        media_kinds=[registration],
    )

    assert calls == [
        (
            "HarmonyGptOss",
            None,
            [
                {
                    "kind": "depth_map",
                    "formatting_token": "<|vector|>",
                    "modality": "depth_frame",
                    "empty_payload": {"values": []},
                }
            ],
        )
    ]


def test_generic_processor_mapping_is_forwarded_to_native_binding() -> None:
    calls: list[tuple[Any, ...]] = []

    class FakeNativeEncoding:
        def render_conversation_multimodal_with_processors(
            self, *args: Any
        ) -> dict[str, Any]:
            calls.append(args)
            return {"events": []}

    def depth_processor(source: Any, config: dict[str, Any]) -> object:
        return (source, config)

    def image_processor(source: bytes, config: dict[str, Any]) -> object:
        return (source, config)

    def timestamp_processor(text: str, role: str, context: dict[str, Any]) -> str:
        return text

    def depth_validator(media: dict[str, Any], config: dict[str, Any]) -> None:
        return None

    conversation = Conversation.from_messages(
        [Message.from_role_and_content(Role.USER, "look")]
    )
    encoding = HarmonyEncoding(FakeNativeEncoding())  # type: ignore[arg-type]

    rendered = encoding.render_conversation_multimodal_with_processors(
        conversation,
        {"patch_size": 14},
        {"depth_map": depth_processor},
        media_validators={"depth_map": depth_validator},
        image_processor=image_processor,
        timestamp_processor=timestamp_processor,
    )

    assert rendered == {"events": []}
    assert json.loads(calls[0][0]) == conversation.to_dict()
    assert calls[0][1:] == (
        {"patch_size": 14},
        {"depth_map": depth_processor},
        image_processor,
        timestamp_processor,
        {"depth_map": depth_validator},
    )


def test_qwen_factory_defaults_are_merged_before_native_normalization() -> None:
    calls: list[tuple[Any, ...]] = []

    class FakeNativeEncoding:
        def render_conversation_multimodal_with_processors(
            self, *args: Any
        ) -> dict[str, Any]:
            calls.append(args)
            return {"events": []}

    processor = create_qwen35_image_processor(
        patch_size=8,
        max_num_patches=576,
        pixel_shuffle_scale=2,
        temporal_patch_size=2,
    )
    encoding = HarmonyEncoding(FakeNativeEncoding())  # type: ignore[arg-type]
    conversation = Conversation.from_messages(
        [Message.from_role_and_content(Role.USER, "look")]
    )

    encoding.render_conversation_multimodal_with_processors(
        conversation,
        {"max_num_patches": 128},
        {"image": processor},
    )

    assert calls[0][1] == {
        "patch_size": 8,
        "max_num_patches": 128,
        "min_num_patches": None,
        "max_pixels": None,
        "min_pixels": None,
        "max_decode_pixels": 64_000_000,
        "pixel_shuffle_scale": 2,
        "temporal_patch_size": 2,
    }


def test_cached_special_tokens_cannot_be_mutated() -> None:
    class FakeNativeEncoding:
        def special_tokens(self) -> list[str]:
            return ["<|special|>"]

        def encode(self, text: str, allowed_special: list[str]) -> list[int]:
            assert text == "<|special|>"
            assert allowed_special == ["<|special|>"]
            return [42]

    encoding = HarmonyEncoding(FakeNativeEncoding())  # type: ignore[arg-type]

    assert encoding.special_tokens_set == frozenset({"<|special|>"})
    assert not hasattr(encoding.special_tokens_set, "clear")
    with pytest.raises(HarmonyError, match="disallowed special token"):
        encoding.encode("<|special|>")
    assert encoding.encode("<|special|>", allowed_special={"<|special|>"}) == [42]


def test_plain_encode_skips_impossible_special_token_regex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeNativeEncoding:
        def special_tokens(self) -> list[str]:
            return ["<|special|>", "<|reserved_1|>"]

        def encode(self, text: str, allowed_special: list[str]) -> list[int]:
            assert text == "plain text"
            assert allowed_special == []
            return [7]

    def fail_if_compiled(tokens: frozenset[str]) -> None:
        raise AssertionError(f"unexpected special-token regex for {tokens}")

    monkeypatch.setattr(mharmony, "_special_token_regex", fail_if_compiled)
    encoding = HarmonyEncoding(FakeNativeEncoding())  # type: ignore[arg-type]

    assert encoding.encode("plain text") == [7]
