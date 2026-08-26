# Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
from __future__ import annotations

# ruff: noqa: E402 -- optional dependencies must be checked before their imports.

import io
import struct
import zlib

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("transformers")
pytest.importorskip("PIL")

from mharmony import (
    create_qwen35_image_processor as root_create_qwen35_image_processor,
)
from mharmony import (
    create_qwen35_mharmony_image_processor as root_create_qwen35_mharmony_image_processor,
)
from mharmony.processors import (
    create_qwen35_image_processor,
    create_qwen35_mharmony_image_processor,
)
from mharmony.processors import qwen35
from mharmony.processors.qwen35 import process_qwen35_image_bytes
from PIL import Image


def _encode_png(array: np.ndarray, *, orientation: int | None = None) -> bytes:
    image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    exif = image.getexif()
    if orientation is not None:
        exif[0x0112] = orientation
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", exif=exif)
    return buffer.getvalue()


def _rewrite_png_dimensions(payload: bytes, width: int, height: int) -> bytes:
    """Change only the IHDR dimensions/CRC, without allocating image pixels."""
    rewritten = bytearray(payload)
    rewritten[16:24] = struct.pack(">II", width, height)
    rewritten[29:33] = struct.pack(">I", zlib.crc32(rewritten[12:29]))
    return bytes(rewritten)


def test_factory_alias_and_callback_config_overrides() -> None:
    assert create_qwen35_mharmony_image_processor is create_qwen35_image_processor
    assert root_create_qwen35_image_processor is create_qwen35_image_processor
    assert root_create_qwen35_mharmony_image_processor is create_qwen35_image_processor

    processor = create_qwen35_image_processor(
        patch_size=4,
        max_num_patches=100,
        pixel_shuffle_scale=1,
    )
    image_bytes = _encode_png(np.zeros((8, 16, 3), dtype=np.uint8))

    result = processor(
        image_bytes,
        {
            "patch_size": 8,
            "max_num_patches": 100,
            "pixel_shuffle_scale": 1,
            "temporal_patch_size": 2,
        },
    )

    assert result.shape == [1, 2, 384]
    assert result.data.shape == (1 * 2 * 384,)
    assert result.data.dtype == np.float32


def test_patch_layout_matches_qwen_channel_and_spatial_order() -> None:
    # Each 2x2 patch has a distinct RGB value. With merge_size=2, the Qwen
    # layout traverses the four spatial patches row-major and keeps each
    # patch's values in channel-major order.
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    patch_colors = (
        (0, 64, 255),
        (32, 96, 224),
        (128, 160, 192),
        (255, 16, 48),
    )
    for index, color in enumerate(patch_colors):
        row, column = divmod(index, 2)
        image[row * 2 : (row + 1) * 2, column * 2 : (column + 1) * 2] = color

    result = process_qwen35_image_bytes(
        _encode_png(image),
        patch_size=2,
        merge_size=2,
        temporal_patch_size=1,
        max_num_patches=16,
        min_num_patches=None,
    )

    expected = np.stack(
        [
            np.concatenate(
                [np.full(4, channel / 127.5 - 1, dtype=np.float32) for channel in color]
            )
            for color in patch_colors
        ]
    ).reshape(2, 2, 12)
    assert result.shape == [2, 2, 12]
    np.testing.assert_allclose(
        result.data.reshape(result.shape), expected, rtol=0, atol=1e-6
    )


def test_exif_orientation_is_applied_before_patchifying() -> None:
    # Orientation 6 rotates the 8x16 source to 16x8 (width x height), so its
    # patch grid changes from [2, 1] to [1, 2].
    image = np.zeros((16, 8, 3), dtype=np.uint8)
    result = process_qwen35_image_bytes(
        _encode_png(image, orientation=6),
        patch_size=8,
        merge_size=1,
        temporal_patch_size=1,
        max_num_patches=100,
        min_num_patches=None,
    )

    assert result.shape == [1, 2, 192]


def test_invalid_image_returns_an_empty_payload() -> None:
    result = process_qwen35_image_bytes(
        b"not an image",
        patch_size=16,
        merge_size=1,
        temporal_patch_size=1,
        max_num_patches=576,
        min_num_patches=None,
    )

    assert result.shape == [0, 0, 0]
    assert result.data.shape == (0,)
    assert result.data.dtype == np.float32


def test_decode_pixel_limit_rejects_oversized_header_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiny_png = _encode_png(np.zeros((1, 1, 3), dtype=np.uint8))
    oversized_header = _rewrite_png_dimensions(tiny_png, 2_000, 2_000)

    def fail_if_decode_continues(_image: Image.Image) -> Image.Image:
        pytest.fail("oversized image reached orientation and RGB conversion")

    monkeypatch.setattr(qwen35, "_apply_exif_orientation", fail_if_decode_continues)

    with pytest.raises(ValueError, match="decode pixel limit"):
        process_qwen35_image_bytes(
            oversized_header,
            patch_size=16,
            merge_size=1,
            temporal_patch_size=1,
            max_num_patches=576,
            min_num_patches=None,
            max_decode_pixels=1_000_000,
        )


def test_factory_forwards_decode_pixel_limit_from_call_config() -> None:
    tiny_png = _encode_png(np.zeros((1, 1, 3), dtype=np.uint8))
    oversized_header = _rewrite_png_dimensions(tiny_png, 2_000, 2_000)
    processor = create_qwen35_image_processor(max_decode_pixels=5_000_000)

    with pytest.raises(ValueError, match="4000000 > 1000000 pixels"):
        processor(oversized_header, {"max_decode_pixels": 1_000_000})


def test_decode_pixel_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_decode_pixels must be positive"):
        process_qwen35_image_bytes(
            _encode_png(np.zeros((1, 1, 3), dtype=np.uint8)),
            patch_size=16,
            merge_size=1,
            temporal_patch_size=1,
            max_num_patches=576,
            min_num_patches=None,
            max_decode_pixels=0,
        )
