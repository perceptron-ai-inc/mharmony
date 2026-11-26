# Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
"""Qwen3.5-compatible image preprocessing for mharmony callbacks."""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from PIL import Image as PILImage  # type: ignore[import-not-found]

_EXIF_ORIENTATION_TAG = 0x0112
_DEFAULT_MAX_DECODE_PIXELS = 64_000_000
_BYTES_TYPES = (bytes, bytearray, memoryview)


class _DecodePixelLimitError(ValueError):
    pass


@dataclass(slots=True)
class Qwen35ProcessedImage:
    """Flattened image patches and their three-dimensional logical shape."""

    data: np.ndarray
    shape: list[int]


Qwen35ImageProcessor = Callable[[Any, dict[str, Any]], Qwen35ProcessedImage]


def _apply_exif_orientation(image: PILImage.Image) -> PILImage.Image:
    """Apply EXIF orientation, tolerating malformed metadata."""
    from PIL import ImageOps

    try:
        if image.getexif().get(_EXIF_ORIENTATION_TAG, 1) == 1:
            return image
        return ImageOps.exif_transpose(image)
    except Exception:  # noqa: BLE001 - malformed EXIF raises several Pillow/backend exception types.
        return image


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int | None,
    max_pixels: int | None,
) -> tuple[int, int]:
    """Return Qwen-aligned dimensions subject to optional pixel bounds."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        smart_resize as hf_smart_resize,
    )

    if min(height, width) <= 0:
        raise ValueError(f"image dimensions must be positive, got {(height, width)}")
    min_pixels = 0 if min_pixels is None else int(min_pixels)
    max_pixels = (2**63 - 1) if max_pixels is None else int(max_pixels)
    resized_height, resized_width = hf_smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return int(resized_height), int(resized_width)


def _patch_count_to_pixels(count: int | None, patch_size: int) -> int | None:
    if count is None:
        return None
    return int(count) * patch_size * patch_size


def _normalize_qwen_image(
    images: torch.Tensor,
    *,
    rescale_factor: float,
    image_mean: tuple[float, float, float],
    image_std: tuple[float, float, float],
) -> torch.Tensor:
    import torch
    from torchvision.transforms.v2 import functional as tvF  # type: ignore[import-untyped]

    mean = [float(value / rescale_factor) for value in image_mean]
    std = [float(value / rescale_factor) for value in image_std]
    return tvF.normalize(images.to(dtype=torch.float32), mean=mean, std=std)


def process_qwen35_image_bytes(
    image_bytes: bytes,
    *,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int,
    max_num_patches: int | None,
    min_num_patches: int | None,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
    max_decode_pixels: int = _DEFAULT_MAX_DECODE_PIXELS,
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    rescale_factor: float = 1 / 255,
) -> Qwen35ProcessedImage:
    """Decode and patchify one image using the Qwen3.5 vision layout."""
    import numpy as np
    from PIL import Image as PILImage

    if max_decode_pixels <= 0:
        raise ValueError("max_decode_pixels must be positive")

    try:
        with PILImage.open(io.BytesIO(image_bytes)) as image:
            width, height = int(image.width), int(image.height)
            decoded_pixels = width * height
            if decoded_pixels > max_decode_pixels:
                raise _DecodePixelLimitError(
                    f"image dimensions {width}x{height} exceed the decode pixel limit: "
                    f"{decoded_pixels} > {max_decode_pixels} pixels"
                )
            pil_image = _apply_exif_orientation(image).convert("RGB")
        image_array = np.array(pil_image, copy=True)
    except PILImage.DecompressionBombError as error:
        raise ValueError("image exceeds Pillow's decode pixel safety limit") from error
    except _DecodePixelLimitError:
        raise
    except Exception:  # noqa: BLE001 - preserve the processor contract for any decode failure.
        empty = np.zeros((0,), dtype=np.float32)
        return Qwen35ProcessedImage(data=empty, shape=[0, 0, 0])

    return _patchify_rgb(
        image_array,
        patch_size=patch_size,
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
        max_num_patches=max_num_patches,
        min_num_patches=min_num_patches,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        image_mean=image_mean,
        image_std=image_std,
        rescale_factor=rescale_factor,
    )


def process_qwen35_image_array(
    image: np.ndarray,
    *,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int,
    max_num_patches: int | None,
    min_num_patches: int | None,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    rescale_factor: float = 1 / 255,
) -> Qwen35ProcessedImage:
    """Patchify one RGB ``uint8`` array shaped ``[height, width, 3]``.

    This is the transform ``process_qwen35_image_bytes`` applies after decoding,
    so an array and a lossless encoding of the same pixels produce identical
    patches. EXIF orientation and the decode pixel limit apply only to encoded
    bytes; the array is used as given.
    """
    import numpy as np

    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            "image array must be uint8 with shape [height, width, 3], got "
            f"dtype {array.dtype} and shape {list(array.shape)}"
        )
    # torch.from_numpy needs a writable array, and the tensor must not alias
    # the caller's buffer.
    return _patchify_rgb(
        np.array(array, copy=True),
        patch_size=patch_size,
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
        max_num_patches=max_num_patches,
        min_num_patches=min_num_patches,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        image_mean=image_mean,
        image_std=image_std,
        rescale_factor=rescale_factor,
    )


def _patchify_rgb(
    image_array: np.ndarray,
    *,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int,
    max_num_patches: int | None,
    min_num_patches: int | None,
    max_pixels: int | None,
    min_pixels: int | None,
    image_mean: tuple[float, float, float],
    image_std: tuple[float, float, float],
    rescale_factor: float,
) -> Qwen35ProcessedImage:
    import numpy as np
    import torch
    from torchvision.transforms import InterpolationMode  # type: ignore[import-untyped]
    from torchvision.transforms.v2 import functional as tvF  # type: ignore[import-untyped]

    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    _, _, height, width = image_tensor.shape

    target_height, target_width = smart_resize(
        height,
        width,
        factor=patch_size * merge_size,
        min_pixels=(
            min_pixels
            if min_pixels is not None
            else _patch_count_to_pixels(min_num_patches, patch_size)
        ),
        max_pixels=(
            max_pixels
            if max_pixels is not None
            else _patch_count_to_pixels(max_num_patches, patch_size)
        ),
    )

    # Extreme aspect ratios can round one side to zero. Qwen still needs at
    # least one merged patch along each axis.
    min_side = patch_size * merge_size
    target_height = max(target_height, min_side)
    target_width = max(target_width, min_side)
    image_tensor = tvF.resize(
        image_tensor,
        size=[target_height, target_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    image_tensor = _normalize_qwen_image(
        image_tensor,
        rescale_factor=rescale_factor,
        image_mean=image_mean,
        image_std=image_std,
    )

    patches = image_tensor.unsqueeze(1)
    if patches.shape[1] % temporal_patch_size != 0:
        repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
        patches = torch.cat([patches, repeats], dim=1)

    batch_size, grid_t, channels = patches.shape[:3]
    grid_t //= temporal_patch_size
    grid_h = target_height // patch_size
    grid_w = target_width // patch_size

    patches = patches.view(
        batch_size,
        grid_t,
        temporal_patch_size,
        channels,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    flat = patches.reshape(
        batch_size,
        grid_t * grid_h * grid_w,
        channels * temporal_patch_size * patch_size * patch_size,
    )
    assert batch_size == 1, f"Expected single-image batch, got {batch_size}"
    patch_dim = int(flat.shape[-1])
    data = (
        flat[0]
        .contiguous()
        .view(grid_t, grid_h, grid_w, patch_dim)
        .squeeze(0)
        .cpu()
        .numpy()
        .reshape(-1)
        .astype(np.float32, copy=False)
    )
    return Qwen35ProcessedImage(data=data, shape=[grid_h, grid_w, patch_dim])


def create_qwen35_image_processor(
    *,
    patch_size: int = 16,
    max_num_patches: int | None = 576,
    min_num_patches: int | None = None,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
    max_decode_pixels: int = _DEFAULT_MAX_DECODE_PIXELS,
    pixel_shuffle_scale: int = 1,
    temporal_patch_size: int = 1,
    resolve_source: Callable[[Any], bytes | np.ndarray] | None = None,
) -> Qwen35ImageProcessor:
    """Create an image callback for ``render_conversation_multimodal_with_processors``.

    The callback receives inline image bytes, or the envelope's ``source`` when
    one is present. ``resolve_source`` maps a source to encoded image bytes or
    to an RGB ``uint8`` array shaped ``[height, width, 3]``; an array is
    patchified without any image codec. A non-bytes source is rejected when no
    resolver is given.
    """

    def image_processor(source: Any, config: dict[str, Any]) -> Qwen35ProcessedImage:
        image = source
        if not isinstance(image, _BYTES_TYPES):
            if resolve_source is None:
                raise TypeError(
                    "the Qwen3.5 image processor received a "
                    f"{type(source).__name__} source; pass resolve_source to "
                    "create_qwen35_image_processor to map sources to image "
                    "bytes or an RGB array"
                )
            image = resolve_source(source)
        options: dict[str, Any] = {
            "patch_size": int(config.get("patch_size") or patch_size),
            "merge_size": int(
                config.get("pixel_shuffle_scale") or pixel_shuffle_scale or 1
            ),
            "temporal_patch_size": int(
                config.get("temporal_patch_size") or temporal_patch_size or 1
            ),
            "max_num_patches": config.get("max_num_patches", max_num_patches),
            "min_num_patches": config.get("min_num_patches", min_num_patches),
            "max_pixels": config.get("max_pixels", max_pixels),
            "min_pixels": config.get("min_pixels", min_pixels),
        }
        if isinstance(image, _BYTES_TYPES):
            return process_qwen35_image_bytes(
                image,
                max_decode_pixels=int(
                    config.get("max_decode_pixels", max_decode_pixels)
                ),
                **options,
            )
        return process_qwen35_image_array(image, **options)

    # The high-level facade merges these defaults with the caller's explicit
    # preprocess_config before native normalization. This keeps factory options
    # meaningful while preserving explicit call-site precedence.
    image_processor._mharmony_preprocess_defaults = {  # type: ignore[attr-defined]
        "patch_size": patch_size,
        "max_num_patches": max_num_patches,
        "min_num_patches": min_num_patches,
        "max_pixels": max_pixels,
        "min_pixels": min_pixels,
        "max_decode_pixels": max_decode_pixels,
        "pixel_shuffle_scale": pixel_shuffle_scale,
        "temporal_patch_size": temporal_patch_size,
    }
    return image_processor


# Keep the longer historical name as an identity alias for existing integrations.
create_qwen35_mharmony_image_processor = create_qwen35_image_processor
