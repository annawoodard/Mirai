"""
zarr_image_loader.py

Utility to open mammography images from either a PNG16 path (original Mirai behavior)
or a Zarr URI of the form: 'zarr:///abs/path/to/exam.zarr#L_CC'.

Returns a PIL.Image in 16-bit mode ('I;16'), so downstream Mirai transforms
and normalization flags (--img_mean/--img_std/--img_size) work unchanged.
"""

from __future__ import annotations

from typing import Tuple

from PIL import Image


def open_image_mono16_any(file_path: str) -> Image.Image:
    """
    open an image path that may be a regular PNG16 file or a zarr uri.

    Parameters
    ----------
    file_path : str
        Either an absolute path to a PNG16 or a 'zarr://<abs>.zarr#<KEY>' URI
        where KEY is one of L_CC, L_MLO, R_CC, R_MLO.

    Returns
    -------
    PIL.Image.Image
        16-bit grayscale image ('I;16')
    """
    if isinstance(file_path, str) and file_path.startswith("zarr://"):
        # lazy import to avoid hard dependency when unused
        import zarr  # type: ignore

        uri, key = _parse_zarr_uri(file_path)
        arr16 = zarr.open(uri, mode="r")[key][:]
        # ensure uint16 for 'I;16'
        if arr16.dtype != "uint16":
            arr16 = arr16.astype("uint16", copy=False)
        return Image.fromarray(arr16, mode="I;16")
    # fallback to original loader (PNG16 on disk)
    return Image.open(file_path)


def _parse_zarr_uri(uri: str) -> Tuple[str, str]:
    """
    split 'zarr:///abs/path.zarr#KEY' into ('/abs/path.zarr', 'KEY')
    """
    body = uri[len("zarr://") :]
    if "#" not in body:
        raise ValueError(f"malformed zarr uri (missing #): {uri}")
    store, key = body.split("#", 1)
    if not store or not key:
        raise ValueError(f"malformed zarr uri: {uri}")
    return store, key
