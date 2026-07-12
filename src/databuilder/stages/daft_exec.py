from __future__ import annotations

import io
import logging
from pathlib import Path

from ..config import Config
from ..utils import owns

log = logging.getLogger("databuilder.daft")

_RUNNER_SET = False

# Formats Daft's Rust decoder handles natively; everything else (HEIC/AVIF/JXL)
# goes through the PIL fallback UDF with the pillow plugins registered.
NATIVE_DECODABLE_RE = r"(?i)\.(jpe?g|png|webp|bmp|gif|tiff?)$"


def require_daft():
    try:
        import daft
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "[daft] enabled = true requires the daft package: "
            "pip install 'databuilder[daft]'"
        ) from exc
    return daft


def init_runner(cfg: Config):
    """Configure the Daft runner (idempotent; Daft allows one runner per process).

    "native" runs Daft locally on the current node, keeping the per-rank SLURM
    sharding; "ray" attaches to an existing Ray cluster and lets it schedule
    the whole stage from rank 0.
    """
    global _RUNNER_SET
    daft = require_daft()
    if _RUNNER_SET:
        return daft
    if cfg.daft.runner == "ray":
        address = cfg.daft.ray_address
        daft.set_runner_ray(address=None if address in ("", "auto") else address)
        log.info("daft runner: ray (address=%s)", address or "auto")
    else:
        daft.set_runner_native()
        log.info("daft runner: native")
    _RUNNER_SET = True
    return daft


def _decode_with_plugins(data: bytes):
    """PIL decode with the HEIF/AVIF/JXL plugins registered. Returns RGB HWC uint8."""
    import numpy as np
    from PIL import Image

    from .headerscan import _init_worker as _init_image_plugins

    _init_image_plugins()
    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(img.convert("RGB"))


def make_abs_path_udf(daft, roots: dict[str, str]):
    """Map canonical '<dataset>/<subpath>' keys onto absolute file paths."""

    @daft.func(return_dtype=daft.DataType.string())
    def to_abs(rel: str) -> str:
        name, _, rest = rel.partition("/")
        return str(Path(roots[name]) / rest)

    return to_abs


def make_pil_decode_udf(daft):
    """Fallback decoder for formats Daft's Rust decoder rejects (HEIC/AVIF/JXL)."""

    @daft.func(return_dtype=daft.DataType.image("RGB"))
    def pil_decode(data):
        if data is None:
            return None
        try:
            return _decode_with_plugins(data)
        except Exception:  # noqa: BLE001 - any failure means undecodable
            return None

    return pil_decode


def make_laplacian_udf(daft, max_side: int):
    """Laplacian variance on a thumbnail, matching the legacy fingerprint pass."""

    @daft.func(return_dtype=daft.DataType.float64())
    def laplacian_var(image):
        if image is None:
            return None
        import cv2
        import numpy as np
        from PIL import Image

        img = Image.fromarray(np.asarray(image))
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return laplacian_var


def make_owns_udf(daft, rank: int, world_size: int):
    """Static shard-ownership filter (same hash as utils.owns)."""

    @daft.func(return_dtype=daft.DataType.bool())
    def owns_row(path: str) -> bool:
        return owns(path, rank, world_size)

    return owns_row


def with_downloaded_image(daft, df, roots: dict[str, str]):
    """Add 'data' (raw bytes) and 'image' (RGB) columns to a dataframe with 'path'.

    Natively-supported formats decode in Rust; the rest (HEIC/AVIF/JXL) route
    through the PIL fallback UDF so nothing decodes twice. Unreadable or
    undecodable rows get a null 'image'.
    """
    from daft import col
    from daft.functions import decode_image, download, regexp

    to_abs = make_abs_path_udf(daft, roots)
    pil_decode = make_pil_decode_udf(daft)
    df = df.with_column("data", download(to_abs(col("path")), on_error="null"))
    is_native = regexp(col("path"), NATIVE_DECODABLE_RE)
    native = df.where(is_native).with_column(
        "image", decode_image(col("data"), on_error="null", mode="RGB")
    )
    fallback = df.where(~is_native).with_column("image", pil_decode(col("data")))
    return native.concat(fallback)
