from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from databuilder.config import (  # noqa: E402
    BalanceConfig,
    ClusteringConfig,
    Config,
    DatasetConfig,
    DedupConfig,
    FiltersConfig,
    RuntimeConfig,
)
from databuilder.state import RunContext  # noqa: E402


@pytest.fixture
def make_ctx(tmp_path):
    def _make(
        datasets: tuple[DatasetConfig, ...] = (),
        filters: FiltersConfig | None = None,
        clustering: ClusteringConfig | None = None,
        balance: BalanceConfig | None = None,
        dedup: DedupConfig | None = None,
        dry_run: bool = False,
        world_size: int = 1,
        rank: int = 0,
    ) -> RunContext:
        kwargs = {}
        if filters is not None:
            kwargs["filters"] = filters
        if clustering is not None:
            kwargs["clustering"] = clustering
        if balance is not None:
            kwargs["balance"] = balance
        if dedup is not None:
            kwargs["dedup"] = dedup
        cfg = Config(
            runtime=RuntimeConfig(
                work_dir=tmp_path / "work",
                data_dir=tmp_path / "data",
                num_workers=1,
                world_size=world_size,
                rank=rank,
            ),
            datasets=tuple(datasets),
            **kwargs,
        )
        return RunContext(cfg=cfg, dry_run=dry_run)

    return _make


def save_image(path: Path, size=(400, 400), kind="gradient", seed=0) -> Path:
    """Write a synthetic PNG. kinds: gradient, flat, noise, circle."""
    import numpy as np
    from PIL import Image

    width, height = size
    if kind == "flat":
        arr = np.full((height, width, 3), 128, dtype=np.uint8)
    elif kind == "noise":
        rng = np.random.default_rng(seed)
        arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    elif kind == "circle":
        yy, xx = np.mgrid[0:height, 0:width]
        cx, cy = width / 2, height / 2
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < (min(width, height) / 3) ** 2
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        arr[mask] = (250, 40, 40)
    else:  # gradient
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        arr[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return path


def build_synthetic_artifacts(ctx, n_per_cluster=20, dim=8, generators=("gen_a", "gen_b"), seed=0):
    """Create survivors.parquet + one embedding shard with two separable blobs.

    n_per_cluster: int for equal blobs, or a (blob0_size, blob1_size) tuple.
    """
    import numpy as np

    from databuilder.stages.fingerprint import FINGERPRINT_SCHEMA
    from databuilder.store import EmbeddingShardWriter
    from databuilder.utils import ParquetShardWriter, image_id

    if isinstance(n_per_cluster, int):
        blob_sizes = (n_per_cluster, n_per_cluster)
    else:
        blob_sizes = tuple(n_per_cluster)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    ids: list[int] = []
    paths: list[str] = []
    vectors: list = []
    index = 0
    for blob, blob_size in enumerate(blob_sizes):
        center = np.zeros(dim, dtype=np.float32)
        center[blob] = 1.0
        for i in range(blob_size):
            generator = generators[index % len(generators)]
            path = f"ds1/{generator}/img_{blob}_{i}.png"
            iid = image_id(path)
            rows.append(
                {
                    "image_id": iid,
                    "path": path,
                    "dataset": "ds1",
                    "label": "fake",
                    "generator": generator,
                    "width": 400,
                    "height": 400,
                    "filesize": 1000 + index,
                    "md5": index.to_bytes(16, "big"),
                    "phash": bytes(18),
                    "colorhash": bytes(6),
                    "laplacian": 10.0,
                }
            )
            ids.append(iid)
            paths.append(path)
            vectors.append(center + rng.normal(0, 0.01, dim))
            index += 1
    with ParquetShardWriter(
        ctx.artifact_dir("dedup") / "survivors.parquet", FINGERPRINT_SCHEMA
    ) as writer:
        for row in rows:
            writer.append(row)
    with EmbeddingShardWriter(
        ctx.artifact_dir("embeddings") / "rank_00000_dev0.parquet", dim
    ) as writer:
        writer.append_batch(
            np.array(ids, dtype=np.uint64), paths, np.array(vectors, dtype=np.float16)
        )
    return rows
