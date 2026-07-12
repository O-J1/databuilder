from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np

from ..state import RunContext
from ..utils import iter_parquet_batches, owns
from .common import dataset_roots, resolve_abs_from_roots
from .headerscan import _init_worker as _init_image_plugins

log = logging.getLogger("databuilder.embed")


@dataclass(frozen=True)
class _WorkerSpec:
    survivors_path: str
    roots: dict
    out_path: str
    model: str
    device: str
    batch_size: int
    image_size: int
    flush_rows: int
    rank: int
    world_size: int
    device_index: int
    device_count: int


def run(ctx: RunContext) -> None:
    """Compute DINOv3 embeddings for this rank's shard of dedup survivors."""
    if ctx.dry_run:
        log.info("dry-run: skipping embedding computation")
        return
    survivors = ctx.artifact_dir("dedup") / "survivors.parquet"
    if not survivors.exists():
        raise FileNotFoundError(f"{survivors} missing; run the dedup stage first")
    if ctx.cfg.daft.enabled:
        _run_daft(ctx, survivors)
        return
    out_dir = ctx.artifact_dir("embeddings")
    devices = _resolve_devices(ctx.cfg.embedding.devices)
    log.info("[rank %d] embedding on devices: %s", ctx.rank, devices)

    specs = [
        _WorkerSpec(
            survivors_path=str(survivors),
            roots=dataset_roots(ctx.cfg),
            out_path=str(out_dir / f"rank_{ctx.rank:05d}_dev{i}.parquet"),
            model=ctx.cfg.embedding.model,
            device=device,
            batch_size=ctx.cfg.embedding.batch_size,
            image_size=ctx.cfg.embedding.image_size,
            flush_rows=ctx.cfg.embedding.flush_rows,
            rank=ctx.rank,
            world_size=ctx.world_size,
            device_index=i,
            device_count=len(devices),
        )
        for i, device in enumerate(devices)
    ]
    if len(specs) == 1:
        _embed_worker(specs[0])
        return
    spawn = mp.get_context("spawn")
    procs = [spawn.Process(target=_embed_worker, args=(spec,)) for spec in specs]
    for proc in procs:
        proc.start()
    for proc, spec in zip(procs, specs):
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"embed worker for {spec.device} exited with {proc.exitcode}")


def _resolve_devices(spec: str) -> list[str]:
    if spec and spec not in {"auto", "cpu"}:
        return [d.strip() for d in spec.split(",") if d.strip()]
    if spec == "cpu":
        return ["cpu"]
    try:
        import torch

        count = torch.cuda.device_count()
    except ImportError as exc:
        raise RuntimeError(
            "The embed stage needs torch/transformers: pip install 'databuilder[embed]'"
        ) from exc
    return [f"cuda:{i}" for i in range(count)] if count else ["cpu"]


def _daft_use_gpu(cfg) -> bool:
    if cfg.embedding.devices == "cpu":
        return False
    if cfg.daft.runner == "ray":
        return True  # scheduling is per-replica; the cluster provides the GPUs
    try:
        import torch

        return torch.cuda.device_count() > 0
    except ImportError as exc:
        raise RuntimeError(
            "The embed stage needs torch/transformers: pip install 'databuilder[embed]'"
        ) from exc


def _daft_concurrency(cfg, use_gpu: bool) -> int:
    if cfg.embedding.concurrency:
        return cfg.embedding.concurrency
    if cfg.daft.runner == "ray":
        raise RuntimeError(
            "embedding.concurrency must be set explicitly for the daft ray runner "
            "(one model replica per GPU you want to occupy)"
        )
    if use_gpu:
        import torch

        return max(1, torch.cuda.device_count())
    return 1


def _make_embed_udf(daft, model: str, image_size: int, batch_size: int, use_gpu: bool):
    """Class UDF holding one model replica; daft schedules `concurrency` copies."""

    @daft.udf(
        return_dtype=daft.DataType.list(daft.DataType.float32()),
        batch_size=batch_size,
        num_gpus=1 if use_gpu else None,
    )
    class DinoEmbed:
        def __init__(self):
            import torch
            from transformers import AutoImageProcessor, AutoModel

            self.torch = torch
            use_cuda = use_gpu and torch.cuda.is_available()
            self.device = "cuda" if use_cuda else "cpu"
            dtype = torch.float16 if use_cuda else torch.float32
            self.processor = AutoImageProcessor.from_pretrained(model)
            self.model = (
                AutoModel.from_pretrained(model, dtype=dtype).to(self.device).eval()
            )
            self.size_kwargs = (
                {"size": {"height": image_size, "width": image_size}} if image_size else {}
            )

        def __call__(self, images):
            arrays = images.to_pylist()
            idx = [i for i, a in enumerate(arrays) if a is not None]
            out: list[list[float] | None] = [None] * len(arrays)
            if not idx:
                return out
            batch = [np.asarray(arrays[i]) for i in idx]
            inputs = self.processor(images=batch, return_tensors="pt", **self.size_kwargs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0]
            pooled = self.torch.nn.functional.normalize(pooled.float(), dim=-1)
            vectors = pooled.cpu().numpy()
            for j, i in enumerate(idx):
                out[i] = vectors[j].tolist()
            return out

    return DinoEmbed


def _run_daft(ctx: RunContext, survivors) -> None:
    """Daft execution path: decode + DINOv3 inference as a scheduled class UDF.

    Native runner: this rank embeds its owned shard using local GPUs. Ray
    runner: rank 0 submits all survivors and Ray spreads `concurrency` model
    replicas (1 GPU each) across the cluster. Results stream back to this
    process and land in the same parquet layout as the legacy path.
    """
    from ..store import EmbeddingShardWriter
    from .daft_exec import init_runner, make_owns_udf, with_downloaded_image

    daft = init_runner(ctx.cfg)
    from daft import col

    cfg = ctx.cfg
    out_path = ctx.artifact_dir("embeddings") / f"rank_{ctx.rank:05d}_daft.parquet"
    use_gpu = _daft_use_gpu(cfg)
    concurrency = _daft_concurrency(cfg, use_gpu)
    log.info(
        "[rank %d] embedding via daft/%s: %d replica(s), gpu=%s",
        ctx.rank, cfg.daft.runner, concurrency, use_gpu,
    )

    df = daft.read_parquet(str(survivors)).select("image_id", "path")
    if cfg.daft.runner != "ray" and ctx.world_size > 1:
        owns_row = make_owns_udf(daft, ctx.rank, ctx.world_size)
        df = df.where(owns_row(col("path")))
    df = with_downloaded_image(daft, df, dataset_roots(cfg))
    embed_udf = _make_embed_udf(
        daft, cfg.embedding.model, cfg.embedding.image_size, cfg.embedding.batch_size, use_gpu
    ).with_concurrency(concurrency)
    result = df.select("image_id", "path", embed_udf(col("image")).alias("embedding"))

    writer: EmbeddingShardWriter | None = None
    ids: list[int] = []
    paths: list[str] = []
    vectors: list[list[float]] = []
    skipped = 0

    def flush() -> None:
        nonlocal writer
        if not ids:
            return
        matrix = np.asarray(vectors, dtype=np.float16)
        if writer is None:
            writer = EmbeddingShardWriter(out_path, matrix.shape[1], cfg.embedding.flush_rows)
        writer.append_batch(np.array(ids, dtype=np.uint64), list(paths), matrix)
        ids.clear()
        paths.clear()
        vectors.clear()

    for row in result.iter_rows():
        if row["embedding"] is None:
            skipped += 1  # unreadable/undecodable at embed time
            continue
        ids.append(row["image_id"])
        paths.append(row["path"])
        vectors.append(row["embedding"])
        if len(ids) >= 1024:
            flush()
    flush()
    if writer is not None:
        writer.close()
        log.info(
            "[rank %d] daft embed: wrote %d embeddings (%d skipped)",
            ctx.rank, writer.rows_written, skipped,
        )
    else:
        log.warning("[rank %d] daft embed: nothing to write (%d skipped)", ctx.rank, skipped)


def _embed_worker(spec: _WorkerSpec) -> None:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    from ..store import EmbeddingShardWriter

    _init_image_plugins()
    use_cuda = spec.device.startswith("cuda")
    dtype = torch.float16 if use_cuda else torch.float32
    processor = AutoImageProcessor.from_pretrained(spec.model)
    model = AutoModel.from_pretrained(spec.model, dtype=dtype).to(spec.device).eval()
    size_kwargs = (
        {"size": {"height": spec.image_size, "width": spec.image_size}}
        if spec.image_size
        else {}
    )

    writer: EmbeddingShardWriter | None = None
    ids: list[int] = []
    paths: list[str] = []
    images: list[Image.Image] = []
    skipped = 0

    def flush() -> None:
        nonlocal writer, skipped
        if not images:
            return
        inputs = processor(images=images, return_tensors="pt", **size_kwargs)
        inputs = {k: v.to(spec.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
        vectors = pooled.to(torch.float16).cpu().numpy()
        if writer is None:
            writer = EmbeddingShardWriter(spec.out_path, vectors.shape[1], spec.flush_rows)
        writer.append_batch(np.array(ids, dtype=np.uint64), list(paths), vectors)
        ids.clear()
        paths.clear()
        images.clear()

    owned_index = 0
    for batch in iter_parquet_batches(spec.survivors_path, columns=["image_id", "path"]):
        for row in batch.to_pylist():
            if not owns(row["path"], spec.rank, spec.world_size):
                continue
            owned_index += 1
            if (owned_index - 1) % spec.device_count != spec.device_index:
                continue
            abs_path = resolve_abs_from_roots(spec.roots, row["path"])
            try:
                with Image.open(abs_path) as img:
                    images.append(img.convert("RGB"))
            except Exception:  # noqa: BLE001 - unreadable at embed time
                skipped += 1
                continue
            ids.append(row["image_id"])
            paths.append(row["path"])
            if len(images) >= spec.batch_size:
                flush()
    flush()
    if writer is not None:
        writer.close()
        logging.getLogger("databuilder.embed").info(
            "%s: wrote %d embeddings (%d skipped)", spec.device, writer.rows_written, skipped
        )
