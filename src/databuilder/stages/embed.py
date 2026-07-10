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


def _embed_worker(spec: _WorkerSpec) -> None:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    from ..store import EmbeddingShardWriter

    _init_image_plugins()
    use_cuda = spec.device.startswith("cuda")
    dtype = torch.float16 if use_cuda else torch.float32
    processor = AutoImageProcessor.from_pretrained(spec.model)
    model = AutoModel.from_pretrained(spec.model, torch_dtype=dtype).to(spec.device).eval()
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
