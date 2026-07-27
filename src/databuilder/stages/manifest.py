from __future__ import annotations

import csv
import logging
import random

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq

from ..config import CSV_MAX_ROWS
from ..state import RunContext
from ..utils import ParquetShardWriter, iter_parquet_batches
from ..wds import compact_dataset, is_webdataset, iter_index, load_marker
from .common import dataset_roots, normalize_label, pipeline_datasets, resolve_abs_from_roots

log = logging.getLogger("databuilder.manifest")

MANIFEST_SCHEMA = pa.schema(
    [
        ("path", pa.string()),
        ("label", pa.int8()),
        ("split", pa.string()),
        ("generator", pa.string()),
        ("source_dataset", pa.string()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("cluster_id", pa.int32()),
        ("image_id", pa.uint64()),
        ("file_hash", pa.string()),
        ("laplacian", pa.float64()),
        ("shard", pa.string()),
        ("member", pa.string()),
        ("offset", pa.int64()),
        ("size", pa.int64()),
    ]
)
_MASK = (1 << 64) - 1


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK
    return (x ^ (x >> 31)) & _MASK


def assign_split(image_id: int, seed: int, val_fraction: float, test_fraction: float) -> str:
    u = splitmix64(image_id ^ seed) / 2**64
    if u < test_fraction:
        return "test"
    if u < test_fraction + val_fraction:
        return "val"
    return "train"


def balance_select(
    ids: np.ndarray,
    generators: np.ndarray,
    clusters: np.ndarray,
    max_per_generator: int,
    per_generator_cluster_cap: int,
    seed: int,
    labels: np.ndarray | None = None,
    max_label_ratio: float = 0.0,
) -> np.ndarray:
    """Pick a generator-, cluster-, and label-balanced subset; returns sorted ids.

    Per generator: optional per-cluster cap, then round-robin draw across its
    clusters until the generator quota is reached. Generators under quota are
    kept in full. When `labels` and `max_label_ratio` > 0 are given, the
    majority label (0=real/1=fake) is then trimmed round-robin across its
    generators until majority <= minority * max_label_ratio; unknown labels
    (-1) are never trimmed. Deterministic for a given seed.
    """
    picked_by_gen: dict[int, list[int]] = {}
    for gen in np.unique(generators):
        idx = np.nonzero(generators == gen)[0]
        quota = len(idx) if max_per_generator <= 0 else min(len(idx), max_per_generator)
        rng = random.Random(f"{seed}:{int(gen)}")
        by_cluster: dict[int, list[int]] = {}
        for i in idx:
            by_cluster.setdefault(int(clusters[i]), []).append(int(i))
        queues = []
        for cluster_id in sorted(by_cluster):
            queue = by_cluster[cluster_id]
            rng.shuffle(queue)
            if per_generator_cluster_cap > 0:
                queue = queue[:per_generator_cluster_cap]
            queues.append(queue)
        picked: list[int] = []
        while len(picked) < quota and queues:
            queues = [q for q in queues if q]
            for queue in queues:
                if len(picked) >= quota:
                    break
                picked.append(queue.pop())
        if picked:
            picked_by_gen[int(gen)] = picked

    chosen = [i for gen in sorted(picked_by_gen) for i in picked_by_gen[gen]]
    if labels is not None and max_label_ratio > 0 and chosen:
        chosen = _trim_majority_label(picked_by_gen, labels, max_label_ratio)
    return np.sort(ids[np.array(chosen, dtype=np.int64)]) if chosen else np.array([], np.uint64)


def _trim_majority_label(
    picked_by_gen: dict[int, list[int]],
    labels: np.ndarray,
    max_label_ratio: float,
) -> list[int]:
    """Cut the over-represented label back to minority * ratio, round-robin per generator."""
    all_picked = [i for gen in sorted(picked_by_gen) for i in picked_by_gen[gen]]
    count_real = sum(1 for i in all_picked if labels[i] == 0)
    count_fake = sum(1 for i in all_picked if labels[i] == 1)
    if count_real == 0 or count_fake == 0:
        return all_picked  # single-label pool: nothing to balance against
    majority = 0 if count_real > count_fake else 1
    minority_count = min(count_real, count_fake)
    target = int(round(minority_count * max_label_ratio))
    majority_count = max(count_real, count_fake)
    if majority_count <= target:
        return all_picked
    # keep `target` majority rows, drawing round-robin across generators in
    # each generator's original pick-priority order
    queues = [
        [i for i in picked_by_gen[gen] if labels[i] == majority]
        for gen in sorted(picked_by_gen)
    ]
    queues = [list(reversed(q)) for q in queues if q]
    kept_majority: set[int] = set()
    while len(kept_majority) < target and queues:
        queues = [q for q in queues if q]
        for queue in queues:
            if len(kept_majority) >= target:
                break
            kept_majority.add(queue.pop())
    return [i for i in all_picked if labels[i] != majority or i in kept_majority]


def run(ctx: RunContext) -> None:
    if ctx.dry_run:
        log.info("dry-run: skipping manifest emit")
        return
    balance = ctx.cfg.balance
    assignments = pq.read_table(
        ctx.artifact_dir("clustering") / "cluster_assignments.parquet"
    )
    a_ids = assignments.column("image_id").to_numpy(zero_copy_only=False).astype(np.uint64)
    a_clusters = assignments.column("cluster_id").to_numpy(zero_copy_only=False)
    a_pruned = assignments.column("pruned").to_numpy(zero_copy_only=False)
    order = np.argsort(a_ids)
    a_ids, a_clusters, a_pruned = a_ids[order], a_clusters[order], a_pruned[order]

    def lookup(ids: np.ndarray, values: np.ndarray, default) -> np.ndarray:
        if len(a_ids) == 0:
            return np.full(len(ids), default)
        pos = np.clip(np.searchsorted(a_ids, ids), 0, len(a_ids) - 1)
        found = a_ids[pos] == ids
        return np.where(found, values[pos], default)

    survivors = ctx.artifact_dir("dedup") / "survivors.parquet"
    datasets = pipeline_datasets(ctx.cfg)
    forced = {
        ds.name: ds.assign_split
        for ds in datasets
        if ds.assign_split in {"val", "test"}
    }
    pinned_train = {ds.name for ds in datasets if ds.assign_split == "train"}
    in_place = {ds.name for ds in datasets if ds.in_place}
    roots = dataset_roots(ctx.cfg)

    # Pass 1: compact arrays of eligible (non-pruned, embedded) rows. Rows from
    # datasets forced to val/test bypass balancing and are always selected.
    ids_parts: list[np.ndarray] = []
    gen_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    forced_parts: list[np.ndarray] = []
    vocab: dict[str, int] = {}
    for batch in iter_parquet_batches(
        survivors, columns=["image_id", "generator", "dataset", "label"]
    ):
        ids = batch.column("image_id").to_numpy(zero_copy_only=False).astype(np.uint64)
        in_assignments = lookup(ids, np.ones(len(a_ids), dtype=bool), False)
        pruned = lookup(ids, a_pruned, True)
        keep = in_assignments & ~pruned.astype(bool)
        is_forced = np.array(
            [name in forced for name in batch.column("dataset").to_pylist()], dtype=bool
        )
        codes = np.array(
            [vocab.setdefault(g, len(vocab)) for g in batch.column("generator").to_pylist()],
            dtype=np.int32,
        )
        label_codes = np.array(
            [normalize_label(value) for value in batch.column("label").to_pylist()],
            dtype=np.int8,
        )
        forced_parts.append(ids[keep & is_forced])
        ids_parts.append(ids[keep & ~is_forced])
        gen_parts.append(codes[keep & ~is_forced])
        label_parts.append(label_codes[keep & ~is_forced])

    eligible_ids = np.concatenate(ids_parts) if ids_parts else np.array([], np.uint64)
    eligible_gens = np.concatenate(gen_parts) if gen_parts else np.array([], np.int32)
    eligible_labels = np.concatenate(label_parts) if label_parts else np.array([], np.int8)
    eligible_clusters = lookup(eligible_ids, a_clusters, -1).astype(np.int64)
    forced_ids = (
        np.sort(np.concatenate(forced_parts)) if forced_parts else np.array([], np.uint64)
    )

    chosen = balance_select(
        eligible_ids,
        eligible_gens,
        eligible_clusters,
        balance.max_per_generator,
        balance.per_generator_cluster_cap,
        balance.seed,
        labels=eligible_labels,
        max_label_ratio=balance.max_label_ratio,
    )
    log.info(
        "balanced selection: %d of %d eligible images (+%d forced val/test)",
        len(chosen),
        len(eligible_ids),
        len(forced_ids),
    )

    def selected(image_id: int) -> bool:
        for pool in (chosen, forced_ids):
            pos = np.searchsorted(pool, np.uint64(image_id))
            if pos < len(pool) and pool[pos] == image_id:
                return True
        return False

    # Pass 2: write manifest rows for chosen ids.
    out_dir = ctx.artifact_dir("manifest")
    parquet_path = out_dir / "manifest.parquet"
    csv_path = out_dir / "manifest.csv"
    rows_total = 0
    csv_rows: list[dict] = []
    with ParquetShardWriter(parquet_path, MANIFEST_SCHEMA) as writer:
        for batch in iter_parquet_batches(survivors):
            for row in batch.to_pylist():
                image_id = row["image_id"]
                if not selected(image_id):
                    continue
                dataset = row["dataset"]
                if dataset in forced:
                    split = forced[dataset]
                elif dataset in pinned_train:
                    split = "train"
                else:
                    split = assign_split(
                        image_id, balance.seed, balance.val_fraction, balance.test_fraction
                    )
                if dataset in in_place:
                    out_path = str(resolve_abs_from_roots(roots, row["path"]))
                else:
                    out_path = row["path"]
                cluster_id = int(lookup(np.array([image_id], np.uint64), a_clusters, -1)[0])
                record = {
                    "path": out_path,
                    "label": normalize_label(row["label"]),
                    "split": split,
                    "generator": row["generator"],
                    "source_dataset": dataset,
                    "width": row["width"],
                    "height": row["height"],
                    "cluster_id": cluster_id,
                    "image_id": image_id,
                    "file_hash": f"{row['file_hash']:016x}",
                    "laplacian": row["laplacian"],
                    "shard": row.get("shard") or "",
                    "member": row.get("member") or "",
                    "offset": int(row.get("offset") or 0),
                    "size": int(row.get("size") or row["filesize"]),
                }
                writer.append(record)
                rows_total += 1
                if balance.emit_csv and rows_total <= CSV_MAX_ROWS:
                    csv_rows.append(record)

    if ctx.cfg.storage.compact_after_manifest and not ctx.dry_run:
        locator_maps = _compact_and_rewrite(ctx, parquet_path, keep_maps=bool(csv_rows))
        for record in csv_rows:
            locator = locator_maps.get(record["source_dataset"], {}).get(record["path"])
            if locator:
                for key in ("shard", "member", "offset", "size"):
                    record[key] = locator[key]

    if balance.emit_csv:
        if rows_total > CSV_MAX_ROWS:
            log.error(
                "Refusing to write CSV: %d rows exceeds the %d row limit. "
                "Use the parquet manifest instead.",
                rows_total,
                CSV_MAX_ROWS,
            )
            csv_path.unlink(missing_ok=True)
        else:
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer_csv = csv.DictWriter(handle, fieldnames=[f.name for f in MANIFEST_SCHEMA])
                writer_csv.writeheader()
                writer_csv.writerows(csv_rows)
            log.info("wrote %s", csv_path)
    log.info("manifest done: %d rows -> %s", rows_total, parquet_path)


def _compact_and_rewrite(
    ctx: RunContext, manifest_path, *, keep_maps: bool
) -> dict[str, dict[str, dict]]:
    """Compact canonical shards, then refresh manifest member offsets."""
    source = pa_ds.dataset(str(manifest_path), format="parquet")
    locator_maps: dict[str, dict[str, dict]] = {}
    partial = manifest_path.with_name(manifest_path.name + ".partial")
    writer = pq.ParquetWriter(partial, MANIFEST_SCHEMA)
    try:
        for ds in pipeline_datasets(ctx.cfg):
            root = ctx.cfg.runtime.data_dir / ds.name
            selected: set[str] = set()
            selected_scanner = source.scanner(
                columns=["path"], filter=pa_ds.field("source_dataset") == ds.name,
                batch_size=8192,
            )
            for batch in selected_scanner.to_batches():
                selected.update(batch.column("path").to_pylist())
            locators: dict[str, dict] = {}
            storage_state = load_marker(root)
            if is_webdataset(root) or (
                storage_state.get("storage") == "webdataset"
                and storage_state.get("state") == "compacting"
            ):
                stats = compact_dataset(root, selected)
                log.info("compacted dataset %r: %s", ds.name, stats)
                locators = {
                    row["path"]: row
                    for row in iter_index(
                        root, columns=["path", "shard", "member", "offset", "size"]
                    )
                }
                if keep_maps:
                    locator_maps[ds.name] = locators
            rows_scanner = source.scanner(
                filter=pa_ds.field("source_dataset") == ds.name, batch_size=8192
            )
            for batch in rows_scanner.to_batches():
                rows = batch.to_pylist()
                for row in rows:
                    locator = locators.get(row["path"])
                    if locator:
                        for key in ("shard", "member", "offset", "size"):
                            row[key] = locator[key]
                writer.write_table(pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA))
            del locators, selected
    finally:
        writer.close()
    partial.replace(manifest_path)
    return locator_maps
