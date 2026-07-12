from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest

from databuilder.config import BalanceConfig, ClusteringConfig
from databuilder.stages import cluster, manifest
from databuilder.stages.cluster import choose_k, semdedup_prune, trigger_clusters
from databuilder.stages.manifest import assign_split, balance_select

from conftest import build_synthetic_artifacts

pytest.importorskip("sklearn")

SK2 = ClusteringConfig(backend="sklearn", k=2)


def test_choose_k():
    assert choose_k(10_000, k=64, aggressiveness=0.5) == 64
    assert choose_k(10_000, k=0, aggressiveness=0.5) == 100  # ~sqrt(n)
    assert choose_k(10_000, k=0, aggressiveness=1.0) == 400  # ~4*sqrt(n)
    assert choose_k(4, k=0, aggressiveness=0.5) <= 4


def test_cluster_separates_blobs(make_ctx):
    ctx = make_ctx(clustering=SK2)
    rows = build_synthetic_artifacts(ctx)
    cluster.run(ctx)

    table = pq.read_table(ctx.artifact_dir("clustering") / "cluster_assignments.parquet")
    assignments = {r["image_id"]: r for r in table.to_pylist()}
    assert len(assignments) == 40
    assert not any(r["pruned"] for r in assignments.values())

    # every image in blob 0 shares a cluster, distinct from blob 1
    blob_cluster = {}
    for row in rows:
        blob = row["path"].rsplit("_", 2)[-2]
        blob_cluster.setdefault(blob, set()).add(assignments[row["image_id"]]["cluster_id"])
    assert all(len(c) == 1 for c in blob_cluster.values())
    assert blob_cluster["0"] != blob_cluster["1"]

    summary = pq.read_table(ctx.artifact_dir("clustering") / "cluster_summary.parquet").to_pylist()
    assert sorted(r["size"] for r in summary) == [20, 20]


def test_trigger_clusters_only_extreme_outliers():
    assert trigger_clusters(np.array([10, 12, 11, 300]), sigma=1.0).tolist() == [3]
    # homogeneous cluster sizes never trigger, even at sigma 0
    assert trigger_clusters(np.array([20, 20]), sigma=0.0).size == 0
    # empty clusters are ignored; a single non-empty cluster is never an outlier
    assert trigger_clusters(np.array([0, 0, 5]), sigma=0.0).size == 0
    assert trigger_clusters(np.zeros(0, dtype=np.int64), sigma=0.0).size == 0


def test_semdedup_prune_removes_only_near_duplicates():
    rng = np.random.default_rng(3)
    base = np.zeros(8, dtype=np.float32)
    base[0] = 1.0
    dupes = base + rng.normal(0, 1e-4, (5, 8)).astype(np.float32)
    unique = np.eye(8, dtype=np.float32)[4:]  # mutually orthogonal members
    emb = np.concatenate([dupes, unique])
    dists = np.linspace(0.0, 1.0, len(emb)).astype(np.float32)
    ids = np.arange(len(emb), dtype=np.uint64)
    pruned = semdedup_prune(emb, dists, ids, threshold=0.96)
    assert pruned[:5].sum() == 4  # one representative of the dupe group survives
    assert not pruned[5:].any()  # unique members are untouched
    # deterministic
    assert np.array_equal(pruned, semdedup_prune(emb, dists, ids, threshold=0.96))


def test_balance_select_round_robins_clusters():
    ids = np.arange(1, 11, dtype=np.uint64)
    generators = np.zeros(10, dtype=np.int32)
    clusters = np.array([0] * 8 + [1] * 2, dtype=np.int64)
    chosen = balance_select(
        ids, generators, clusters, max_per_generator=5, per_generator_cluster_cap=0, seed=1
    )
    assert len(chosen) == 5
    chosen_clusters = clusters[[int(i) - 1 for i in chosen]]
    # both clusters contribute; the small one fully
    assert (chosen_clusters == 1).sum() == 2
    assert (chosen_clusters == 0).sum() == 3


def test_balance_select_trims_majority_label():
    # two real generators (12 rows) vs one fake generator (4 rows)
    ids = np.arange(1, 17, dtype=np.uint64)
    generators = np.array([0] * 6 + [1] * 6 + [2] * 4, dtype=np.int32)
    labels = np.array([0] * 12 + [1] * 4, dtype=np.int8)
    clusters = np.zeros(16, dtype=np.int64)

    chosen = balance_select(
        ids, generators, clusters, 0, 0, seed=1, labels=labels, max_label_ratio=1.0
    )
    chosen_idx = [int(i) - 1 for i in chosen]
    kept_real = [i for i in chosen_idx if labels[i] == 0]
    kept_fake = [i for i in chosen_idx if labels[i] == 1]
    assert len(kept_fake) == 4  # minority untouched
    assert len(kept_real) == 4  # majority trimmed to minority * 1.0
    # trimming is round-robin: both real generators stay represented
    assert {int(generators[i]) for i in kept_real} == {0, 1}
    # deterministic
    again = balance_select(
        ids, generators, clusters, 0, 0, seed=1, labels=labels, max_label_ratio=1.0
    )
    assert np.array_equal(chosen, again)


def test_balance_select_label_ratio_disabled_and_single_label():
    ids = np.arange(1, 17, dtype=np.uint64)
    generators = np.array([0] * 12 + [1] * 4, dtype=np.int32)
    clusters = np.zeros(16, dtype=np.int64)
    labels = np.array([0] * 12 + [1] * 4, dtype=np.int8)
    # ratio 0 disables trimming entirely
    chosen = balance_select(
        ids, generators, clusters, 0, 0, seed=1, labels=labels, max_label_ratio=0.0
    )
    assert len(chosen) == 16
    # single-label pool: ratio has nothing to balance against
    all_fake = np.ones(16, dtype=np.int8)
    chosen = balance_select(
        ids, generators, clusters, 0, 0, seed=1, labels=all_fake, max_label_ratio=1.0
    )
    assert len(chosen) == 16


def test_balance_select_label_ratio_two_to_one():
    ids = np.arange(1, 16, dtype=np.uint64)
    generators = np.array([0] * 12 + [1] * 3, dtype=np.int32)
    clusters = np.zeros(15, dtype=np.int64)
    labels = np.array([0] * 12 + [1] * 3, dtype=np.int8)
    chosen = balance_select(
        ids, generators, clusters, 0, 0, seed=1, labels=labels, max_label_ratio=2.0
    )
    chosen_idx = [int(i) - 1 for i in chosen]
    assert sum(1 for i in chosen_idx if labels[i] == 0) == 6  # 3 * 2.0
    assert sum(1 for i in chosen_idx if labels[i] == 1) == 3


def test_assign_split_deterministic_fractions():
    splits = [assign_split(i, seed=42, val_fraction=0.2, test_fraction=0.1) for i in range(5000)]
    assert splits == [
        assign_split(i, seed=42, val_fraction=0.2, test_fraction=0.1) for i in range(5000)
    ]
    frac_val = splits.count("val") / len(splits)
    frac_test = splits.count("test") / len(splits)
    assert 0.17 < frac_val < 0.23
    assert 0.08 < frac_test < 0.12


def test_manifest_balances_generators(make_ctx):
    ctx = make_ctx(
        clustering=SK2,
        balance=BalanceConfig(max_per_generator=5, val_fraction=0.2),
    )
    build_synthetic_artifacts(ctx)
    cluster.run(ctx)
    manifest.run(ctx)

    table = pq.read_table(ctx.artifact_dir("manifest") / "manifest.parquet").to_pylist()
    assert len(table) == 10
    per_generator = {}
    for row in table:
        per_generator[row["generator"]] = per_generator.get(row["generator"], 0) + 1
        assert row["label"] == 1
        assert row["split"] in {"train", "val"}
        assert row["cluster_id"] in {0, 1}
        assert len(row["file_hash"]) == 16  # zero-padded xxh3_64 hex
    assert per_generator == {"gen_a": 5, "gen_b": 5}
    assert (ctx.artifact_dir("manifest") / "manifest.csv").exists()


def test_manifest_refuses_csv_over_limit(make_ctx, monkeypatch):
    monkeypatch.setattr(manifest, "CSV_MAX_ROWS", 5)
    ctx = make_ctx(clustering=SK2)
    build_synthetic_artifacts(ctx)
    cluster.run(ctx)
    manifest.run(ctx)

    assert not (ctx.artifact_dir("manifest") / "manifest.csv").exists()
    assert pq.read_table(ctx.artifact_dir("manifest") / "manifest.parquet").num_rows == 40
