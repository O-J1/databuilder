from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from databuilder.config import ClusteringConfig
from databuilder.stages import cluster
from databuilder.stages.dedup import REMOVED_SCHEMA
from databuilder.viz.prepare import prepare, prepare_pairs, stratified_quotas

from conftest import build_synthetic_artifacts, save_image

pytest.importorskip("sklearn")

SK2 = ClusteringConfig(backend="sklearn", k=2)
SK2_PRUNE = ClusteringConfig(backend="sklearn", k=2, prune_trigger_sigma=0.0, semdedup_threshold=0.9)


def write_removed(ctx, rows: list[dict]) -> None:
    columns = {name: [row[name] for row in rows] for name in REMOVED_SCHEMA.names}
    pq.write_table(
        pa.table(columns, schema=REMOVED_SCHEMA),
        ctx.artifact_dir("dedup") / "removed.parquet",
    )


def test_stratified_quotas_floor_and_proportional():
    sizes = {0: 1000, 1: 10, 2: 5}
    quotas = stratified_quotas(sizes, sample_size=100, min_per_cluster=20)
    assert quotas[1] == 10 and quotas[2] == 5  # small clusters fully kept
    assert quotas[0] >= 20
    assert sum(quotas.values()) <= 105  # rounding slack only


def test_stratified_quotas_under_budget_keeps_all():
    assert stratified_quotas({0: 5, 1: 3}, sample_size=100, min_per_cluster=2) == {0: 5, 1: 3}


def test_viz_prepare_stratified(make_ctx):
    ctx = make_ctx(clustering=SK2)
    build_synthetic_artifacts(ctx)
    cluster.run(ctx)
    out = prepare(
        ctx.work_dir, sample_size=10, min_per_cluster=3, seed=1, data_dir=ctx.data_dir
    )
    table = pq.read_table(out)
    rows = table.to_pylist()
    assert len(rows) == 10
    per_cluster = {}
    for row in rows:
        per_cluster[row["cluster_id"]] = per_cluster.get(row["cluster_id"], 0) + 1
        assert row["generator"] in {"gen_a", "gen_b"}
        assert isinstance(row["x"], float) and isinstance(row["y"], float)
    assert set(per_cluster) == {0, 1}
    assert all(count >= 3 for count in per_cluster.values())
    assert table.schema.metadata[b"databuilder.data_dir"].decode() == str(ctx.data_dir)


def test_viz_server_endpoints(make_ctx):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from databuilder.viz.server import create_app

    ctx = make_ctx(clustering=SK2)
    build_synthetic_artifacts(ctx)
    cluster.run(ctx)
    prepare(ctx.work_dir, sample_size=10, min_per_cluster=3, seed=1, data_dir=ctx.data_dir)

    client = TestClient(create_app(ctx.work_dir, data_dir=ctx.data_dir))

    points = client.get("/api/points").json()
    assert len(points["ids"]) == len(points["x"]) == len(points["cluster"]) == 10
    assert set(points["generators"]) == {"gen_a", "gen_b"}

    clusters = client.get("/api/clusters").json()
    assert sorted(c["cluster_id"] for c in clusters) == [0, 1]

    examples = client.get("/api/cluster/0/examples?n=4").json()
    assert 1 <= len(examples) <= 4

    assert client.get("/thumb/1234567").status_code == 404

    # materialize one sampled image so a real thumbnail can be served
    sample = examples[0]
    save_image(ctx.data_dir / sample["path"], size=(300, 300))
    response = client.get(f"/thumb/{sample['id']}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

    assert client.get("/image/1234567").status_code == 404
    full = client.get(f"/image/{sample['id']}")
    assert full.status_code == 200
    assert full.headers["content-type"].startswith("image/")


def test_viz_server_path_and_flags(make_ctx):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from databuilder.viz.server import create_app

    ctx = make_ctx(clustering=SK2)
    build_synthetic_artifacts(ctx)
    cluster.run(ctx)
    prepare(ctx.work_dir, sample_size=10, min_per_cluster=3, seed=1, data_dir=ctx.data_dir)

    client = TestClient(create_app(ctx.work_dir, data_dir=ctx.data_dir))
    points = client.get("/api/points").json()
    assert points["flagged"] == [0] * len(points["ids"])
    image_id = points["ids"][0]

    # path resolution
    assert client.get("/api/path/999999999").status_code == 404
    info = client.get(f"/api/path/{image_id}").json()
    assert info["abs_path"] == str(ctx.data_dir / info["path"])

    # flag roundtrip
    resp = client.post("/api/flag/999999999", json={"flagged": True})
    assert resp.status_code == 404, resp.text
    result = client.post(f"/api/flag/{image_id}", json={"flagged": True}).json()
    assert result == {"id": image_id, "flagged": True, "count": 1}
    flags = client.get("/api/flags").json()
    assert [f["id"] for f in flags] == [image_id]
    assert flags[0]["abs_path"] == info["abs_path"]
    assert client.get("/api/points").json()["flagged"].count(1) == 1

    txt = client.get("/api/flags.txt")
    assert txt.status_code == 200
    assert txt.text == info["abs_path"] + "\n"

    # flags persist across server restarts
    client2 = TestClient(create_app(ctx.work_dir, data_dir=ctx.data_dir))
    assert [f["id"] for f in client2.get("/api/flags").json()] == [image_id]

    # unflag
    result = client.post(f"/api/flag/{image_id}", json={"flagged": False}).json()
    assert result["count"] == 0
    assert client.get("/api/flags.txt").text == ""


def test_viz_server_refuses_remote_bind(tmp_path):
    from databuilder.viz.server import serve

    with pytest.raises(SystemExit, match="Refusing"):
        serve(tmp_path, host="0.0.0.0")


def test_prepare_pairs_cluster_counterparts(make_ctx):
    ctx = make_ctx(clustering=SK2_PRUNE)
    # unbalanced blobs: the big one is a size outlier full of near-duplicates
    build_synthetic_artifacts(ctx, n_per_cluster=(30, 4))
    cluster.run(ctx)
    assignments = pq.read_table(
        ctx.artifact_dir("clustering") / "cluster_assignments.parquet"
    ).to_pylist()
    pruned_ids = {row["image_id"] for row in assignments if row["pruned"]}
    kept_by_cluster = {}
    for row in assignments:
        if not row["pruned"]:
            kept_by_cluster.setdefault(row["cluster_id"], set()).add(row["image_id"])
    assert pruned_ids  # sigma=0 must trigger the outsized near-duplicate blob

    out = prepare_pairs(ctx.work_dir, kept_sample=8, seed=1)
    pairs = pq.read_table(out).to_pylist()
    assert {row["pruned_image_id"] for row in pairs} == pruned_ids
    for row in pairs:
        assert row["kind"] == "cluster"
        assert row["reason"] == "semantic_duplicate"
        assert row["kept_image_id"] in kept_by_cluster[row["cluster_id"]]
        assert row["pruned_path"] and row["kept_path"]
        assert row["dist"] >= 0.0


def test_prepare_pairs_dedup_and_old_schema(make_ctx):
    ctx = make_ctx(clustering=SK2)
    rows = build_synthetic_artifacts(ctx)
    cluster.run(ctx)  # no pruning: equal-size clusters never trigger
    write_removed(
        ctx,
        [
            {
                "image_id": 111,
                "path": "ds1/gen_a/dupe1.png",
                "dataset": "ds1",
                "reason": "duplicate_exact",
                "kept_image_id": rows[0]["image_id"],
            },
            {
                "image_id": 222,
                "path": "ds1/gen_b/dupe2.png",
                "dataset": "ds1",
                "reason": "duplicate_phash",
                "kept_image_id": rows[1]["image_id"],
            },
        ],
    )
    out = prepare_pairs(ctx.work_dir)
    pairs = sorted(pq.read_table(out).to_pylist(), key=lambda r: r["pruned_image_id"])
    assert [row["kind"] for row in pairs] == ["dedup", "dedup"]
    assert pairs[0]["kept_path"] == rows[0]["path"]
    assert pairs[1]["kept_path"] == rows[1]["path"]
    assert all(row["cluster_id"] == -1 for row in pairs)

    # legacy removed.parquet without kept_image_id degrades to zero dedup pairs
    legacy = pa.table(
        {
            "path": ["ds1/gen_a/dupe1.png"],
            "dataset": ["ds1"],
            "reason": ["duplicate_md5"],
        }
    )
    pq.write_table(legacy, ctx.artifact_dir("dedup") / "removed.parquet")
    out = prepare_pairs(ctx.work_dir)
    assert pq.read_table(out).num_rows == 0


def test_viz_server_pairs_endpoints(make_ctx):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from databuilder.viz.server import create_app

    ctx = make_ctx(clustering=SK2)
    rows = build_synthetic_artifacts(ctx)
    cluster.run(ctx)
    prepare(ctx.work_dir, sample_size=10, min_per_cluster=3, seed=1, data_dir=ctx.data_dir)

    # no pairs.parquet yet -> hint, no failure
    client = TestClient(create_app(ctx.work_dir, data_dir=ctx.data_dir))
    empty = client.get("/api/pairs").json()
    assert empty["total"] == 0 and "hint" in empty
    assert client.get("/api/pairs/summary").json()["dedup"] == 0

    write_removed(
        ctx,
        [
            {
                "image_id": 111,
                "path": "ds1/gen_a/dupe1.png",
                "dataset": "ds1",
                "reason": "duplicate_exact",
                "kept_image_id": rows[0]["image_id"],
            },
            {
                "image_id": 222,
                "path": "ds1/gen_b/dupe2.png",
                "dataset": "ds1",
                "reason": "duplicate_phash",
                "kept_image_id": rows[1]["image_id"],
            },
        ],
    )
    prepare_pairs(ctx.work_dir)
    client = TestClient(create_app(ctx.work_dir, data_dir=ctx.data_dir))

    summary = client.get("/api/pairs/summary").json()
    assert summary == {"dedup": 2, "cluster": 0, "clusters": []}

    page = client.get("/api/pairs?page_size=1").json()
    assert page["total"] == 2 and len(page["rows"]) == 1
    first = page["rows"][0]
    page2 = client.get("/api/pairs?page_size=1&page=1").json()
    assert len(page2["rows"]) == 1
    assert page2["rows"][0]["pruned_id"] != first["pruned_id"]
    assert client.get("/api/pairs?kind=cluster").json()["total"] == 0
    assert client.get("/api/pairs?page_size=1&page=99").json()["rows"] == []

    # pair-only ids resolve for path, flags and thumbs (viz sample excludes them)
    info = client.get("/api/path/111").json()
    assert info["path"] == "ds1/gen_a/dupe1.png"
    assert client.post("/api/flag/111", json={"flagged": True}).json()["count"] == 1
    assert client.get("/api/pairs?page_size=1").json()["rows"][0]["pruned_flagged"] in (0, 1)

    assert client.get("/thumb/111").status_code == 404  # file deleted from disk
    save_image(ctx.data_dir / "ds1" / "gen_a" / "dupe1.png", size=(64, 64))
    assert client.get("/thumb/111").status_code == 200
    assert client.get("/image/111").status_code == 200
