from __future__ import annotations

import io
import json
import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..wds import ImageRef, image_media_type, resolve_logical_path

log = logging.getLogger("databuilder.viz")

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
THUMB_SIZE = 256
THUMB_CACHE_BYTES = 256 * 1024 * 1024
PAIRS_PAGE_MAX = 100
PAIRS_HINT = "no pairs table; run `databuilder viz-prepare` (pairs are built by default)"

SSH_INSTRUCTIONS = """
The viewer binds to 127.0.0.1 only: it is invisible to the network and opens
NO ports on this machine. To view it from your laptop, tunnel through your
existing SSH access (traffic rides inside the encrypted SSH session):

    ssh -L {port}:127.0.0.1:{port} <user>@<this-node>

via a bastion/login node:

    ssh -J <user>@<bastion> -L {port}:127.0.0.1:{port} <user>@<this-node>

then open  http://localhost:{port}  in your local browser.
"""


class _ThumbCache:
    def __init__(self, max_bytes: int = THUMB_CACHE_BYTES):
        self.max_bytes = max_bytes
        self.total = 0
        self._store: OrderedDict[int, bytes] = OrderedDict()

    def get(self, key: int) -> bytes | None:
        data = self._store.get(key)
        if data is not None:
            self._store.move_to_end(key)
        return data

    def put(self, key: int, data: bytes) -> None:
        if key in self._store:
            return
        self._store[key] = data
        self.total += len(data)
        while self.total > self.max_bytes and self._store:
            _, evicted = self._store.popitem(last=False)
            self.total -= len(evicted)


def load_viz_table(work_dir: Path):
    path = Path(work_dir) / "viz" / "viz.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `databuilder viz-prepare` first")
    return pq.read_table(path)


class _PairsIndex:
    """Memory-safe view over viz/pairs.parquet (may hold millions of rows).

    The table stays columnar in Arrow; only the requested page is ever
    converted to python objects. Image-id lookup uses sorted numpy arrays +
    searchsorted instead of a per-id dict (a dict would cost hundreds of MB
    at the million-row scale).
    """

    def __init__(self, path: Path):
        self.table = None
        if not path.exists():
            return
        table = pq.read_table(path)
        self.table = table
        self.is_dedup = pc.equal(table.column("kind"), "dedup").to_numpy(
            zero_copy_only=False
        )
        self.clusters = table.column("cluster_id").to_numpy(zero_copy_only=False)
        pruned_ids = (
            table.column("pruned_image_id").to_numpy(zero_copy_only=False).astype(np.uint64)
        )
        kept_ids = (
            table.column("kept_image_id").to_numpy(zero_copy_only=False).astype(np.uint64)
        )
        self._n = len(pruned_ids)
        all_ids = np.concatenate([pruned_ids, kept_ids])
        self._order = np.argsort(all_ids, kind="stable")
        self._sorted_ids = all_ids[self._order]

    def path_of(self, image_id: int) -> str | None:
        """Relative path for any pruned or kept id in the pairs table."""
        if self.table is None or len(self._sorted_ids) == 0:
            return None
        pos = int(np.searchsorted(self._sorted_ids, np.uint64(image_id)))
        if pos >= len(self._sorted_ids) or int(self._sorted_ids[pos]) != image_id:
            return None
        flat = int(self._order[pos])
        row, col = (flat, "pruned_path") if flat < self._n else (flat - self._n, "kept_path")
        return self.table.column(col)[row].as_py()

    def page(
        self, kind: str | None, cluster: int | None, page: int, page_size: int
    ) -> tuple[list[dict], int]:
        mask = np.ones(len(self.clusters), dtype=bool)
        if kind == "dedup":
            mask &= self.is_dedup
        elif kind == "cluster":
            mask &= ~self.is_dedup
        if cluster is not None:
            mask &= self.clusters == cluster
        idx = np.nonzero(mask)[0]
        chunk = idx[page * page_size : (page + 1) * page_size]
        rows = self.table.take(pa.array(chunk)).to_pylist() if len(chunk) else []
        return rows, int(len(idx))

    def summary(self) -> dict:
        dedup_count = int(self.is_dedup.sum())
        cluster_mask = ~self.is_dedup
        unique, counts = np.unique(self.clusters[cluster_mask], return_counts=True)
        return {
            "dedup": dedup_count,
            "cluster": int(cluster_mask.sum()),
            "clusters": [
                {"cluster_id": int(c), "pairs": int(n)} for c, n in zip(unique, counts)
            ],
        }


class _FlagStore:
    """Set of flagged image ids persisted to a JSON file (write-through)."""

    def __init__(self, path: Path):
        self.path = path
        self.ids: set[int] = set()
        if path.exists():
            try:
                self.ids = {int(i) for i in json.loads(path.read_text())}
            except (ValueError, OSError) as exc:
                log.warning("could not read %s (%s); starting with no flags", path, exc)

    def set(self, image_id: int, flagged: bool) -> None:
        if flagged:
            self.ids.add(image_id)
        else:
            self.ids.discard(image_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(sorted(self.ids)))
        tmp.replace(self.path)


def create_app(
    work_dir: Path | str,
    data_dir: Path | str | None = None,
    roots: dict[str, str] | None = None,
):
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import FileResponse, PlainTextResponse, Response
    from fastapi.staticfiles import StaticFiles

    work_dir = Path(work_dir)
    table = load_viz_table(work_dir)
    meta = table.schema.metadata or {}
    if roots is None:
        stored_roots = meta.get(b"databuilder.roots", b"").decode()
        roots = json.loads(stored_roots) if stored_roots else {}
    if data_dir is None:
        stored = meta.get(b"databuilder.data_dir", b"").decode()
        data_dir = Path(stored) if stored else None

    ids = table.column("image_id").to_numpy(zero_copy_only=False).astype(np.uint64)
    paths = table.column("path").to_pylist()
    xs = table.column("x").to_numpy(zero_copy_only=False)
    ys = table.column("y").to_numpy(zero_copy_only=False)
    clusters = table.column("cluster_id").to_numpy(zero_copy_only=False)
    pruned = table.column("pruned").to_numpy(zero_copy_only=False)
    generators = table.column("generator").to_pylist()
    datasets = table.column("dataset").to_pylist()
    labels = table.column("label").to_pylist()

    id_to_row = {int(i): idx for idx, i in enumerate(ids.tolist())}
    gen_vocab: dict[str, int] = {}
    gen_codes = [gen_vocab.setdefault(g, len(gen_vocab)) for g in generators]

    summary_path = work_dir / "artifacts" / "clustering" / "cluster_summary.parquet"
    if summary_path.exists():
        summary = pq.read_table(summary_path).to_pylist()
    else:
        unique, counts = np.unique(clusters, return_counts=True)
        summary = [
            {"cluster_id": int(c), "size": int(n), "pruned": 0}
            for c, n in zip(unique, counts)
        ]

    locator_cache: dict[str, dict] = {}

    def locator_of(rel: str) -> dict:
        name, _, rest = rel.partition("/")
        if rel in locator_cache:
            return locator_cache[rel]
        effective_roots = dict(roots)
        if name not in effective_roots and data_dir is not None:
            effective_roots[name] = str(Path(data_dir) / name)
        if name not in effective_roots:
            raise HTTPException(
                503, "dataset roots unknown; pass --data-dir to `databuilder viz`"
            )
        try:
            row = resolve_logical_path(effective_roots, rel)
        except (KeyError, FileNotFoundError, OSError) as exc:
            raise HTTPException(404, f"image missing: {exc}") from exc
        row["_roots"] = effective_roots
        locator_cache[rel] = row
        return row

    def display_rel(rel: str) -> str:
        row = locator_of(rel)
        return ImageRef.from_row(row).display(row["_roots"])

    pairs = _PairsIndex(work_dir / "viz" / "pairs.parquet")

    def rel_of(image_id: int) -> str | None:
        """Relative path for an id from the viz table or the pairs table."""
        row = id_to_row.get(image_id)
        if row is not None:
            return paths[row]
        return pairs.path_of(image_id)

    flags = _FlagStore(work_dir / "viz" / "flags.json")
    cache = _ThumbCache()
    app = FastAPI(title="databuilder cluster viewer")
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/points")
    def points() -> dict:
        return {
            "ids": [str(i) for i in ids.tolist()],
            "x": xs.tolist(),
            "y": ys.tolist(),
            "cluster": clusters.tolist(),
            "pruned": pruned.astype(int).tolist(),
            "generator": gen_codes,
            "generators": list(gen_vocab),
            "flagged": [1 if int(i) in flags.ids else 0 for i in ids.tolist()],
        }

    @app.get("/api/path/{image_id}")
    def image_path(image_id: int) -> dict:
        rel = rel_of(image_id)
        if rel is None:
            raise HTTPException(404, "unknown image id")
        return {"path": rel, "abs_path": display_rel(rel)}

    @app.post("/api/flag/{image_id}")
    def set_flag(image_id: int, flagged: bool = Body(embed=True)) -> dict:
        if rel_of(image_id) is None:
            raise HTTPException(404, "unknown image id")
        flags.set(image_id, flagged)
        return {"id": str(image_id), "flagged": flagged, "count": len(flags.ids)}

    @app.get("/api/flags")
    def list_flags() -> list[dict]:
        out = []
        for image_id in sorted(flags.ids):
            rel = rel_of(image_id)
            if rel is None:
                continue
            out.append(
                {
                    "id": str(image_id),
                    "path": rel,
                    "abs_path": display_rel(rel),
                }
            )
        return out

    @app.get("/api/flags.txt")
    def flags_txt() -> PlainTextResponse:
        lines = []
        for image_id in sorted(flags.ids):
            rel = rel_of(image_id)
            if rel is not None:
                lines.append(display_rel(rel))
        return PlainTextResponse(
            "\n".join(lines) + ("\n" if lines else ""),
            headers={"Content-Disposition": "attachment; filename=flags.txt"},
        )

    @app.get("/api/clusters")
    def cluster_list() -> list[dict]:
        return summary

    @app.get("/api/pairs")
    def pairs_page(
        kind: str | None = None,
        cluster: int | None = None,
        page: int = 0,
        page_size: int = 24,
    ) -> dict:
        if pairs.table is None:
            return {"total": 0, "page": 0, "page_size": page_size, "rows": [], "hint": PAIRS_HINT}
        page = max(0, page)
        page_size = max(1, min(page_size, PAIRS_PAGE_MAX))
        rows, total = pairs.page(kind, cluster, page, page_size)
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "rows": [
                {
                    "kind": row["kind"],
                    "pruned_id": str(row["pruned_image_id"]),
                    "pruned_path": row["pruned_path"],
                    "kept_id": str(row["kept_image_id"]),
                    "kept_path": row["kept_path"],
                    "cluster_id": row["cluster_id"],
                    "reason": row["reason"],
                    "dist": row["dist"],
                    "pruned_flagged": int(row["pruned_image_id"] in flags.ids),
                    "kept_flagged": int(row["kept_image_id"] in flags.ids),
                }
                for row in rows
            ],
        }

    @app.get("/api/pairs/summary")
    def pairs_summary() -> dict:
        if pairs.table is None:
            return {"dedup": 0, "cluster": 0, "clusters": [], "hint": PAIRS_HINT}
        return pairs.summary()

    @app.get("/api/cluster/{cluster_id}/examples")
    def examples(cluster_id: int, n: int = 24) -> list[dict]:
        rows = [i for i in range(len(ids)) if clusters[i] == cluster_id][:n]
        return [
            {
                "id": str(int(ids[i])),
                "path": paths[i],
                "generator": generators[i],
                "dataset": datasets[i],
                "label": labels[i],
                "pruned": bool(pruned[i]),
                "flagged": int(ids[i]) in flags.ids,
            }
            for i in rows
        ]

    @app.get("/thumb/{image_id}")
    def thumb(image_id: int) -> Response:
        from PIL import Image

        rel = rel_of(image_id)
        if rel is None:
            raise HTTPException(404, "unknown image id")
        cached = cache.get(image_id)
        if cached is None:
            try:
                row = locator_of(rel)
                data = ImageRef.from_row(row).read_bytes(row["_roots"])
                with Image.open(io.BytesIO(data)) as img:
                    img = img.convert("RGB")
                    img.thumbnail((THUMB_SIZE, THUMB_SIZE))
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=85)
            except (OSError, FileNotFoundError) as exc:
                raise HTTPException(404, f"cannot load image: {exc}") from exc
            cached = buffer.getvalue()
            cache.put(image_id, cached)
        return Response(cached, media_type="image/jpeg")

    @app.get("/image/{image_id}")
    def full_image(image_id: int) -> Response:
        rel = rel_of(image_id)
        if rel is None:
            raise HTTPException(404, "unknown image id")
        row = locator_of(rel)
        try:
            data = ImageRef.from_row(row).read_bytes(row["_roots"])
        except OSError as exc:
            raise HTTPException(404, f"image missing: {exc}") from exc
        return Response(data, media_type=image_media_type(row))

    return app


def serve(
    work_dir: Path | str,
    host: str = "127.0.0.1",
    port: int = 8765,
    allow_unsafe_remote: bool = False,
    data_dir: Path | str | None = None,
    roots: dict[str, str] | None = None,
) -> None:
    if host not in LOOPBACK_HOSTS and not allow_unsafe_remote:
        raise SystemExit(
            f"Refusing to bind to non-loopback host {host!r}. The secure way to "
            "access the viewer remotely is an SSH tunnel:\n"
            + SSH_INSTRUCTIONS.format(port=port)
            + "\nIf you really must bind externally, pass --allow-unsafe-remote."
        )
    import uvicorn

    app = create_app(work_dir, data_dir=data_dir, roots=roots)
    print(SSH_INSTRUCTIONS.format(port=port))
    uvicorn.run(app, host=host, port=port, log_level="info")
