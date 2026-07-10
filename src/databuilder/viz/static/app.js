"use strict";

const canvas = document.getElementById("scatter");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const tooltipImg = document.getElementById("tooltip-img");
const tooltipText = document.getElementById("tooltip-text");
const toast = document.getElementById("toast");

let pts = null;            // columnar point data from /api/points
let view = { x: 0, y: 0, scale: 1 };  // world->screen: s = (w - offset) * scale
let selectedCluster = null;
let clusterOrder = [];     // cluster ids in sidebar order (size desc)
let colorMode = "cluster"; // "cluster" | "generator"
let hidePruned = false;
let grid = new Map();      // spatial hash for hover lookup
const GRID_CELL = 16;      // px

const THUMB_MODE_MAX = 30;   // switch to thumbnails at/below this many DISTINCT visible dots
const THUMB_PX = 128;         // thumbnail size on screen (CSS px)
let thumbMode = false;
let thumbRects = [];         // [{i, x, y, w, h}] in CSS px, for hit-testing
const thumbCache = new Map(); // id -> {img, loaded}
let baseScale = 0;           // full-fit scale; thumb mode only engages when zoomed IN past this
let hullPts = [];            // world-coord hull of the selected cluster
const modal = document.getElementById("modal");
const modalImg = document.getElementById("modal-img");

const scatterView = document.getElementById("scatter-view");
const pairsView = document.getElementById("pairs-view");
let activeTab = "scatter";   // "scatter" | "pairs"
const pairsState = { kind: "", cluster: "", page: 0, total: 0, pageSize: 24, loaded: false };

function hueColor(id, faded) {
  if (id < 0) return faded ? "#55555514" : "#999999";
  const hue = (id * 137.508) % 360;
  return `hsla(${hue}, 70%, 60%, ${faded ? 0.05 : 0.8})`;
}

function pointColor(i, faded) {
  const id = colorMode === "generator" ? pts.generator[i] : pts.cluster[i];
  return hueColor(id, faded);
}

function clusterColor(id, faded) { return hueColor(id, faded); }

function resize() {
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;
  draw();
}

function fitBounds(minX, maxX, minY, maxY, pad = 0.9) {
  const w = canvas.width, h = canvas.height;
  const spanX = maxX - minX || 1, spanY = maxY - minY || 1;
  view.scale = pad * Math.min(w / spanX, h / spanY);
  view.x = minX - (w / view.scale - spanX) / 2;
  view.y = minY - (h / view.scale - spanY) / 2;
}

function fitView() {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < pts.x.length; i++) {
    minX = Math.min(minX, pts.x[i]); maxX = Math.max(maxX, pts.x[i]);
    minY = Math.min(minY, pts.y[i]); maxY = Math.max(maxY, pts.y[i]);
  }
  fitBounds(minX, maxX, minY, maxY);
  baseScale = view.scale;
}

function fitCluster(clusterId) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, found = false;
  for (let i = 0; i < pts.x.length; i++) {
    if (pts.cluster[i] !== clusterId) continue;
    found = true;
    minX = Math.min(minX, pts.x[i]); maxX = Math.max(maxX, pts.x[i]);
    minY = Math.min(minY, pts.y[i]); maxY = Math.max(maxY, pts.y[i]);
  }
  if (found) fitBounds(minX, maxX, minY, maxY, 0.8);
}

function worldToScreen(wx, wy) {
  return [(wx - view.x) * view.scale, (wy - view.y) * view.scale];
}

function getThumb(id) {
  let entry = thumbCache.get(id);
  if (!entry) {
    const img = new Image();
    entry = { img, loaded: false };
    img.onload = () => { entry.loaded = true; requestAnimationFrame(draw); };
    img.src = "/thumb/" + id;
    thumbCache.set(id, entry);
  }
  return entry;
}

function drawThumbs(groups) {
  thumbRects = [];
  const dpr = devicePixelRatio;
  const size = THUMB_PX * dpr;
  for (const group of groups) {
    const i = group[0];
    const [sx, sy] = worldToScreen(pts.x[i], pts.y[i]);
    const entry = getThumb(pts.ids[i]);
    const x = sx - size / 2, y = sy - size / 2;
    ctx.save();
    if (pts.pruned[i]) ctx.globalAlpha = 0.45;
    let bx = x / dpr, by = y / dpr;
    if (entry.loaded) {
      const iw = entry.img.naturalWidth, ih = entry.img.naturalHeight;
      const s = Math.min(size / iw, size / ih);
      const dw = iw * s, dh = ih * s;
      const dx = sx - dw / 2, dy = sy - dh / 2;
      ctx.drawImage(entry.img, dx, dy, dw, dh);
      ctx.lineWidth = (pts.flagged[i] ? 3 : 2) * dpr;
      ctx.strokeStyle = pts.flagged[i] ? "#e14b4b" : pointColor(i, false);
      ctx.strokeRect(dx, dy, dw, dh);
      thumbRects.push({ i, x: dx / dpr, y: dy / dpr, w: dw / dpr, h: dh / dpr });
      bx = dx / dpr; by = dy / dpr;
    } else {
      ctx.fillStyle = pointColor(i, false);
      ctx.fillRect(sx - 4 * dpr, sy - 4 * dpr, 8 * dpr, 8 * dpr);
      thumbRects.push({ i, x: bx, y: by, w: THUMB_PX, h: THUMB_PX });
    }
    if (group.length > 1) {
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#14161add";
      ctx.fillRect(bx * dpr, by * dpr, 34 * dpr, 16 * dpr);
      ctx.fillStyle = "#d8dbe0";
      ctx.font = `${11 * dpr}px system-ui`;
      ctx.fillText(`\u00d7${group.length}`, (bx + 4) * dpr, (by + 12) * dpr);
    }
    ctx.restore();
  }
}

// Group viewport-visible points that overlap on screen (near-identical UMAP
// coords never separate no matter the zoom), so thumb mode keys off the
// number of DISTINCT dots the user can actually see.
function groupVisible(visible) {
  const cell = THUMB_PX * devicePixelRatio;
  const groups = new Map();
  for (const i of visible) {
    const [sx, sy] = worldToScreen(pts.x[i], pts.y[i]);
    const key = Math.round(sx / cell) + "," + Math.round(sy / cell);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(i);
  }
  return [...groups.values()];
}

// Convex hull (monotone chain) of the selected cluster, in world coords.
function convexHull(clusterId) {
  const p = [];
  for (let i = 0; i < pts.x.length; i++) {
    if (pts.cluster[i] === clusterId) p.push([pts.x[i], pts.y[i]]);
  }
  if (p.length < 3) return p;
  p.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (iter) => {
    const out = [];
    for (const q of iter) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], q) <= 0) out.pop();
      out.push(q);
    }
    out.pop();
    return out;
  };
  return half(p).concat(half([...p].reverse()));
}

function drawHull() {
  if (selectedCluster === null || hullPts.length < 3) return;
  ctx.beginPath();
  hullPts.forEach(([wx, wy], k) => {
    const [sx, sy] = worldToScreen(wx, wy);
    k ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy);
  });
  ctx.closePath();
  if (selectedCluster < 0) {
    ctx.fillStyle = "rgba(150, 150, 150, 0.45)";
    ctx.strokeStyle = "#555";
  } else {
    const hue = (selectedCluster * 137.508) % 360;
    ctx.fillStyle = `hsla(${hue}, 70%, 65%, 0.45)`;
    ctx.strokeStyle = `hsl(${hue}, 70%, 30%)`;
  }
  ctx.lineWidth = 2 * devicePixelRatio;
  ctx.fill();
  ctx.stroke();
}

function draw() {
  if (!pts) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  grid = new Map();
  drawHull();
  const n = pts.x.length;
  const r = Math.max(1.2, Math.min(3, 900 / Math.sqrt(n))) * devicePixelRatio;
  const pad = THUMB_PX * devicePixelRatio;  // generous cull margin: keeps edge thumbs from flickering
  const visible = [];
  for (let i = 0; i < n; i++) {
    if (hidePruned && pts.pruned[i] === 1) continue;
    const [sx, sy] = worldToScreen(pts.x[i], pts.y[i]);
    if (sx < -pad || sy < -pad || sx > canvas.width + pad || sy > canvas.height + pad) continue;
    visible.push(i);
  }
  const zoomedIn = view.scale > baseScale * 1.01;  // never in thumb mode at/below the default fit
  const groups = zoomedIn && visible.length ? groupVisible(visible) : [];
  thumbMode = groups.length > 0 && groups.length <= THUMB_MODE_MAX;
  if (thumbMode) { drawThumbs(groups); return; }
  thumbRects = [];
  for (const i of visible) {
    const [sx, sy] = worldToScreen(pts.x[i], pts.y[i]);
    const faded = pts.pruned[i] === 1 ||
      (selectedCluster !== null && pts.cluster[i] !== selectedCluster);
    ctx.fillStyle = pointColor(i, faded);
    ctx.fillRect(sx - r / 2, sy - r / 2, r, r);
    if (pts.flagged[i]) {
      ctx.strokeStyle = "#e14b4b";
      ctx.lineWidth = devicePixelRatio;
      ctx.strokeRect(sx - r, sy - r, r * 2, r * 2);
    }
    const key = ((sx / devicePixelRatio / GRID_CELL) | 0) + "," + ((sy / devicePixelRatio / GRID_CELL) | 0);
    if (!grid.has(key)) grid.set(key, []);
    grid.get(key).push(i);
  }
}

function nearestPoint(mx, my) {
  // mx,my in CSS px
  if (thumbMode) {
    for (let k = thumbRects.length - 1; k >= 0; k--) {
      const t = thumbRects[k];
      if (mx >= t.x && mx <= t.x + t.w && my >= t.y && my <= t.y + t.h) return t.i;
    }
    return -1;
  }
  let best = -1, bestD = 12 * 12;
  const cx = (mx / GRID_CELL) | 0, cy = (my / GRID_CELL) | 0;
  for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
    const cell = grid.get((cx + dx) + "," + (cy + dy));
    if (!cell) continue;
    for (const i of cell) {
      const [sx, sy] = worldToScreen(pts.x[i], pts.y[i]);
      const d = (sx / devicePixelRatio - mx) ** 2 + (sy / devicePixelRatio - my) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    }
  }
  return best;
}

let dragging = false, lastMouse = null, hoverTimer = null;
let downPos = null, dragDist = 0;
let toastTimer = null;

function showToast(text) {
  toast.textContent = text;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 2200);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
}

async function copyPath(id) {
  const res = await fetch("/api/path/" + id);
  if (!res.ok) { showToast(`path lookup failed (${res.status})`); return; }
  const data = await res.json();
  await copyText(data.abs_path);
  showToast("copied: " + data.abs_path);
}

async function toggleFlag(i) {
  const id = pts.ids[i];
  const next = pts.flagged[i] ? 0 : 1;
  const ok = await setFlagById(id, !!next);
  if (!ok) return;
  pts.flagged[i] = next;
  updateFlagUI();
  requestAnimationFrame(draw);
}

// Flag any image id (scatter points AND pair images that may not be sampled).
async function setFlagById(id, flagged) {
  const res = await fetch("/api/flag/" + id, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ flagged }),
  });
  if (!res.ok) { showToast(`flag failed (${res.status})`); return false; }
  const i = pts ? pts.ids.indexOf(id) : -1;  // keep scatter state in sync
  if (i >= 0) {
    pts.flagged[i] = flagged ? 1 : 0;
    updateFlagUI();
    requestAnimationFrame(draw);
  }
  showToast(flagged ? "flagged as bad data" : "flag removed");
  return true;
}

canvas.addEventListener("mousedown", (e) => {
  dragging = true; dragDist = 0;
  lastMouse = [e.clientX, e.clientY]; downPos = [e.clientX, e.clientY];
  canvas.style.cursor = "grabbing";
});
window.addEventListener("mouseup", () => { dragging = false; canvas.style.cursor = "grab"; });

function openModal(id) {
  modalImg.src = "/image/" + id;
  modal.hidden = false;
}

function closeModal() {
  modal.hidden = true;
  modalImg.src = "";
}

modal.addEventListener("click", closeModal);

canvas.addEventListener("click", (e) => {
  if (dragDist > 4) return;  // was a drag, not a click
  const rect = canvas.getBoundingClientRect();
  const i = nearestPoint(e.clientX - rect.left, e.clientY - rect.top);
  if (i < 0) return;
  if (e.ctrlKey) copyPath(pts.ids[i]);
  else if (e.altKey) toggleFlag(i);
  else if (thumbMode) openModal(pts.ids[i]);
});

canvas.addEventListener("mousemove", (e) => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  if (dragging) {
    dragDist += Math.abs(e.clientX - lastMouse[0]) + Math.abs(e.clientY - lastMouse[1]);
    view.x -= (e.clientX - lastMouse[0]) * devicePixelRatio / view.scale;
    view.y -= (e.clientY - lastMouse[1]) * devicePixelRatio / view.scale;
    lastMouse = [e.clientX, e.clientY];
    tooltip.hidden = true;
    requestAnimationFrame(draw);
    return;
  }
  clearTimeout(hoverTimer);
  hoverTimer = setTimeout(() => {
    const i = nearestPoint(mx, my);
    if (i < 0) { tooltip.hidden = true; return; }
    if (thumbMode) {
      tooltipImg.src = "";
      tooltipImg.hidden = true;
    } else {
      tooltipImg.hidden = false;
      tooltipImg.src = "/thumb/" + pts.ids[i];
    }
    tooltipText.textContent =
      `cluster ${pts.cluster[i]}${pts.pruned[i] ? " (pruned)" : ""}${pts.flagged[i] ? " [FLAGGED]" : ""}\n` +
      `generator: ${pts.generators[pts.generator[i]] || "?"}`;
    tooltip.style.left = Math.min(mx + 14, rect.width - 290) + "px";
    tooltip.style.top = Math.min(my + 14, rect.height - 300) + "px";
    tooltip.hidden = false;
  }, 60);
});

canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * devicePixelRatio, my = (e.clientY - rect.top) * devicePixelRatio;
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  const wx = view.x + mx / view.scale, wy = view.y + my / view.scale;
  view.scale *= factor;
  view.x = wx - mx / view.scale;
  view.y = wy - my / view.scale;
  requestAnimationFrame(draw);
}, { passive: false });

function flagCounts() {
  const counts = new Map();
  for (let i = 0; i < pts.flagged.length; i++) {
    if (pts.flagged[i]) counts.set(pts.cluster[i], (counts.get(pts.cluster[i]) || 0) + 1);
  }
  return counts;
}

function updateFlagUI() {
  const total = pts.flagged.reduce((a, b) => a + b, 0);
  document.getElementById("flag-count").textContent = total;
  const counts = flagCounts();
  document.querySelectorAll("#cluster-list li").forEach((li) => {
    const c = Number(li.dataset.cluster);
    const badge = li.querySelector(".flag-badge");
    const n = counts.get(c) || 0;
    badge.textContent = n ? `\u2691${n}` : "";
  });
}

async function loadClusters() {
  const clusters = await (await fetch("/api/clusters")).json();
  clusters.sort((a, b) => b.size - a.size);
  clusterOrder = clusters.map((c) => c.cluster_id);
  const maxSize = clusters.length ? clusters[0].size : 1;
  const list = document.getElementById("cluster-list");
  list.innerHTML = "";
  for (const c of clusters) {
    const li = document.createElement("li");
    li.dataset.cluster = c.cluster_id;
    const keptW = Math.max(2, 120 * (c.size - c.pruned) / maxSize);
    const prunedW = Math.max(c.pruned > 0 ? 2 : 0, 120 * c.pruned / maxSize);
    li.innerHTML =
      `<span class="swatch" style="background:${clusterColor(c.cluster_id, false)}"></span>` +
      `<span>#${c.cluster_id}</span>` +
      `<span class="bar" style="width:${keptW}px"></span>` +
      `<span class="pruned-bar" style="width:${prunedW}px"></span>` +
      `<span>${c.size}</span>` +
      `<span class="flag-badge"></span>`;
    li.onclick = () => selectCluster(c.cluster_id, li);
    list.appendChild(li);
  }
  updateFlagUI();
}

async function selectCluster(clusterId, li) {
  const wasSelected = selectedCluster === clusterId;
  selectedCluster = wasSelected ? null : clusterId;
  hullPts = wasSelected ? [] : convexHull(clusterId);
  document.querySelectorAll("#cluster-list li").forEach((el) => el.classList.remove("selected"));
  const examplesDiv = document.getElementById("examples");
  const title = document.getElementById("examples-title");
  if (wasSelected) { examplesDiv.innerHTML = ""; title.hidden = true; fitView(); draw(); return; }
  li.classList.add("selected");
  li.scrollIntoView({ block: "nearest" });
  fitCluster(clusterId);
  title.hidden = false;
  title.textContent = `Examples: cluster ${clusterId}`;
  const rows = await (await fetch(`/api/cluster/${clusterId}/examples?n=24`)).json();
  examplesDiv.innerHTML = "";
  for (const row of rows) {
    const img = document.createElement("img");
    img.src = "/thumb/" + row.id;
    img.title = `${row.generator} | ${row.dataset} | ${row.label}${row.pruned ? " | PRUNED" : ""}\nctrl+click: copy path | alt+click: flag`;
    if (row.pruned) img.classList.add("pruned");
    if (row.flagged) img.classList.add("flagged");
    img.addEventListener("click", (e) => {
      if (e.ctrlKey) { copyPath(row.id); return; }
      if (e.altKey) {
        const i = pts.ids.indexOf(row.id);
        if (i >= 0) {
          toggleFlag(i).then(() => img.classList.toggle("flagged", !!pts.flagged[i]));
        }
        return;
      }
      openModal(row.id);
    });
    examplesDiv.appendChild(img);
  }
  draw();
}

function stepCluster(delta) {
  if (!clusterOrder.length) return;
  let idx = clusterOrder.indexOf(selectedCluster);
  idx = idx < 0 ? (delta > 0 ? 0 : clusterOrder.length - 1)
                : (idx + delta + clusterOrder.length) % clusterOrder.length;
  const clusterId = clusterOrder[idx];
  const li = document.querySelector(`#cluster-list li[data-cluster="${clusterId}"]`);
  selectedCluster = null;  // force select (not toggle-off)
  selectCluster(clusterId, li);
}

window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
  if (!modal.hidden) {
    if (e.key === "Escape") { e.preventDefault(); closeModal(); }
    return;
  }
  if (activeTab === "pairs") {
    if (e.key === "ArrowRight") { e.preventDefault(); stepPairsPage(1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); stepPairsPage(-1); }
    return;
  }
  if (e.key === "n" || e.key === "ArrowRight") { e.preventDefault(); stepCluster(1); }
  else if (e.key === "p" || e.key === "ArrowLeft") { e.preventDefault(); stepCluster(-1); }
  else if (e.key === "Escape" && selectedCluster !== null) {
    const li = document.querySelector(`#cluster-list li[data-cluster="${selectedCluster}"]`);
    selectCluster(selectedCluster, li);  // toggles off + fitView
  }
});

// --- Pruned pairs tab ---------------------------------------------------

function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll("#tabs .tab").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === tab));
  scatterView.hidden = tab !== "scatter";
  pairsView.hidden = tab !== "pairs";
  tooltip.hidden = true;
  if (tab === "scatter") { resize(); return; }
  if (!pairsState.loaded) { pairsState.loaded = true; initPairs(); }
}

async function initPairs() {
  const summary = await (await fetch("/api/pairs/summary")).json();
  const clusterSel = document.getElementById("pairs-cluster");
  for (const c of summary.clusters || []) {
    const opt = document.createElement("option");
    opt.value = c.cluster_id;
    opt.textContent = `#${c.cluster_id} (${c.pairs})`;
    clusterSel.appendChild(opt);
  }
  loadPairsPage();
}

function stepPairsPage(delta) {
  const maxPage = Math.max(0, Math.ceil(pairsState.total / pairsState.pageSize) - 1);
  const next = Math.min(maxPage, Math.max(0, pairsState.page + delta));
  if (next === pairsState.page && pairsState.total > 0) return;
  pairsState.page = next;
  loadPairsPage();
}

function pairCell(id, path, role, flagged) {
  const cell = document.createElement("div");
  cell.className = "cell";
  const label = document.createElement("div");
  label.className = "role" + (role === "pruned" ? " pruned" : "");
  label.textContent = role;
  cell.appendChild(label);
  const img = document.createElement("img");
  img.loading = "lazy";
  img.src = "/thumb/" + id;
  img.title = `${path}\nclick: full size | ctrl+click: copy path | alt+click: flag bad`;
  if (flagged) img.classList.add("flagged");
  img.addEventListener("error", () => {
    const missing = document.createElement("div");
    missing.className = "missing";
    missing.textContent = "file deleted";
    missing.title = path;
    img.replaceWith(missing);
  });
  img.addEventListener("click", (e) => {
    if (e.ctrlKey) { copyPath(id); return; }
    if (e.altKey) {
      const next = !img.classList.contains("flagged");
      setFlagById(id, next).then((ok) => { if (ok) img.classList.toggle("flagged", next); });
      return;
    }
    openModal(id);
  });
  cell.appendChild(img);
  return cell;
}

async function loadPairsPage() {
  const params = new URLSearchParams({
    page: pairsState.page,
    page_size: pairsState.pageSize,
  });
  if (pairsState.kind) params.set("kind", pairsState.kind);
  if (pairsState.cluster !== "") params.set("cluster", pairsState.cluster);
  const res = await fetch("/api/pairs?" + params);
  if (!res.ok) { showToast(`pairs fetch failed (${res.status})`); return; }
  const data = await res.json();
  pairsState.total = data.total;

  const hint = document.getElementById("pairs-hint");
  hint.hidden = !(data.total === 0);
  if (data.total === 0) {
    hint.textContent = data.hint || "no pairs match the current filters";
  }

  const grid = document.getElementById("pairs-grid");
  grid.innerHTML = "";
  for (const row of data.rows) {
    const card = document.createElement("div");
    card.className = "pair-card";
    const imgs = document.createElement("div");
    imgs.className = "pair-imgs";
    imgs.appendChild(pairCell(row.pruned_id, row.pruned_path, "pruned", row.pruned_flagged));
    imgs.appendChild(pairCell(row.kept_id, row.kept_path, "kept", row.kept_flagged));
    card.appendChild(imgs);
    const caption = document.createElement("div");
    caption.className = "pair-caption";
    const where = row.kind === "cluster" ? `cluster ${row.cluster_id} | dist ${row.dist.toFixed(3)}` : "dedup";
    caption.textContent = `${row.reason} | ${where}`;
    caption.title = `${row.pruned_path}\n${row.kept_path}`;
    card.appendChild(caption);
    grid.appendChild(card);
  }
  grid.scrollTop = 0;

  const maxPage = Math.max(1, Math.ceil(data.total / pairsState.pageSize));
  document.getElementById("pairs-page-label").textContent =
    data.total ? `page ${pairsState.page + 1} / ${maxPage} (${data.total.toLocaleString()} pairs)` : "";
  document.getElementById("pairs-prev").disabled = pairsState.page <= 0;
  document.getElementById("pairs-next").disabled = pairsState.page + 1 >= maxPage;
}

async function init() {
  pts = await (await fetch("/api/points")).json();
  document.getElementById("stats").textContent =
    `${pts.x.length.toLocaleString()} sampled points | ${pts.generators.length} generators`;
  document.getElementById("color-mode").addEventListener("change", (e) => {
    colorMode = e.target.value;
    draw();
  });
  document.getElementById("hide-pruned").addEventListener("change", (e) => {
    hidePruned = e.target.checked;
    draw();
  });
  document.querySelectorAll("#tabs .tab").forEach((b) =>
    b.addEventListener("click", () => switchTab(b.dataset.tab)));
  document.getElementById("pairs-kind").addEventListener("change", (e) => {
    pairsState.kind = e.target.value; pairsState.page = 0; loadPairsPage();
  });
  document.getElementById("pairs-cluster").addEventListener("change", (e) => {
    pairsState.cluster = e.target.value; pairsState.page = 0; loadPairsPage();
  });
  document.getElementById("pairs-prev").addEventListener("click", () => stepPairsPage(-1));
  document.getElementById("pairs-next").addEventListener("click", () => stepPairsPage(1));
  resize();
  fitView();
  draw();
  loadClusters();
}

window.addEventListener("resize", resize);
init();
