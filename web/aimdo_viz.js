import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let pollInterval = 500;
const FADE_TICKS = 6;
const GRAPH_POINTS = 120;

const execState = { running: false, node: null, progress: null };
let peakVramUsed = 0;

const STORAGE_KEY = "aimdo_viz_state";
function loadState() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; }
    catch { return {}; }
}
function saveState(patch) {
    const s = loadState();
    Object.assign(s, patch);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
}

// color palette — dark chrome, colored data
const C = {
    vram:       "#e67e22",
    torch:      "#2ecc71",
    pinned:     "#4a9eff",
    unloaded:   "#3a3a3a",
    torchCache: "#1a7a3a",
    python:     "#9b59b6",
    other:      "#505050",
    text:       "#b0b0b0",
    textDim:    "#707070",
    running:    "#b0b0b0",
    bg:         "#181818",
    headerBg:   "#202020",
    border:     "#2a2a2a",
    btn:        "#2a2a2a",
    btnText:    "#888",
    graphBg:    "#0e0e0e",
    gridLine:   "#1e1e1e",
    totalLine:  "#d0d0d0",
    capLine:    "#555",
    barBg:      "#222",
    fadeInFrom:  [255, 220, 0],
    fadeInTo:    [230, 126, 34],
    fadeOutFrom: [200, 60, 60],
    fadeOutTo:   [58, 58, 58],
};


function escHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function formatBytes(bytes) {
    if (bytes == null) return "?";
    if (bytes >= 1024 ** 3) return (bytes / 1024 ** 3).toFixed(1) + " GB";
    if (bytes >= 1024 ** 2) return (bytes / 1024 ** 2).toFixed(0) + " MB";
    return (bytes / 1024).toFixed(0) + " KB";
}

// rolling history — ring buffer to avoid shift()
const history = {
    torch_active: new Float64Array(GRAPH_POINTS),
    aimdo_usage: new Float64Array(GRAPH_POINTS),
    free_vram: new Float64Array(GRAPH_POINTS),
    total_vram: 1,
    head: 0,
    len: 0,
};

function pushHistory(data) {
    history.total_vram = data.total_vram;
    const i = history.head;
    // store non-overlapping values matching the bar logic
    if (data.aimdo_usage > 0) {
        history.aimdo_usage[i] = data.aimdo_usage;
        history.torch_active[i] = 0;
    } else {
        history.aimdo_usage[i] = 0;
        history.torch_active[i] = data.torch_active;
    }
    history.free_vram[i] = data.free_vram;
    history.head = (i + 1) % GRAPH_POINTS;
    if (history.len < GRAPH_POINTS) history.len++;
}

function historyGet(arr, idx) {
    // idx 0 = oldest, idx len-1 = newest
    return arr[(history.head - history.len + idx + GRAPH_POINTS) % GRAPH_POINTS];
}

function drawGraph(ctx, w, h) {
    const total = history.total_vram;
    const len = history.len;
    if (len < 2) return;

    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = C.gridLine;
    ctx.lineWidth = 1;
    for (const pct of [0.25, 0.5, 0.75]) {
        const y = Math.round(h - h * pct) + 0.5;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    const stepX = w / (GRAPH_POINTS - 1);
    const yFor = val => h - (val / total) * h;

    // aimdo area
    const dataStartX = (GRAPH_POINTS - len) * stepX;
    ctx.beginPath();
    ctx.moveTo(dataStartX, h);
    for (let i = 0; i < len; i++) {
        ctx.lineTo((GRAPH_POINTS - len + i) * stepX, yFor(historyGet(history.aimdo_usage, i)));
    }
    ctx.lineTo((GRAPH_POINTS - 1) * stepX, h);
    ctx.closePath();
    ctx.fillStyle = "rgba(230,126,34,0.35)";
    ctx.fill();

    // torch area stacked
    ctx.beginPath();
    ctx.moveTo(dataStartX, yFor(historyGet(history.aimdo_usage, 0)));
    for (let i = 0; i < len; i++) {
        const x = (GRAPH_POINTS - len + i) * stepX;
        ctx.lineTo(x, yFor(historyGet(history.aimdo_usage, i) + historyGet(history.torch_active, i)));
    }
    for (let i = len - 1; i >= 0; i--) {
        ctx.lineTo((GRAPH_POINTS - len + i) * stepX, yFor(historyGet(history.aimdo_usage, i)));
    }
    ctx.closePath();
    ctx.fillStyle = "rgba(46,204,113,0.4)";
    ctx.fill();

    // total used line
    ctx.beginPath();
    ctx.strokeStyle = C.totalLine;
    ctx.lineWidth = 1.5;
    for (let i = 0; i < len; i++) {
        const x = (GRAPH_POINTS - len + i) * stepX;
        const y = yFor(total - historyGet(history.free_vram, i));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // capacity line
    ctx.strokeStyle = C.capLine;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, yFor(total));
    ctx.lineTo(w, yFor(total));
    ctx.stroke();
    ctx.setLineDash([]);
}

// per-model residency diff state
const modelState = {};

function diffResidency(key, residency) {
    let st = modelState[key];
    if (!st || st.prev.length !== residency.length) {
        st = { prev: new Uint8Array(residency), changeAge: new Uint8Array(residency.length) };
        modelState[key] = st;
        return st;
    }

    for (let i = 0; i < residency.length; i++) {
        if (residency[i] !== st.prev[i]) {
            st.changeAge[i] = FADE_TICKS;
        } else if (st.changeAge[i] > 0) {
            st.changeAge[i]--;
        }
        st.prev[i] = residency[i];
    }
    return st;
}

// draw page grid to canvas — much faster than 700 DOM divs
function drawPageGrid(ctx, w, residency, changeAge) {
    const cellSize = 6;
    const gap = 1;
    const step = cellSize + gap;
    const cols = Math.floor((w + gap) / step);
    const rows = Math.ceil(residency.length / cols);
    const h = rows * step;

    ctx.canvas.height = h || 1;
    ctx.canvas.style.height = (h || 1) + "px";
    ctx.clearRect(0, 0, w, h);

    // batch: draw all static vram cells, then all static unloaded, then animated individually
    const animated = [];

    ctx.fillStyle = C.vram;
    for (let i = 0; i < residency.length; i++) {
        if (changeAge[i] > 0) { animated.push(i); continue; }
        if (!(residency[i] & 1)) continue;
        ctx.fillRect((i % cols) * step, Math.floor(i / cols) * step, cellSize, cellSize);
    }

    ctx.fillStyle = C.unloaded;
    for (let i = 0; i < residency.length; i++) {
        if (changeAge[i] > 0 || (residency[i] & 1)) continue;
        ctx.fillRect((i % cols) * step, Math.floor(i / cols) * step, cellSize, cellSize);
    }

    // animated cells need individual colors
    for (const i of animated) {
        const resident = residency[i] & 1;
        const t = changeAge[i] / FADE_TICKS;
        const [fr, fg, fb] = resident ? C.fadeInFrom : C.fadeOutFrom;
        const [tr, tg, tb] = resident ? C.fadeInTo : C.fadeOutTo;
        ctx.fillStyle = `rgb(${Math.round(fr * t + tr * (1 - t))},${Math.round(fg * t + tg * (1 - t))},${Math.round(fb * t + tb * (1 - t))})`;
        ctx.fillRect((i % cols) * step, Math.floor(i / cols) * step, cellSize, cellSize);
    }
}

function createPanel() {
    const saved = loadState();
    if (saved.pollInterval) pollInterval = saved.pollInterval;

    const panel = document.createElement("div");
    panel.id = "aimdo-viz-panel";
    panel.style.cssText = `
        position: fixed;
        background: ${C.bg}; color: ${C.text};
        border: 1px solid ${C.border}; border-radius: 8px;
        padding: 0; font-family: monospace; font-size: 12px;
        z-index: 50; min-width: 200px; width: 340px; max-height: 90vh;
        box-shadow: 0 4px 12px rgba(0,0,0,0.7);
        user-select: none; resize: horizontal; overflow-y: auto;
    `;
    if (saved.width != null) panel.style.width = saved.width + "px";

    // .graph-canvas-panel shrinks when sidebars open; #graph-canvas-container doesn't
    function getCanvasEl() {
        return document.querySelector(".graph-canvas-panel")
            || document.getElementById("graph-canvas-container")
            || document.getElementById("graph-canvas")
            || (app && app.canvasEl)
            || null;
    }

    function getCanvasBounds() {
        const el = getCanvasEl();
        if (el) {
            const r = el.getBoundingClientRect();
            if (r.width > 0 && r.height > 0) return r;
        }
        return { left: 0, top: 0, right: window.innerWidth, bottom: window.innerHeight };
    }

    let rightOffset = saved.rightOffset != null ? saved.rightOffset : 10;
    let bottomOffset = saved.bottomOffset != null ? saved.bottomOffset : 10;

    // viewport bounds, not canvas-panel bounds — topbar shouldn't trap the panel
    function clampOffsets(ro, bo) {
        const b = getCanvasBounds();
        const w = panel.offsetWidth, h = panel.offsetHeight;
        const vw = window.innerWidth, vh = window.innerHeight;
        return {
            ro: Math.max(b.right - vw, Math.min(ro, b.right - w)),
            bo: Math.max(b.bottom - vh, Math.min(bo, b.bottom - h)),
            b, w, h,
        };
    }

    // visual-only clamp; closure offsets stay as user intent so they survive temporary shrinks
    function applyOffsets() {
        const { ro, bo, b, w, h } = clampOffsets(rightOffset, bottomOffset);
        panel.style.left = (b.right - w - ro) + "px";
        panel.style.top = (b.bottom - h - bo) + "px";
        panel.style.right = "auto";
        panel.style.bottom = "auto";
    }
    window.addEventListener("resize", applyOffsets);

    const header = document.createElement("div");
    header.style.cssText = `
        display: flex; justify-content: space-between; align-items: center;
        padding: 6px 10px; background: ${C.headerBg};
        border-radius: 8px 8px 0 0; cursor: move;
    `;
    const titleSpan = document.createElement("span");
    titleSpan.style.cssText = `font-weight:bold;color:${C.text};white-space:nowrap;`;
    titleSpan.textContent = "Memory";
    header.appendChild(titleSpan);

    const miniBar = document.createElement("div");
    miniBar.style.cssText = `display:none;padding:4px 10px 6px;font-size:10px;color:${C.textDim};`;
    miniBar.innerHTML = `<div style="display:flex;justify-content:space-between;margin-bottom:2px;">
        <span>RAM</span><span class="mini-ram-usage"></span>
    </div>
    <div style="background:${C.barBg};border-radius:2px;height:4px;overflow:hidden;display:flex;margin-bottom:4px;" class="mini-ram-bar"></div>
    <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
        <span>VRAM</span><span class="mini-vram-usage"></span>
    </div>
    <div style="background:${C.barBg};border-radius:2px;height:4px;overflow:hidden;display:flex;" class="mini-vram-bar"></div>`;

    const headerRight = document.createElement("div");
    headerRight.style.cssText = "display:flex;align-items:center;gap:6px;";

    const intervalSelect = document.createElement("select");
    intervalSelect.title = "Polling interval — how often the panel refreshes VRAM/RAM stats";
    intervalSelect.style.cssText = `font-size:9px;background:${C.btn};color:${C.btnText};border:none;border-radius:2px;padding:1px 2px;cursor:pointer;`;
    for (const ms of [100, 250, 500, 1000, 2000, 5000]) {
        const opt = document.createElement("option");
        opt.value = ms;
        opt.textContent = ms < 1000 ? `${ms}ms` : `${ms/1000}s`;
        if (ms === pollInterval) opt.selected = true;
        intervalSelect.appendChild(opt);
    }
    intervalSelect.addEventListener("change", () => { pollInterval = parseInt(intervalSelect.value); saveState({ pollInterval }); });

    const unloadBtn = document.createElement("span");
    unloadBtn.textContent = "unload ▾";
    unloadBtn.title = "Unload models / free cache (click for options)";
    unloadBtn.style.cssText = `cursor:pointer;font-size:10px;padding:1px 6px;background:${C.btn};border-radius:3px;color:${C.btnText};white-space:nowrap;`;

    const unloadMenu = document.createElement("div");
    unloadMenu.style.cssText = `
        display:none; position:fixed; z-index:51;
        background:${C.headerBg}; color:${C.text};
        border:1px solid ${C.border}; border-radius:4px;
        padding:2px 0; min-width:160px;
        box-shadow:0 4px 12px rgba(0,0,0,0.7);
        font-family:monospace; font-size:10px;
    `;
    const unloadOptions = [
        { label: "aimdo (immediate)", title: "Immediately unload aimdo-managed models", run: () =>
            api.fetchApi("/aimdo/unload_all", { method: "POST" }) },
        { label: "models", title: "ComfyUI /free — unload models on next queue tick", run: () =>
            api.fetchApi("/free", { method: "POST",
                body: JSON.stringify({ unload_models: true }),
                headers: { "Content-Type": "application/json" } }) },
        { label: "models + node cache", title: "ComfyUI /free — unload models and clear node output cache", run: () =>
            api.fetchApi("/free", { method: "POST",
                body: JSON.stringify({ unload_models: true, free_memory: true }),
                headers: { "Content-Type": "application/json" } }) },
    ];
    for (const opt of unloadOptions) {
        const item = document.createElement("div");
        item.textContent = opt.label;
        item.title = opt.title;
        item.style.cssText = `padding:4px 10px;cursor:pointer;white-space:nowrap;`;
        item.addEventListener("mouseenter", () => item.style.background = C.btn);
        item.addEventListener("mouseleave", () => item.style.background = "");
        item.addEventListener("click", async (e) => {
            e.stopPropagation();
            unloadMenu.style.display = "none";
            unloadBtn.textContent = "...";
            try { await opt.run(); }
            finally { unloadBtn.textContent = "unload ▾"; }
        });
        unloadMenu.appendChild(item);
    }
    document.body.appendChild(unloadMenu);

    unloadBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (unloadMenu.style.display === "block") {
            unloadMenu.style.display = "none";
            return;
        }
        const r = unloadBtn.getBoundingClientRect();
        unloadMenu.style.left = Math.max(4, r.right - 160) + "px";
        unloadMenu.style.top = (r.bottom + 2) + "px";
        unloadMenu.style.display = "block";
    });
    document.addEventListener("click", (e) => {
        if (e.target !== unloadBtn && !unloadMenu.contains(e.target)) {
            unloadMenu.style.display = "none";
        }
    });

    const resetBtn = document.createElement("span");
    resetBtn.textContent = "reset";
    resetBtn.title = "Reset peak VRAM marker and clear history graph";
    resetBtn.style.cssText = `cursor:pointer;font-size:10px;padding:1px 6px;background:${C.btn};border-radius:3px;color:${C.btnText};`;
    resetBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        peakVramUsed = 0;
        history.head = 0;
        history.len = 0;
        history.torch_active.fill(0);
        history.aimdo_usage.fill(0);
        history.free_vram.fill(0);
    });

    const toggleBtn = document.createElement("span");
    toggleBtn.textContent = "\u2212";
    toggleBtn.title = "Collapse / expand panel";
    toggleBtn.style.cssText = `cursor:pointer;font-size:16px;padding:0 4px;color:${C.btnText};`;

    const body = document.createElement("div");
    body.id = "aimdo-viz-body";
    body.style.cssText = "padding: 8px 10px;";

    let collapsed = !!saved.collapsed;
    if (collapsed) {
        body.style.display = "none";
        toggleBtn.textContent = "+";
        miniBar.style.display = "block";
    }
    toggleBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        collapsed = !collapsed;
        body.style.display = collapsed ? "none" : "block";
        miniBar.style.display = collapsed ? "block" : "none";
        toggleBtn.textContent = collapsed ? "+" : "\u2212";
        saveState({ collapsed });
    });

    headerRight.appendChild(intervalSelect);
    headerRight.appendChild(resetBtn);
    headerRight.appendChild(unloadBtn);
    headerRight.appendChild(toggleBtn);
    header.appendChild(headerRight);
    panel.appendChild(header);
    panel.appendChild(miniBar);
    panel.appendChild(body);

    let dragging = false, dx = 0, dy = 0;
    header.addEventListener("mousedown", (e) => {
        dragging = true;
        dx = e.clientX - panel.offsetLeft;
        dy = e.clientY - panel.offsetTop;
    });
    document.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        const b = getCanvasBounds();
        const w = panel.offsetWidth, h = panel.offsetHeight;
        rightOffset = b.right - (e.clientX - dx) - w;
        bottomOffset = b.bottom - (e.clientY - dy) - h;
        applyOffsets();
    });
    document.addEventListener("mouseup", () => {
        if (dragging) {
            // clamp before persist
            const c = clampOffsets(rightOffset, bottomOffset);
            rightOffset = c.ro;
            bottomOffset = c.bo;
            applyOffsets();
            saveState({ rightOffset, bottomOffset });
        }
        dragging = false;
    });

    // edge resize handles — left grows leftward (right edge is dock anchor, ro stays);
    // right grows rightward (left edge gets anchored via ResizeObserver's ro delta)
    let suppressWidthAnchor = false;
    let edgeDrag = null;
    function makeEdgeHandle(side) {
        const h = document.createElement("div");
        h.title = "Drag to resize";
        h.style.cssText = `position:absolute;top:28px;bottom:0;${side}:0;width:4px;cursor:ew-resize;z-index:1;`;
        h.addEventListener("mousedown", (e) => {
            e.preventDefault();
            e.stopPropagation();
            edgeDrag = { side, startX: e.clientX, startWidth: panel.offsetWidth };
            if (side === "left") suppressWidthAnchor = true;
        });
        panel.appendChild(h);
    }
    makeEdgeHandle("left");
    makeEdgeHandle("right");

    document.addEventListener("mousemove", (e) => {
        if (!edgeDrag) return;
        const delta = e.clientX - edgeDrag.startX;
        const newWidth = edgeDrag.side === "left" ? edgeDrag.startWidth - delta : edgeDrag.startWidth + delta;
        panel.style.width = Math.max(200, newWidth) + "px";
    });
    document.addEventListener("mouseup", () => {
        if (edgeDrag) {
            edgeDrag = null;
            suppressWidthAnchor = false;
        }
    });

    document.body.appendChild(panel);
    applyOffsets();

    // bottom-right handle changes width → anchor left edge via ro delta;
    // left handle sets suppressWidthAnchor to keep the right edge anchored instead
    let lastPanelWidth = null;
    if (typeof ResizeObserver !== "undefined") {
        new ResizeObserver(() => {
            const w = panel.offsetWidth;
            if (lastPanelWidth !== null && w !== lastPanelWidth) {
                if (!suppressWidthAnchor) rightOffset -= (w - lastPanelWidth);
                saveState({ width: w, rightOffset, bottomOffset });
            }
            lastPanelWidth = w;
            applyOffsets();
        }).observe(panel);
    }

    // sidebar toggles don't fire window.resize. canvas may not exist yet — poll until it does.
    let observed = null;
    function attachCanvasObserver() {
        const el = getCanvasEl();
        if (!el) {
            requestAnimationFrame(attachCanvasObserver);
            return;
        }
        if (el === observed) return;
        observed = el;
        if (typeof ResizeObserver !== "undefined") {
            new ResizeObserver(applyOffsets).observe(el);
        }
        applyOffsets();
    }
    attachCanvasObserver();
    body._titleSpan = titleSpan;
    body._miniBar = miniBar;
    body._panel = panel;
    return body;
}

// persistent DOM refs to avoid re-querying / re-creating
let refs = null;

function ensureStructure(body) {
    if (refs) return refs;

    body.innerHTML = "";

    const contentDiv = document.createElement("div");
    contentDiv.id = "aimdo-content";
    body.appendChild(contentDiv);

    const graphCanvas = document.createElement("canvas");
    graphCanvas.width = 300;
    graphCanvas.height = 80;
    graphCanvas.style.cssText = `width:100%;height:80px;border-radius:3px;background:${C.graphBg};`;
    body.appendChild(graphCanvas);

    const modelsDiv = document.createElement("div");
    modelsDiv.id = "aimdo-models";
    body.appendChild(modelsDiv);

    refs = {
        contentDiv,
        graphCanvas,
        graphCtx: graphCanvas.getContext("2d"),
        modelsDiv,
        pageCanvases: {},   // keyed by `${index}_${vi}`
        pageCtxs: {},
        modelRows: {},      // keyed by m.index — refs to mutable row parts
        noModelsMsg: null,
        bottomLegend: null,
    };
    return refs;
}

// build (or reuse) a model row, mutating only what changed
function renderModelRow(r, m, data) {
    const wantsWm = m.dynamic && data.aimdo_active;
    const btnStyle = `cursor:pointer;font-size:9px;padding:0px 4px;background:${C.btn};border-radius:2px;color:${C.btnText};`;
    let row = r.modelRows[m.index];
    if (!row) {
        const el = document.createElement("div");
        el.style.cssText = "margin-top:6px;";
        const head = document.createElement("div");
        head.style.cssText = "display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;";
        const nameSpan = document.createElement("span");
        const right = document.createElement("span");
        right.style.cssText = "display:flex;align-items:center;gap:6px;";
        const sizeSpan = document.createElement("span");
        right.appendChild(sizeSpan);
        const unloadBtn = document.createElement("span");
        unloadBtn.className = "aimdo-unload-btn";
        unloadBtn.dataset.index = m.index;
        unloadBtn.textContent = "x";
        unloadBtn.title = "Unload this model";
        unloadBtn.style.cssText = btnStyle;
        right.appendChild(unloadBtn);
        head.appendChild(nameSpan);
        head.appendChild(right);
        el.appendChild(head);
        const bar = document.createElement("div");
        bar.style.cssText = `background:${C.barBg};border-radius:3px;height:10px;overflow:hidden;display:flex;`;
        el.appendChild(bar);
        const legend = document.createElement("div");
        legend.style.cssText = `display:flex;gap:8px;font-size:10px;color:${C.textDim};margin-top:2px;`;
        el.appendChild(legend);
        const vbarsDiv = document.createElement("div");
        el.appendChild(vbarsDiv);
        row = { el, nameSpan, sizeSpan, right, unloadBtn, bar, barSegs: [], legend, vbarsDiv, vbarRefs: [], wmBtn: null, lastDynamic: null, lastVbarSig: "" };
        r.modelRows[m.index] = row;
    }

    row.nameSpan.textContent = m.name + (m.dynamic ? "" : " (static)");
    row.sizeSpan.textContent = formatBytes(m.total_size);

    if (wantsWm && !row.wmBtn) {
        const wm = document.createElement("span");
        wm.className = "aimdo-reset-wm-btn";
        wm.dataset.index = m.index;
        wm.textContent = "wm";
        wm.title = "reset watermark";
        wm.style.cssText = btnStyle;
        row.right.insertBefore(wm, row.unloadBtn);
        row.wmBtn = wm;
    } else if (!wantsWm && row.wmBtn) {
        row.wmBtn.remove();
        row.wmBtn = null;
    }

    if (row.lastDynamic !== m.dynamic) {
        row.bar.innerHTML = "";
        row.barSegs = [];
        for (const color of (m.dynamic ? [C.vram, C.pinned, C.unloaded] : [C.vram, C.pinned])) {
            const seg = document.createElement("div");
            seg.style.cssText = `background:${color};height:100%;`;
            row.bar.appendChild(seg);
            row.barSegs.push(seg);
        }
        row.lastDynamic = m.dynamic;
    }

    if (m.dynamic) {
        const pinnedRam = m.pinned_ram || 0;
        const unloadedSize = Math.max(0, m.total_size - m.vbar_loaded - pinnedRam);
        const total = m.total_size || 1;
        row.barSegs[0].style.width = (m.vbar_loaded / total * 100) + "%";
        row.barSegs[0].title = "VRAM: " + formatBytes(m.vbar_loaded);
        row.barSegs[1].style.width = (pinnedRam / total * 100) + "%";
        row.barSegs[1].title = "pinned RAM: " + formatBytes(pinnedRam);
        row.barSegs[2].style.width = (unloadedSize / total * 100) + "%";
        row.barSegs[2].title = "unloaded: " + formatBytes(unloadedSize);
        row.legend.innerHTML =
            `<span><span style="color:${C.vram};">&#9632;</span> VRAM ${formatBytes(m.vbar_loaded)}</span>` +
            (pinnedRam > 0 ? `<span><span style="color:${C.pinned};">&#9632;</span> pinned ${formatBytes(pinnedRam)}</span>` : "") +
            `<span><span style="color:${C.unloaded};">&#9632;</span> unloaded ${formatBytes(unloadedSize)}</span>`;
    } else {
        const inRam = Math.max(0, m.total_size - m.loaded_size);
        const total = m.total_size || 1;
        row.barSegs[0].style.width = (m.loaded_size / total * 100) + "%";
        row.barSegs[0].title = "VRAM: " + formatBytes(m.loaded_size);
        row.barSegs[1].style.width = (inRam / total * 100) + "%";
        row.barSegs[1].title = "RAM: " + formatBytes(inRam);
        row.legend.innerHTML =
            `<span><span style="color:${C.vram};">&#9632;</span> VRAM ${formatBytes(m.loaded_size)}</span>` +
            (inRam > 0 ? `<span><span style="color:${C.pinned};">&#9632;</span> RAM ${formatBytes(inRam)}</span>` : "");
    }

    // vbars: rebuild structure only when device list / count changes
    const vbars = (m.vbars || []).filter(v => v.residency && v.residency.length > 0);
    const sig = vbars.map(v => v.device + ":" + v.residency.length).join("|");
    if (row.lastVbarSig !== sig) {
        row.vbarsDiv.innerHTML = "";
        row.vbarRefs = [];
        const showLabel = vbars.length > 1;
        for (let vi = 0; vi < vbars.length; vi++) {
            const vb = vbars[vi];
            if (showLabel) {
                const lbl = document.createElement("div");
                lbl.style.cssText = `font-size:10px;color:${C.textDim};margin-top:3px;`;
                lbl.textContent = vb.device;
                row.vbarsDiv.appendChild(lbl);
            }
            const pgrid = document.createElement("div");
            pgrid.style.cssText = "margin-top:2px;";
            row.vbarsDiv.appendChild(pgrid);
            const stats = document.createElement("div");
            stats.style.cssText = `color:${C.textDim};font-size:10px;margin-top:2px;`;
            row.vbarsDiv.appendChild(stats);
            row.vbarRefs.push({ vi, pgrid, stats });
        }
        row.lastVbarSig = sig;
    }
    return row;
}

function renderData(body, data) {
    if (!data.enabled) {
        body.innerHTML = `<div style="color:${C.textDim};">not available</div>`;
        refs = null;
        return;
    }

    const r = ensureStructure(body);
    const pw = body._panel.offsetWidth;
    body._titleSpan.textContent =
        pw >= 320 && data.aimdo_active ? "Memory (aimdo)" :
        pw >= 240 ? "Memory" : "";
    pushHistory(data);

    const used = data.total_vram - data.free_vram;
    if (used > peakVramUsed) peakVramUsed = used;

    // aimdo allocates through pytorch's caching allocator, so aimdo_usage
    // and torch_reserved overlap. Derive non-overlapping segments from
    // the driver-level total (used) as ground truth.
    let aimdo, torchActive, torchCache, otherUsed;
    if (data.aimdo_usage > 0) {
        // aimdo active: torch stats are a subset of aimdo, not additive
        aimdo = data.aimdo_usage;
        torchActive = 0;
        torchCache = 0;
        otherUsed = Math.max(0, used - aimdo);
    } else {
        // no aimdo: torch stats are the full picture
        aimdo = 0;
        torchActive = data.torch_active;
        torchCache = Math.max(0, data.torch_reserved - data.torch_active);
        otherUsed = Math.max(0, used - data.torch_reserved);
    }
    const aimdoPct = (aimdo / data.total_vram * 100).toFixed(0);
    const torchPct = (torchActive / data.total_vram * 100).toFixed(0);
    const torchCachePct = (torchCache / data.total_vram * 100).toFixed(0);
    const otherPct = (otherUsed / data.total_vram * 100).toFixed(0);

    const ramUsed = data.used_ram || 0;
    const ramTotal = data.total_ram || 1;
    const processRam = data.process_ram || 0;
    const pinnedRamTotal = data.pinned_ram || 0;
    const pythonOther = Math.max(0, processRam - pinnedRamTotal);
    const ramOther = Math.max(0, ramUsed - processRam);
    const pinnedRamPct = (pinnedRamTotal / ramTotal * 100).toFixed(0);
    const pythonOtherPct = (pythonOther / ramTotal * 100).toFixed(0);
    const ramOtherPct = (ramOther / ramTotal * 100).toFixed(0);

    const mb = body._miniBar;
    mb.querySelector(".mini-vram-usage").textContent = `${formatBytes(used)} / ${formatBytes(data.total_vram)}`;
    mb.querySelector(".mini-vram-bar").innerHTML =
        `<div style="background:${C.vram};height:100%;width:${aimdoPct}%;"></div>` +
        `<div style="background:${C.torch};height:100%;width:${torchPct}%;"></div>` +
        `<div style="background:${C.torchCache};height:100%;width:${torchCachePct}%;"></div>` +
        `<div style="background:${C.other};height:100%;width:${otherPct}%;"></div>`;
    mb.querySelector(".mini-ram-usage").textContent = `${formatBytes(ramUsed)} / ${formatBytes(ramTotal)}`;
    mb.querySelector(".mini-ram-bar").innerHTML =
        `<div style="background:${C.pinned};height:100%;width:${pinnedRamPct}%;"></div>` +
        `<div style="background:${C.python};height:100%;width:${pythonOtherPct}%;"></div>` +
        `<div style="background:${C.other};height:100%;width:${ramOtherPct}%;"></div>`;

    r.contentDiv.innerHTML = `<div style="margin-bottom:4px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
            <span>RAM</span>
            <span>${formatBytes(ramUsed)} / ${formatBytes(ramTotal)}</span>
        </div>
        <div style="background:${C.barBg};border-radius:3px;height:8px;overflow:hidden;display:flex;">
            <div style="background:${C.pinned};height:100%;width:${pinnedRamPct}%;" title="pinned: ${formatBytes(pinnedRamTotal)}"></div>
            <div style="background:${C.python};height:100%;width:${pythonOtherPct}%;" title="python: ${formatBytes(pythonOther)}"></div>
            <div style="background:${C.other};height:100%;width:${ramOtherPct}%;" title="other: ${formatBytes(ramOther)}"></div>
        </div>
        <div style="display:flex;gap:8px;font-size:10px;color:${C.textDim};margin-top:2px;">
            <span><span style="color:${C.pinned};">&#9632;</span> pinned ${formatBytes(pinnedRamTotal)}</span>
            <span><span style="color:${C.python};">&#9632;</span> python ${formatBytes(pythonOther)}</span>
            <span><span style="color:${C.other};">&#9632;</span> other ${formatBytes(ramOther)}</span>
        </div>
    </div>
    <div style="margin-bottom:4px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
            <span>VRAM</span>
            <span>${formatBytes(used)} / ${formatBytes(data.total_vram)}</span>
        </div>
        <div style="background:${C.barBg};border-radius:3px;height:8px;overflow:hidden;display:flex;">
            <div style="background:${C.vram};height:100%;width:${aimdoPct}%;" title="models: ${formatBytes(aimdo)}"></div>
            <div style="background:${C.torch};height:100%;width:${torchPct}%;" title="torch: ${formatBytes(torchActive)}"></div>
            <div style="background:${C.torchCache};height:100%;width:${torchCachePct}%;" title="cache: ${formatBytes(torchCache)}"></div>
            <div style="background:${C.other};height:100%;width:${otherPct}%;" title="other: ${formatBytes(otherUsed)}"></div>
        </div>
        <div style="display:flex;gap:8px;font-size:10px;color:${C.textDim};margin-top:2px;">
            ${aimdo > 0 ? `<span><span style="color:${C.vram};">&#9632;</span> models ${formatBytes(aimdo)}</span>` : ""}
            ${torchActive > 0 ? `<span><span style="color:${C.torch};">&#9632;</span> torch ${formatBytes(torchActive)}</span>` : ""}
            ${torchCache > 0 ? `<span><span style="color:${C.torchCache};">&#9632;</span> cache ${formatBytes(torchCache)}</span>` : ""}
            <span><span style="color:${C.other};">&#9632;</span> other ${formatBytes(otherUsed)}</span>
        </div>
        <div style="display:flex;gap:10px;font-size:10px;color:${C.textDim};margin-top:2px;">
            <span>peak: ${formatBytes(peakVramUsed)}</span>
            <span>cache: ${formatBytes(data.torch_reserved - data.torch_active)}</span>
            ${execState.running ? `<span style="color:${C.running};">&#9679; ${execState.node || "running"}${execState.progress ? " " + execState.progress : ""}</span>` : `<span>&#9679; idle</span>`}
        </div>
    </div>`;

    // sync canvas resolution to display size
    const displayW = r.graphCanvas.clientWidth || 300;
    if (r.graphCanvas.width !== displayW) r.graphCanvas.width = displayW;
    drawGraph(r.graphCtx, r.graphCanvas.width, r.graphCanvas.height);

    // models section — incremental DOM updates: keep rows across polls, only mutate text/widths
    if (data.models.length === 0 && !r.noModelsMsg) {
        r.noModelsMsg = document.createElement("div");
        r.noModelsMsg.textContent = "No models loaded";
        r.noModelsMsg.style.cssText = `color:${C.textDim};margin-top:6px;`;
        r.modelsDiv.insertBefore(r.noModelsMsg, r.modelsDiv.firstChild);
    } else if (data.models.length > 0 && r.noModelsMsg) {
        r.noModelsMsg.remove();
        r.noModelsMsg = null;
    }

    // remove rows for models no longer present and clean up their canvas refs
    const liveIndices = new Set(data.models.map(m => m.index));
    for (const idx of Object.keys(r.modelRows)) {
        if (!liveIndices.has(parseInt(idx))) {
            r.modelRows[idx].el.remove();
            delete r.modelRows[idx];
        }
    }
    for (const key of Object.keys(r.pageCanvases)) {
        const idx = parseInt(key.split("_")[0]);
        if (!liveIndices.has(idx)) {
            delete r.pageCanvases[key];
            delete r.pageCtxs[key];
            delete modelState[key];
        }
    }

    // create/update each row, append new ones before the bottom legend
    for (const m of data.models) {
        const isNew = !r.modelRows[m.index];
        const row = renderModelRow(r, m, data);
        if (isNew) {
            if (r.bottomLegend) r.modelsDiv.insertBefore(row.el, r.bottomLegend);
            else r.modelsDiv.appendChild(row.el);
        }
    }

    // bottom legend — created once
    if (!r.bottomLegend) {
        r.bottomLegend = document.createElement("div");
        r.bottomLegend.style.cssText = `display:flex;flex-wrap:wrap;gap:8px;font-size:10px;color:${C.textDim};margin-top:6px;border-top:1px solid ${C.border};padding-top:4px;`;
        r.bottomLegend.innerHTML =
            `<span><span style="color:${C.vram};">&#9632;</span> VRAM</span>` +
            `<span><span style="color:${C.pinned};">&#9632;</span> pinned</span>` +
            `<span><span style="color:${C.unloaded};">&#9632;</span> unloaded</span>` +
            `<span><span style="color:${C.torch};">&#9632;</span> torch</span>` +
            `<span><span style="color:${C.totalLine};">&#9472;</span> total used</span>`;
        r.modelsDiv.appendChild(r.bottomLegend);
    }

    // draw page grids and update vbar stat text
    for (const m of data.models) {
        const row = r.modelRows[m.index];
        if (!row || !row.vbarRefs.length) continue;
        const vbars = (m.vbars || []).filter(v => v.residency && v.residency.length > 0);
        for (let vi = 0; vi < row.vbarRefs.length; vi++) {
            const vb = vbars[vi];
            if (!vb) continue;
            const ref = row.vbarRefs[vi];
            const vkey = `${m.index}_${ref.vi}`;
            const st = diffResidency(vkey, vb.residency);
            let residentCount = 0, pinnedCount = 0;
            for (let i = 0; i < vb.residency.length; i++) {
                const flag = vb.residency[i];
                if (flag & 2) pinnedCount++;
                else if (flag & 1) residentCount++;
            }
            const PAGE = 32 * 1024 * 1024;
            const vramPages = residentCount + pinnedCount;
            const ramPages = vb.residency.length - vramPages;
            ref.stats.innerHTML =
                `<span style="color:${C.vram};">${vramPages} VRAM (${formatBytes(vramPages * PAGE)})</span>` +
                ` + <span style="color:${C.unloaded};">${ramPages} unloaded (${formatBytes(ramPages * PAGE)})</span>`;

            let canvas = r.pageCanvases[vkey];
            if (!canvas) {
                canvas = document.createElement("canvas");
                canvas.style.cssText = "width:100%;border-radius:2px;";
                r.pageCanvases[vkey] = canvas;
                r.pageCtxs[vkey] = canvas.getContext("2d");
            }
            if (canvas.parentElement !== ref.pgrid) ref.pgrid.appendChild(canvas);
            canvas.width = ref.pgrid.clientWidth || r.modelsDiv.clientWidth || 300;
            drawPageGrid(r.pageCtxs[vkey], canvas.width, vb.residency, st ? st.changeAge : new Uint8Array(vb.residency.length));
        }
    }

    // attach button handlers via event delegation (once)
    if (!r.modelsDiv._delegated) {
        r.modelsDiv._delegated = true;
        r.modelsDiv.addEventListener("click", async (e) => {
            const wmBtn = e.target.closest(".aimdo-reset-wm-btn");
            if (wmBtn) {
                const idx = parseInt(wmBtn.dataset.index);
                wmBtn.textContent = "...";
                try {
                    await api.fetchApi("/aimdo/reset_watermark", {
                        method: "POST",
                        body: JSON.stringify({ index: idx }),
                        headers: { "Content-Type": "application/json" },
                    });
                } finally {
                    wmBtn.textContent = "wm";
                }
                return;
            }
            const unloadBtn = e.target.closest(".aimdo-unload-btn");
            if (unloadBtn) {
                const idx = parseInt(unloadBtn.dataset.index);
                unloadBtn.textContent = "...";
                try {
                    await api.fetchApi("/aimdo/unload_model", {
                        method: "POST",
                        body: JSON.stringify({ index: idx }),
                        headers: { "Content-Type": "application/json" },
                    });
                } catch { /* next poll will reflect state */ }
            }
        });
    }
}

app.registerExtension({
    name: "aimdo.VRAMVisualization",
    async setup() {
        const body = createPanel();

        api.addEventListener("execution_start", () => {
            execState.running = true;
            execState.node = null;
            execState.progress = null;
        });
        api.addEventListener("executing", ({ detail }) => {
            execState.running = detail != null;
            execState.node = null;
            execState.progress = null;
        });
        api.addEventListener("progress", ({ detail }) => {
            if (detail) {
                execState.progress = `${detail.value}/${detail.max}`;
            }
        });

        async function poll() {
            try {
                const resp = await api.fetchApi("/aimdo/vram");
                const data = await resp.json();
                renderData(body, data);
            } catch (e) {
                body.innerHTML = `<div style="color:#aa5555;">Error fetching data</div>`;
                refs = null;
            }
            setTimeout(poll, pollInterval);
        }

        poll();
    }
});
