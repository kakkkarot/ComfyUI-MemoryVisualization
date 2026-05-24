	import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let pollInterval = 500;
const FADE_TICKS = 6;
const GRAPH_POINTS = 120;        // visible window size (data points drawn on the canvas at once)
const HISTORY_BUFFER = 1200;     // total points retained for scrollback (~20 min at 1s history tick)
const HISTORY_TICK_MS = 1000;    // history snapshots fire at most this often regardless of poll rate

const execState = { running: false, node: null, progress: null };
let peakVramUsed = 0;
let gpuLineVisible = true;
let modelCollapsed = {};
let colorModelBars = false;
let colorModelStroke = true;
let colorModelName = true;
let showLegends = true;
let showRamInMini = true;
let showVramInMini = true;
let showGpuInMini = true;
let showCpuInMini = true;
let showHwNames = true;
let showTitle = true;
let showExecBtn = false;  // optional play / cancel-running button in the header
let miniShowNumbers = true;
let miniShowUnits = true;
let miniShowType = true;
let miniShowGpuTemp = true;
let miniShowGpuPower = true;
let graphHeight = 80;
let currentTheme = "default";

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

// Themes live in aimdo_viz.css as `[data-aimdo-theme="<name>"]` blocks layered
// over a :root default. Selecting a theme is just setting that attribute on
// <html> — CSS does the rest. JS keeps a parallel `C` palette object only
// because <canvas> can't read CSS variables; canvas rendering reads hex/rgb
// strings from C, which we refresh from computed CSS on each theme switch.
const THEME_NAMES = ["default", "light", "sepia", "fallout", "pink", "lucifer"];

// Keys whose values are color strings the canvas can read directly. fadeOutFrom /
// fadeOutTo are stored as comma-separated RGB triplets in CSS and parsed into
// [r,g,b] arrays for the canvas fade animation.
const PALETTE_KEYS = Object.freeze([
    "vram","torch","pinned","loadedRam","unloaded","torchCache","python","other",
    "text","textDim","running","bg","rowBg","headerBg","border","btn","btnText",
    "graphBg","gridLine","totalLine","gpuUtil","gpuUtilHi","capLine","barBg",
]);
const MODEL_TYPE_KEYS = Object.freeze([
    "model","vae","clip","clip_vision","controlnet","style_model","gligen","upscale_model",
]);
const RGB_KEYS = Object.freeze(["fadeOutFrom", "fadeOutTo"]);

// Filled by applyPalette() from computed CSS once the stylesheet is loaded.
// Canvas drawing reads from here because <canvas> can't read CSS variables;
// the DOM reads var(--aimdo-X) directly from CSS and never touches C.
const C = {};
const MODEL_TYPE_COLOR = {};

// Load the stylesheet and expose the load as a promise. Init code awaits this
// before touching C so the canvas never draws against an unpopulated palette.
const cssLoaded = new Promise((resolve) => {
    const id = "aimdo-viz-stylesheet";
    if (document.getElementById(id)) { resolve(); return; }
    const link = document.createElement("link");
    link.id = id;
    link.rel = "stylesheet";
    link.href = new URL("aimdo_viz.css", import.meta.url).href;
    link.addEventListener("load", () => resolve(), { once: true });
    link.addEventListener("error", () => resolve(), { once: true });  // proceed even on 404
    document.head.appendChild(link);
});

function parseRgbTriplet(s) {
    const parts = s.split(",").map(x => parseInt(x.trim(), 10));
    if (parts.length === 3 && parts.every(Number.isFinite)) return parts;
    return null;
}

// Set the theme on <html> so the matching CSS block takes effect, then
// refresh C from computed CSS variables for canvas use.
function applyPalette(name) {
    const root = document.documentElement;
    if (name && name !== "default" && THEME_NAMES.includes(name)) {
        root.dataset.aimdoTheme = name;
    } else {
        delete root.dataset.aimdoTheme;
    }
    const cs = getComputedStyle(root);
    for (const k of PALETTE_KEYS) {
        const v = cs.getPropertyValue(`--aimdo-${k}`).trim();
        if (v) C[k] = v;
    }
    for (const k of MODEL_TYPE_KEYS) {
        const v = cs.getPropertyValue(`--aimdo-type-${k}`).trim();
        if (v) MODEL_TYPE_COLOR[k] = v;
    }
    for (const k of RGB_KEYS) {
        const v = cs.getPropertyValue(`--aimdo-${k}`).trim();
        const rgb = v && parseRgbTriplet(v);
        if (rgb) C[k] = rgb;
    }
}

function hexToRgb(hex) {
    const h = hex.replace("#", "");
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}
function hexToRgba(hex, alpha) {
    const [r, g, b] = hexToRgb(hex);
    return `rgba(${r},${g},${b},${alpha})`;
}
// blend [r,g,b] toward white. Used so fade-in stays in the type's hue family.
function lightenRgb([r, g, b], amount) {
    return [
        Math.round(r + (255 - r) * amount),
        Math.round(g + (255 - g) * amount),
        Math.round(b + (255 - b) * amount),
    ];
}


function escHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function gpuUtilColor(pct) {
    if (pct < 10) return C.textDim;
    if (pct < 80) return C.gpuUtil;
    return C.gpuUtilHi;
}

function gpuTempColor(c) {
    if (c < 60) return C.textDim;
    if (c < 80) return C.gpuUtil;
    return C.gpuUtilHi;
}

function gpuPowerColor(draw_mW, limit_mW) {
    if (draw_mW == null) return C.textDim;
    if (limit_mW == null || limit_mW <= 0) return C.gpuUtil;
    return gpuUtilColor(draw_mW / limit_mW * 100);
}

function formatPower(draw_mW, limit_mW, withUnit = true) {
    if (draw_mW == null) return null;
    const draw = Math.round(draw_mW / 1000);
    const u = withUnit ? "W" : "";
    if (limit_mW == null || limit_mW <= 0) return `${draw}${u}`;
    return `${draw}/${Math.round(limit_mW / 1000)}${u}`;
}

function formatClock(ms) {
    const d = new Date(ms);
    const pad = n => n.toString().padStart(2, "0");
    return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

function shortenGpuName(name) {
    return name.replace(/^NVIDIA\s+/i, "").replace(/^GeForce\s+/i, "").replace(/\s+Laptop GPU$/i, "");
}

function shortenCpuName(name) {
    return name
        .replace(/\(R\)|\(TM\)|\(tm\)/gi, "")
        .replace(/\s+CPU\s+@.*$/i, "")           // "i9-12900K CPU @ 3.20GHz" → "i9-12900K"
        .replace(/\s+\d+-Core\s+Processor$/i, "")// "Ryzen 9 7950X 16-Core Processor" → "Ryzen 9 7950X"
        .replace(/^Intel\s+Core\s+/i, "")
        .replace(/^AMD\s+/i, "")
        .replace(/\s+/g, " ")
        .trim();
}

function formatBytes(bytes, withUnit = true) {
    if (bytes == null) return "?";
    // non-breaking space ( ) so "8.4 GB" never line-wraps between value and unit.
    if (bytes >= 1024 ** 3) return (bytes / 1024 ** 3).toFixed(1) + (withUnit ? " GB" : "");
    if (bytes >= 1024 ** 2) return (bytes / 1024 ** 2).toFixed(0) + (withUnit ? " MB" : "");
    return (bytes / 1024).toFixed(0) + (withUnit ? " KB" : "");
}

// rolling history — ring buffer for ~20 min of data at 1s history tick; the visible window of
// GRAPH_POINTS slides along this buffer based on viewOffset (0 = following live).
const history = {
    torch_active: new Float64Array(HISTORY_BUFFER),
    aimdo_usage: new Float64Array(HISTORY_BUFFER),
    free_vram: new Float64Array(HISTORY_BUFFER),
    gpu_util: new Float64Array(HISTORY_BUFFER),
    times: new Float64Array(HISTORY_BUFFER),    // ms timestamps; absolute so pollInterval changes don't distort
    total_vram: 1,
    head: 0,
    len: 0,
    viewOffset: 0,      // points back from newest; 0 = right edge (live)
    followLive: true,
    execEvents: [],     // {type: "start"|"end", time: ms} — drawn as vertical marks
};
const EXEC_EVENTS_MAX = 200;
const EXEC_NOOP_MS = 1500;  // start→end shorter than this is treated as a queue-with-no-changes; both events dropped
function pushExecEvent(type) {
    if (type === "end" && history.execEvents.length > 0) {
        const last = history.execEvents[history.execEvents.length - 1];
        if (last.type === "start" && Date.now() - last.time < EXEC_NOOP_MS) {
            history.execEvents.pop();
            return;
        }
    }
    history.execEvents.push({ type, time: Date.now() });
    if (history.execEvents.length > EXEC_EVENTS_MAX) {
        history.execEvents.splice(0, history.execEvents.length - EXEC_EVENTS_MAX);
    }
}

// NVML util frequently dips to 0 mid-workload; peak-hold preserves real peaks.
const GPU_SMOOTH_WINDOW = 3;
const gpuRawBuf = [];
function smoothGpuUtil(raw) {
    if (raw == null) {
        gpuRawBuf.length = 0;
        return null;
    }
    gpuRawBuf.push(raw);
    if (gpuRawBuf.length > GPU_SMOOTH_WINDOW) gpuRawBuf.shift();
    let m = 0;
    for (const v of gpuRawBuf) if (v > m) m = v;
    return m;
}

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
    history.gpu_util[i] = data.gpu_util != null ? data.gpu_util : 0;
    history.times[i] = Date.now();
    history.head = (i + 1) % HISTORY_BUFFER;
    if (history.len < HISTORY_BUFFER) history.len++;
    // when scrolled back, advance viewOffset so the user's pinned window keeps
    // showing the same chronological data instead of sliding with new points.
    if (!history.followLive) {
        const maxOffset = Math.max(0, history.len - GRAPH_POINTS);
        history.viewOffset = Math.min(maxOffset, history.viewOffset + 1);
    }
}

function historyGet(arr, idx) {
    // idx 0 = oldest valid, idx len-1 = newest
    return arr[(history.head - history.len + idx + HISTORY_BUFFER) % HISTORY_BUFFER];
}

// history persistence — separate localStorage key from the panel settings since
// it's larger and changes constantly. ~50 KB at full buffer.
const HISTORY_STORAGE_KEY = "aimdo_viz_history";
function saveHistory() {
    try {
        const len = history.len;
        if (len === 0) { localStorage.removeItem(HISTORY_STORAGE_KEY); return; }
        // serialize in chronological order so the ring's head/wrap is irrelevant on load
        const ordered = arr => {
            const out = new Array(len);
            for (let i = 0; i < len; i++) {
                out[i] = arr[(history.head - len + i + HISTORY_BUFFER) % HISTORY_BUFFER];
            }
            return out;
        };
        localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify({
            v: 1,
            len,
            total_vram: history.total_vram,
            times: ordered(history.times),
            torch_active: ordered(history.torch_active),
            aimdo_usage: ordered(history.aimdo_usage),
            free_vram: ordered(history.free_vram),
            gpu_util: ordered(history.gpu_util),
            execEvents: history.execEvents,
        }));
    } catch {
        // quota or other failure — silently skip; next save attempt may succeed
    }
}
function loadHistory() {
    try {
        const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
        if (!raw) return;
        const data = JSON.parse(raw);
        if (!data || data.v !== 1 || typeof data.len !== "number") return;
        const len = Math.min(data.len, HISTORY_BUFFER);
        history.total_vram = data.total_vram || 1;
        for (let i = 0; i < len; i++) {
            history.times[i] = data.times[i];
            history.torch_active[i] = data.torch_active[i];
            history.aimdo_usage[i] = data.aimdo_usage[i];
            history.free_vram[i] = data.free_vram[i];
            history.gpu_util[i] = data.gpu_util[i];
        }
        history.len = len;
        history.head = len % HISTORY_BUFFER;
        if (Array.isArray(data.execEvents)) history.execEvents = data.execEvents.slice(-EXEC_EVENTS_MAX);
    } catch {
        // corrupted blob — leave history empty
    }
}
loadHistory();
setInterval(saveHistory, 10000);
window.addEventListener("beforeunload", saveHistory);

function drawGraph(ctx, w, h) {
    const total = history.total_vram;
    const len = history.len;
    if (len < 2) return;

    // windowed slice: render `visible` points starting at chronological index `startIdx`.
    const visible = Math.min(GRAPH_POINTS, len - history.viewOffset);
    if (visible < 2) return;
    const startIdx = len - visible - history.viewOffset;

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
    // right-align partial windows so new data flows in from the right
    const dataStartX = (GRAPH_POINTS - visible) * stepX;
    const xFor = i => (GRAPH_POINTS - visible + i) * stepX;
    const at = (arr, i) => historyGet(arr, startIdx + i);

    // aimdo area
    ctx.beginPath();
    ctx.moveTo(dataStartX, h);
    for (let i = 0; i < visible; i++) {
        ctx.lineTo(xFor(i), yFor(at(history.aimdo_usage, i)));
    }
    ctx.lineTo(xFor(visible - 1), h);
    ctx.closePath();
    ctx.fillStyle = hexToRgba(C.vram, 0.35);
    ctx.fill();

    // torch area stacked on top of aimdo
    ctx.beginPath();
    ctx.moveTo(dataStartX, yFor(at(history.aimdo_usage, 0)));
    for (let i = 0; i < visible; i++) {
        ctx.lineTo(xFor(i), yFor(at(history.aimdo_usage, i) + at(history.torch_active, i)));
    }
    for (let i = visible - 1; i >= 0; i--) {
        ctx.lineTo(xFor(i), yFor(at(history.aimdo_usage, i)));
    }
    ctx.closePath();
    ctx.fillStyle = hexToRgba(C.torch, 0.4);
    ctx.fill();

    // total used line
    ctx.beginPath();
    ctx.strokeStyle = C.totalLine;
    ctx.lineWidth = 1.5;
    for (let i = 0; i < visible; i++) {
        const x = xFor(i);
        const y = yFor(total - at(history.free_vram, i));
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

    // exec start/end markers — full-canvas time axis so partial-data fills don't bunch events at the right edge.
    if (history.execEvents.length) {
        const newest = historyGet(history.times, startIdx + visible - 1);
        const oldestVisible = historyGet(history.times, startIdx);
        const timePerStep = visible >= 2 ? (newest - oldestVisible) / (visible - 1) : HISTORY_TICK_MS;
        const rightEdgeX = (GRAPH_POINTS - 1) * stepX;
        if (timePerStep > 0) {
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 3]);
            for (const evt of history.execEvents) {
                const x = rightEdgeX - ((newest - evt.time) / timePerStep) * stepX;
                if (x < 0 || x > rightEdgeX + 0.5) continue;
                ctx.strokeStyle = evt.type === "start" ? C.torch : C.gpuUtilHi;
                ctx.beginPath();
                ctx.moveTo(x + 0.5, 0);
                ctx.lineTo(x + 0.5, h);
                ctx.stroke();
            }
            ctx.setLineDash([]);
        }
    }

    // gpu line uses its own 0..100 scale, not the VRAM byte scale
    if (gpuLineVisible) {
        ctx.beginPath();
        ctx.strokeStyle = C.gpuUtil;
        ctx.lineWidth = 1.25;
        for (let i = 0; i < visible; i++) {
            const y = h - (at(history.gpu_util, i) / 100) * h;
            if (i === 0) ctx.moveTo(xFor(i), y); else ctx.lineTo(xFor(i), y);
        }
        ctx.stroke();
    }

    if (graphHover.x != null && graphHover.idx != null) {
        const hi = graphHover.idx - startIdx;
        if (hi >= 0 && hi < visible) {
            ctx.strokeStyle = C.vram;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(graphHover.x + 0.5, 0);
            ctx.lineTo(graphHover.x + 0.5, h);
            ctx.stroke();

            // dots on the lines crossed by the hover line. Stroke around each dot
            // in the graph bg so it pops off the colored area fills behind it.
            const dot = (x, y, fill) => {
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, Math.PI * 2);
                ctx.fillStyle = fill;
                ctx.fill();
                ctx.lineWidth = 1.5;
                ctx.strokeStyle = C.graphBg;
                ctx.stroke();
            };
            const totalY = yFor(total - at(history.free_vram, hi));
            dot(graphHover.x, totalY, C.totalLine);
            if (gpuLineVisible) {
                const gpuY = h - (at(history.gpu_util, hi) / 100) * h;
                dot(graphHover.x, gpuY, C.gpuUtil);
            }
        }
    }
}

// hover-line state for the graph; null when cursor is outside the data range.
const graphHover = { x: null, idx: null };

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

// draw page grid to canvas — much faster than 700 DOM divs.
// vramColor (hex) tints static cells and the fade-in landing color per model.
function drawPageGrid(ctx, cssW, residency, changeAge, panelScale, vramColor) {
    const vramHex = vramColor || C.vram;
    const vramRgb = hexToRgb(vramHex);
    const fadeInFromRgb = lightenRgb(vramRgb, 0.55);
    const cellSize = 6;
    const gap = 1;
    const step = cellSize + gap;
    const cols = Math.max(1, Math.floor((cssW + gap) / step));
    const rows = Math.ceil(residency.length / cols);
    const cssH = rows * step;

    const dpr = window.devicePixelRatio || 1;
    const totalScale = panelScale * dpr;
    const canvas = ctx.canvas;
    const backingW = Math.max(1, Math.round(cssW * totalScale));
    const backingH = Math.max(1, Math.round((cssH || 1) * totalScale));

    // skip draw when nothing's animating and inputs match the previous call.
    let anyAnimating = false;
    for (let i = 0; i < changeAge.length; i++) if (changeAge[i] > 0) { anyAnimating = true; break; }
    const sig = `${vramHex}|var(--aimdo-unloaded)|${backingW}x${backingH}|${residency.length}`;
    if (!anyAnimating && canvas._lastSig === sig) return;
    canvas._lastSig = sig;

    if (canvas.width !== backingW) canvas.width = backingW;
    if (canvas.height !== backingH) canvas.height = backingH;
    canvas.style.height = (cssH || 1) + "px";
    ctx.setTransform(totalScale, 0, 0, totalScale, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH || 1);

    // batch: draw all static vram cells, then all static unloaded, then animated individually
    const animated = [];

    ctx.fillStyle = vramColor || C.vram;
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

    // fade-in: lightened type color → type color. fade-out: red → gray (universal "removed").
    for (const i of animated) {
        const resident = residency[i] & 1;
        const t = changeAge[i] / FADE_TICKS;
        const [fr, fg, fb] = resident ? fadeInFromRgb : C.fadeOutFrom;
        const [tr, tg, tb] = resident ? vramRgb : C.fadeOutTo;
        ctx.fillStyle = `rgb(${Math.round(fr * t + tr * (1 - t))},${Math.round(fg * t + tg * (1 - t))},${Math.round(fb * t + tb * (1 - t))})`;
        ctx.fillRect((i % cols) * step, Math.floor(i / cols) * step, cellSize, cellSize);
    }
}

function createPanel() {
    const saved = loadState();
    if (saved.pollInterval) pollInterval = saved.pollInterval;
    if (typeof saved.gpuLineVisible === "boolean") gpuLineVisible = saved.gpuLineVisible;
    if (typeof saved.colorModelBars === "boolean") colorModelBars = saved.colorModelBars;
    if (typeof saved.colorModelStroke === "boolean") colorModelStroke = saved.colorModelStroke;
    if (typeof saved.colorModelName === "boolean") colorModelName = saved.colorModelName;
    if (typeof saved.showLegends === "boolean") showLegends = saved.showLegends;
    if (typeof saved.showRamInMini === "boolean") showRamInMini = saved.showRamInMini;
    if (typeof saved.showVramInMini === "boolean") showVramInMini = saved.showVramInMini;
    if (typeof saved.showGpuInMini === "boolean") showGpuInMini = saved.showGpuInMini;
    if (typeof saved.showCpuInMini === "boolean") showCpuInMini = saved.showCpuInMini;
    if (typeof saved.showHwNames === "boolean") showHwNames = saved.showHwNames;
    if (typeof saved.showTitle === "boolean") showTitle = saved.showTitle;
    if (typeof saved.showExecBtn === "boolean") showExecBtn = saved.showExecBtn;
    if (typeof saved.miniShowNumbers === "boolean") miniShowNumbers = saved.miniShowNumbers;
    if (typeof saved.miniShowUnits === "boolean") miniShowUnits = saved.miniShowUnits;
    if (typeof saved.miniShowType === "boolean") miniShowType = saved.miniShowType;
    if (typeof saved.miniShowGpuTemp === "boolean") miniShowGpuTemp = saved.miniShowGpuTemp;
    if (typeof saved.miniShowGpuPower === "boolean") miniShowGpuPower = saved.miniShowGpuPower;
    if (typeof saved.graphHeight === "number" && saved.graphHeight > 0) graphHeight = saved.graphHeight;
    if (typeof saved.theme === "string" && THEME_NAMES.includes(saved.theme)) {
        currentTheme = saved.theme;
    }
    // always run — primes C from computed CSS so the canvas matches the stylesheet
    applyPalette(currentTheme);
    if (saved.modelCollapsed && typeof saved.modelCollapsed === "object") modelCollapsed = saved.modelCollapsed;
    let panelScale = typeof saved.scale === "number" ? saved.scale : 1;

    // structural styles live in aimdo_viz.css; CSS variables on :root carry the palette.
    // theme switches go through applyPalette → setCssVars; no per-element repaint here.
    const panel = document.createElement("div");
    panel.id = "aimdo-viz-panel";
    panel.style.zoom = panelScale;
    panel._scale = panelScale;
    if (saved.width != null) panel.style.width = Math.min(saved.width, window.innerWidth) / panelScale + "px";
    // explicit height only when expanded; collapsed shrinks to header.
    // separate height persistence per mode — expanding/collapsing loads the right one
    const initialHeight = saved.collapsed ? saved.heightCollapsed : saved.height;
    if (initialHeight != null) panel.style.height = Math.min(initialHeight, window.innerHeight) / panelScale + "px";

    // CSS sizes are pre-zoom (logical); divide visual targets by panelScale.
    // 50vh cap when auto-growing; relaxes to viewport when user sets explicit height.
    let pipWindow = null;  // set when the panel is moved into a PiP window
    const isPoppedOut = () => pipWindow && !pipWindow.closed;
    function applyConstraints() {
        if (isPoppedOut()) return;  // PiP fills its own window; main-page bounds don't apply
        const heightCapFrac = panel.style.height ? 1.0 : 0.5;
        panel.style.minWidth = (200 / panelScale) + "px";
        panel.style.maxWidth = (window.innerWidth / panelScale) + "px";
        panel.style.maxHeight = (window.innerHeight * heightCapFrac / panelScale) + "px";
    }
    applyConstraints();

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

    // ComfyUI's topbar / workflow tabs have higher z-index than us, so the
    // panel must clamp below them. Leaf bars (transparent bg) only block where
    // their items actually sit, letting the panel rise into empty regions.
    function getTopChromeBottom(panelLeft, panelRight) {
        const fullSels = [".comfyui-body-top", ".topbar-container", ".workflow-tabs-container", ".workflow-tabs"];
        const leafSels = [".actionbar-container"];
        let bottom = 0;
        for (const s of fullSels) {
            for (const el of document.querySelectorAll(s)) {
                const r = el.getBoundingClientRect();
                if (r.height > 0 && r.bottom > bottom) bottom = r.bottom;
            }
        }
        if (panelLeft == null) {
            for (const s of leafSels) {
                for (const el of document.querySelectorAll(s)) {
                    const r = el.getBoundingClientRect();
                    if (r.height > 0 && r.bottom > bottom) bottom = r.bottom;
                }
            }
            return bottom;
        }
        for (const s of leafSels) {
            for (const el of document.querySelectorAll(s)) {
                const r = el.getBoundingClientRect();
                if (r.height <= 0) continue;
                if (r.right <= panelLeft || r.left >= panelRight) continue;
                for (const node of el.querySelectorAll("*")) {
                    if (node.children.length > 0) continue;
                    const nr = node.getBoundingClientRect();
                    if (nr.width <= 0 || nr.height <= 0) continue;
                    if (nr.right <= panelLeft || nr.left >= panelRight) continue;
                    if (nr.bottom > bottom) bottom = nr.bottom;
                }
            }
        }
        return bottom;
    }

    // side toolbar can dock left or right; classify each by anchored edge.
    // opaque bars clamp at the container edge (not at leaf icons — would slip under padding).
    function isLeftAnchored(r) { return r.left < 8; }
    function isRightAnchored(r) { return window.innerWidth - r.right < 8; }
    function getSideChromeBounds(panelTop, panelBottom) {
        let minLeft = 0;
        let maxRight = window.innerWidth;
        for (const el of document.querySelectorAll(".side-toolbar-container")) {
            const r = el.getBoundingClientRect();
            if (r.width <= 0) continue;
            if (panelTop != null && (r.bottom <= panelTop || r.top >= panelBottom)) continue;
            if (isLeftAnchored(r)) {
                if (r.right > minLeft) minLeft = r.right;
            } else if (isRightAnchored(r)) {
                if (r.left < maxRight) maxRight = r.left;
            }
        }
        return { minLeft, maxRight };
    }

    function clampOffsets(ro, bo) {
        const b = getCanvasBounds();
        const pr = panel.getBoundingClientRect();
        const w = pr.width, h = pr.height;
        const vw = window.innerWidth, vh = window.innerHeight;
        const panelRight = b.right - ro;
        const panelLeft = panelRight - w;
        const panelTop = b.bottom - bo - h;
        const panelBottom = b.bottom - bo;
        const minTop = getTopChromeBottom(panelLeft, panelRight);
        const { minLeft, maxRight } = getSideChromeBounds(panelTop, panelBottom);

        // when window's too short for both constraints, prefer the chrome
        // clamp over the viewport edge so we never overlap the topbar/sidebar.
        const boHi = b.bottom - h - minTop;
        const boLo = b.bottom - vh;
        const boClamped = boHi < boLo ? boHi : Math.max(boLo, Math.min(bo, boHi));

        // roLo combines viewport-right and right-chrome constraints (whichever is tighter).
        const roHi = b.right - w - minLeft;
        const roLo = Math.max(b.right - vw, b.right - maxRight);
        const roClamped = roHi < roLo ? roHi : Math.max(roLo, Math.min(ro, roHi));
        return { ro: roClamped, bo: boClamped, b, w, h };
    }

    // visual-only clamp; closure offsets stay as user intent so they survive temporary shrinks.
    // CSS zoom scales style.left/top along with size, so we divide by panelScale to land at
    // the intended viewport position rather than position × scale.
    function applyOffsets() {
        if (pipWindow && !pipWindow.closed) return;  // PiP owns layout while popped out
        if (isDocked) return;                        // docked panel lives in the actionbar flex flow
        const { ro, bo, b, w, h } = clampOffsets(rightOffset, bottomOffset);
        // while actively dragging, bypass clamping so the user can fly over the topbar
        // to reach the dock drop zone. mouseup re-clamps before persisting.
        const useRo = dragging ? rightOffset : ro;
        const useBo = dragging ? bottomOffset : bo;
        panel.style.left = ((b.right - w - useRo) / panelScale) + "px";
        panel.style.top = ((b.bottom - h - useBo) / panelScale) + "px";
        panel.style.right = "auto";
        panel.style.bottom = "auto";
    }
    window.addEventListener("resize", () => { applyConstraints(); applyOffsets(); positionDockedBody(); });

    // --- Docking to ComfyUI's top actionbar ---
    // Same pattern as ComfyActionbar.vue: during a drag we expose a drop zone inside
    // .actionbar-container; mouseup-while-hovering reparents the panel into that container
    // and switches it to a flat horizontal mini-bar layout via the .aimdo-docked class.
    let isDocked = !!saved.docked;
    let dockSide = saved.dockSide === "left" ? "left" : "right";
    let dockExpanded = false;     // session-only: body shown as overlay below the docked mini-bar
    let autoDockPending = false;  // true while the post-load redock poll is running
    let dockSectionWidth = (typeof saved.dockSectionWidth === "number" && saved.dockSectionWidth > 40)
        ? saved.dockSectionWidth : 110;
    let savedPanelCss = null;     // snapshot before docking so undock can restore exact styles
    let dropZoneLeft = null;      // present only while a drag is in progress
    let dropZoneRight = null;
    let dropZoneHoverSide = null; // "left" | "right" | null
    function applyDockSectionWidth() {
        panel.style.setProperty("--aimdo-dock-section-w", dockSectionWidth + "px");
        positionDockedBody();
    }
    applyDockSectionWidth();

    // overlay anchored under the docked panel; right-docked panels extend the overlay leftward
    // so it doesn't get pushed off-screen when the panel sits at the topbar's right edge.
    function positionDockedBody() {
        if (!isDocked || !dockExpanded) return;
        const r = panel.getBoundingClientRect();
        const overlayW = 340;
        const anchorLeft = dockSide === "right"
            ? r.right - overlayW
            : r.left;
        const left = Math.max(4, Math.min(window.innerWidth - overlayW - 4, anchorLeft));
        body.style.top = (r.bottom + 4) + "px";
        body.style.left = left + "px";
    }

    function getActionbarContainer() {
        return document.querySelector(".actionbar-container");
    }

    // two zones — one before the existing actionbar items, one after — let the user choose
    // which side to dock on via CSS order on the panel.
    function makeDropZone(side) {
        const dz = document.createElement("div");
        dz.className = "aimdo-dropzone aimdo-dropzone-" + side;
        dz.textContent = "Dock " + side;
        dz.addEventListener("mouseenter", () => {
            if (!dragging) return;
            dropZoneHoverSide = side;
            dz.classList.add("is-hover");
        });
        dz.addEventListener("mouseleave", () => {
            if (dropZoneHoverSide === side) dropZoneHoverSide = null;
            dz.classList.remove("is-hover");
        });
        return dz;
    }
    function ensureDropZone() {
        const ac = getActionbarContainer();
        if (!ac) return null;
        if (dropZoneLeft && dropZoneLeft.parentNode === ac) return ac;
        if (dropZoneLeft) dropZoneLeft.remove();
        if (dropZoneRight) dropZoneRight.remove();
        dropZoneLeft = makeDropZone("left");
        dropZoneRight = makeDropZone("right");
        ac.appendChild(dropZoneLeft);
        ac.appendChild(dropZoneRight);
        return ac;
    }
    function clearDropZone() {
        dropZoneHoverSide = null;
        if (dropZoneLeft) { dropZoneLeft.remove(); dropZoneLeft = null; }
        if (dropZoneRight) { dropZoneRight.remove(); dropZoneRight = null; }
    }

    function dock(side) {
        const ac = getActionbarContainer();
        if (!ac) return false;
        if (side === "left" || side === "right") dockSide = side;
        // CSS order: -1 puts the panel before existing actionbar items (default order 0), +1 after
        if (isDocked) {
            panel.style.order = dockSide === "left" ? "-1" : "1";
            if (panel.parentNode !== ac) ac.appendChild(panel);
            saveState({ dockSide });
            return true;
        }
        // collapsed mode shows the mini-bar; CSS overrides to lay it out horizontally
        if (!collapsed) {
            collapsed = true;
            body.style.display = "none";
            miniBar.style.display = "block";
            toggleBtn.textContent = "+";
        }
        dockExpanded = false;
        // capture before adding order/class so undock restores clean floating styles
        savedPanelCss = panel.style.cssText;
        panel.classList.add("aimdo-docked");
        panel.style.order = dockSide === "left" ? "-1" : "1";
        ac.appendChild(panel);
        isDocked = true;
        saveState({ docked: true, dockSide });
        return true;
    }

    function undock() {
        autoDockPending = false;  // explicit undock cancels any in-flight auto-redock poll
        if (!isDocked) return;
        const wasExpanded = dockExpanded;
        dockExpanded = false;
        panel.classList.remove("aimdo-docked", "aimdo-docked-expanded");
        // clear the overlay positioning we set on the body so it returns to normal in-flow layout
        body.style.top = "";
        body.style.left = "";
        if (savedPanelCss != null) panel.style.cssText = savedPanelCss;
        savedPanelCss = null;
        document.body.appendChild(panel);
        isDocked = false;
        // if they were viewing the full body docked, keep it visible when re-floating
        if (wasExpanded) {
            collapsed = false;
            body.style.display = "flex";
            miniBar.style.display = "none";
            toggleBtn.textContent = "−";
            const s = loadState();
            const h = s.height;
            panel.style.height = h != null ? (Math.min(h, window.innerHeight) / panelScale + "px") : "";
            saveState({ collapsed });
        }
        saveState({ docked: false });
        applyConstraints();
        applyOffsets();
    }

    const header = document.createElement("div");
    header.className = "aimdo-header";
    // visible drag affordance matching ComfyUI's docked actionbar handle — six dots in a 2x3 grid
    const dragHandle = document.createElement("span");
    dragHandle.className = "aimdo-drag-handle";
    dragHandle.title = "Drag to move (or to dock at the top)";
    dragHandle.innerHTML = "<span></span>".repeat(6);
    header.appendChild(dragHandle);
    const titleSpan = document.createElement("span");
    titleSpan.className = "aimdo-title";
    titleSpan.textContent = "Memory";
    header.appendChild(titleSpan);

    const miniBar = document.createElement("div");
    miniBar.className = "aimdo-mini-bar";
    miniBar.innerHTML = `<div class="mini-ram-section">
        <div class="aimdo-mini-row">
            <span class="mini-ram-label">RAM</span><span class="mini-ram-usage"></span>
        </div>
        <div class="aimdo-mini-track mini-ram-bar"></div>
    </div>
    <div class="mini-vram-section">
        <div class="aimdo-mini-row">
            <span class="mini-vram-label">VRAM</span><span class="mini-vram-usage"></span>
        </div>
        <div class="aimdo-mini-track mini-vram-bar"></div>
    </div>
    <div class="mini-cpu-section">
        <div class="aimdo-mini-row">
            <span class="mini-cpu-label">CPU</span><span class="mini-cpu-usage"></span>
        </div>
        <div class="aimdo-mini-track mini-cpu-bar"><div class="aimdo-mini-fill mini-cpu-fill"></div></div>
    </div>
    <div class="mini-gpu-section">
        <div class="aimdo-mini-row mini-gpu-row">
            <span class="mini-gpu-label">GPU</span><span class="mini-gpu-header-value"></span>
        </div>
        <div class="aimdo-mini-inline mini-util-row">
            <div class="aimdo-mini-track mini-gpu-bar"><div class="aimdo-mini-fill mini-gpu-fill"></div></div>
            <span class="mini-gpu-usage"></span>
        </div>
        <div class="aimdo-mini-inline mini-temp-row">
            <div class="aimdo-mini-track mini-temp-bar"><div class="aimdo-mini-fill mini-temp-fill"></div></div>
            <span class="mini-temp-usage"></span>
        </div>
        <div class="aimdo-mini-inline mini-power-row">
            <div class="aimdo-mini-track mini-power-bar"><div class="aimdo-mini-fill mini-power-fill"></div></div>
            <span class="mini-power-usage"></span>
        </div>
    </div>`;

    const headerRight = document.createElement("div");
    headerRight.className = "aimdo-header-right";

    // optional play / cancel-running button. Toggles based on execState.running which
    // setup() keeps current via api event listeners; we also call updateExecBtnState
    // directly from those listeners so the button flips the instant execution starts/ends.
    const execBtn = document.createElement("span");
    execBtn.className = "aimdo-exec-btn";
    execBtn.style.display = showExecBtn ? "" : "none";
    function updateExecBtnState() {
        if (execState.running) {
            execBtn.classList.add("is-running");
            execBtn.textContent = "■";
            execBtn.title = "Cancel running workflow";
        } else {
            execBtn.classList.remove("is-running");
            execBtn.textContent = "▶";
            execBtn.title = "Run workflow";
        }
    }
    updateExecBtnState();
    execBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        try {
            if (execState.running) await api.interrupt(null);
            else await app.queuePrompt(0);
        } catch (err) {
            console.error("aimdo-viz: exec/interrupt failed", err);
        }
    });
    headerRight.appendChild(execBtn);

    const unloadBtn = document.createElement("span");
    unloadBtn.className = "aimdo-btn";
    unloadBtn.textContent = "unload ▾";
    unloadBtn.title = "Unload models / free cache (click for options)";

    const unloadMenu = document.createElement("div");
    unloadMenu.className = "aimdo-menu";
    unloadMenu.style.minWidth = "160px";
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
        item.className = "aimdo-menu-item";
        item.textContent = opt.label;
        item.title = opt.title;
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
        unloadMenu.style.zoom = panelScale;
        // style.left/top are pre-zoom on a zoomed element, divide visual targets by scale
        unloadMenu.style.left = (Math.max(4, r.right - 160 * panelScale) / panelScale) + "px";
        unloadMenu.style.top = ((r.bottom + 2) / panelScale) + "px";
        unloadMenu.style.display = "block";
    });
    document.addEventListener("click", (e) => {
        if (e.target !== unloadBtn && !unloadMenu.contains(e.target)) {
            unloadMenu.style.display = "none";
        }
    });

    function resetHistory() {
        peakVramUsed = 0;
        history.head = 0;
        history.len = 0;
        history.viewOffset = 0;
        history.followLive = true;
        history.torch_active.fill(0);
        history.aimdo_usage.fill(0);
        history.free_vram.fill(0);
        history.gpu_util.fill(0);
        history.times.fill(0);
        history.execEvents.length = 0;
        try { localStorage.removeItem(HISTORY_STORAGE_KEY); } catch {}
    }

    const popoutBtn = document.createElement("span");
    popoutBtn.className = "aimdo-btn-icon";
    popoutBtn.style.fontSize = "12px";
    popoutBtn.textContent = "\u2924";
    popoutBtn.title = "Pop out into a Picture-in-Picture window";
    popoutBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        if (pipWindow && !pipWindow.closed) { pipWindow.close(); return; }
        if (!window.documentPictureInPicture) {
            alert("Picture-in-Picture isn't supported here. Try Chrome or Edge.");
            return;
        }
        // PiP rewrites size to fill its window; .aimdo-docked's !important rules would fight it
        if (isDocked) undock();
        const r = panel.getBoundingClientRect();
        pipWindow = await window.documentPictureInPicture.requestWindow({
            width: Math.max(280, Math.round(r.width)),
            height: Math.max(200, Math.round(r.height)),
        });
        // mirror the stylesheet into the PiP window. Wait for the cloned <link> to finish
        // loading before continuing, otherwise the panel briefly renders unstyled while the
        // CSS file is being fetched in the PiP document.
        const styleSrc = document.getElementById("aimdo-viz-stylesheet");
        if (styleSrc) {
            const clone = styleSrc.cloneNode(true);
            await new Promise((resolve) => {
                clone.addEventListener("load", resolve, { once: true });
                clone.addEventListener("error", resolve, { once: true });  // proceed anyway after a 404
                pipWindow.document.head.appendChild(clone);
            });
        }
        // mirror the active theme onto PiP's <html> — the cloned stylesheet contains
        // every theme's overrides, the data attribute picks which block applies
        const themeAttr = document.documentElement.dataset.aimdoTheme;
        if (themeAttr) pipWindow.document.documentElement.dataset.aimdoTheme = themeAttr;
        // remember origins so we can restore on close
        const moved = [];
        const remember = el => moved.push({ el, parent: el.parentNode, next: el.nextSibling });
        remember(panel);
        for (const m of [rootMenu, unloadMenu, ...allSubmenus]) remember(m);
        const origPanelCss = panel.style.cssText;
        // fill the PiP window; drop the fixed positioning the main-page math expects
        Object.assign(panel.style, {
            position: "static", left: "auto", top: "auto", right: "auto", bottom: "auto",
            width: "100%", height: "100vh", maxWidth: "none", maxHeight: "none",
            border: "none", borderRadius: "0", boxShadow: "none",
        });
        pipWindow.document.body.style.margin = "0";
        pipWindow.document.body.style.background = C.bg;
        for (const { el } of moved) pipWindow.document.body.appendChild(el);
        pipWindow.addEventListener("pagehide", () => {
            panel.style.cssText = origPanelCss;
            for (const { el, parent, next } of moved) {
                if (!parent) continue;
                if (next && next.parentNode === parent) parent.insertBefore(el, next);
                else parent.appendChild(el);
            }
            pipWindow = null;
            applyOffsets();
        }, { once: true });
    });

    const toggleBtn = document.createElement("span");
    toggleBtn.className = "aimdo-btn-icon";
    toggleBtn.style.fontSize = "16px";
    toggleBtn.textContent = "\u2212";
    toggleBtn.title = "Collapse / expand panel";

    const body = document.createElement("div");
    body.id = "aimdo-viz-body";

    let collapsed = !!saved.collapsed;
    if (collapsed) {
        body.style.display = "none";
        toggleBtn.textContent = "+";
        miniBar.style.display = "block";
    }
    toggleBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        // docked: toggle the overlay body below the docked mini-bar; stays docked
        if (isDocked) {
            dockExpanded = !dockExpanded;
            if (dockExpanded) {
                panel.classList.add("aimdo-docked-expanded");
                positionDockedBody();
                toggleBtn.textContent = "\u2212";
            } else {
                panel.classList.remove("aimdo-docked-expanded");
                toggleBtn.textContent = "+";
            }
            return;
        }
        collapsed = !collapsed;
        body.style.display = collapsed ? "none" : "flex";
        miniBar.style.display = collapsed ? "block" : "none";
        toggleBtn.textContent = collapsed ? "+" : "\u2212";
        // each mode persists its own height so toggling restores the right size.
        const s = loadState();
        const h = collapsed ? s.heightCollapsed : s.height;
        panel.style.height = h != null ? (Math.min(h, window.innerHeight) / panelScale + "px") : "";
        applyConstraints();
        saveState({ collapsed });
    });

    headerRight.appendChild(unloadBtn);
    headerRight.appendChild(popoutBtn);
    headerRight.appendChild(toggleBtn);
    header.appendChild(headerRight);
    panel.appendChild(header);
    panel.appendChild(miniBar);
    panel.appendChild(body);

    let dragging = false, dx = 0, dy = 0;
    let dragSavedPointerEvents = null;
    // pendingDrag distinguishes "clicking a button in the header" from "grabbing to drag".
    // mousedown only arms; we wait for >DRAG_THRESHOLD px of movement before promoting to a real drag,
    // so a click on unload/reset/popout never triggers an undock.
    let pendingDrag = null;
    const DRAG_THRESHOLD = 5;

    function promoteDrag(e) {
        if (pendingDrag.dockedAtStart && isDocked) {
            // jump the now-floating panel under the cursor with a reasonable grab offset
            undock();
            const r = panel.getBoundingClientRect();
            dx = Math.min(r.width - 20, Math.max(20, 40));
            dy = Math.min(r.height - 8, Math.max(8, 14));
            const b = getCanvasBounds();
            rightOffset = b.right - (e.clientX - dx) - r.width;
            bottomOffset = b.bottom - (e.clientY - dy) - r.height;
            applyOffsets();
        }
        // for non-docked starts, dx/dy were captured at mousedown
        dragging = true;
        pendingDrag = null;
        dragSavedPointerEvents = panel.style.pointerEvents || "";
        panel.style.pointerEvents = "none";
        ensureDropZone();
    }

    header.addEventListener("mousedown", (e) => {
        if (e.button !== 0) return;
        // when docked, only the drag handle starts a drag — the title and buttons
        // around it must stay plain-clickable. floating mode keeps the whole header.
        if (isDocked && !dragHandle.contains(e.target)) return;
        autoDockPending = false;  // user intent overrides the pending auto-redock
        pendingDrag = { startX: e.clientX, startY: e.clientY, dockedAtStart: isDocked };
        if (!isDocked) {
            const r = panel.getBoundingClientRect();
            dx = e.clientX - r.left;
            dy = e.clientY - r.top;
        }
    });
    // Ctrl/Cmd + mousedown anywhere on the panel starts a drag — capture phase
    // so we intercept before child elements (buttons, edge handles) react
    const isModifier = e => (e.ctrlKey || e.metaKey) && e.button === 0;
    const updateCursor = (e) => { panel.style.cursor = (e.ctrlKey || e.metaKey) ? "move" : ""; };
    document.addEventListener("keydown", updateCursor);
    document.addEventListener("keyup", updateCursor);
    window.addEventListener("blur", () => { panel.style.cursor = ""; });
    panel.addEventListener("mousedown", (e) => {
        if (!isModifier(e) || dragging || pendingDrag) return;
        e.preventDefault();
        e.stopPropagation();
        pendingDrag = { startX: e.clientX, startY: e.clientY, dockedAtStart: isDocked };
        if (!isDocked) {
            const r = panel.getBoundingClientRect();
            dx = e.clientX - r.left;
            dy = e.clientY - r.top;
        }
    }, true);
    // suppress the click that follows a Ctrl+drag so a Ctrl+click on a button
    // doesn't trigger its action after the drag ends
    panel.addEventListener("click", (e) => {
        if (isModifier(e)) {
            e.preventDefault();
            e.stopPropagation();
        }
    }, true);
    document.addEventListener("mousemove", (e) => {
        if (pendingDrag) {
            const adx = Math.abs(e.clientX - pendingDrag.startX);
            const ady = Math.abs(e.clientY - pendingDrag.startY);
            if (Math.max(adx, ady) < DRAG_THRESHOLD) return;
            promoteDrag(e);
        }
        if (!dragging) return;
        const b = getCanvasBounds();
        const r = panel.getBoundingClientRect();
        rightOffset = b.right - (e.clientX - dx) - r.width;
        bottomOffset = b.bottom - (e.clientY - dy) - r.height;
        applyOffsets();
    });
    // shared drag-end cleanup so a mouseup off-window (or alt-tab during drag) doesn't
    // leave pendingDrag armed, pointer-events stuck at "none", or drop zones in the DOM.
    function endDrag(commit) {
        pendingDrag = null;
        if (!dragging) {
            // pendingDrag-only path (click without movement). No drop zones to clear,
            // but call clearDropZone defensively in case a stale one slipped through.
            clearDropZone();
            return;
        }
        const wantDockSide = commit ? dropZoneHoverSide : null;
        clearDropZone();
        if (dragSavedPointerEvents !== null) {
            panel.style.pointerEvents = dragSavedPointerEvents;
            dragSavedPointerEvents = null;
        }
        dragging = false;  // before applyOffsets so it uses the clamped position
        if (commit) {
            if (wantDockSide) {
                dock(wantDockSide);
            } else {
                const c = clampOffsets(rightOffset, bottomOffset);
                rightOffset = c.ro;
                bottomOffset = c.bo;
                applyOffsets();
                saveState({ rightOffset, bottomOffset });
            }
        } else {
            // abort (blur / alt-tab): snap back into bounds visually without persisting
            applyOffsets();
        }
    }
    document.addEventListener("mouseup", () => endDrag(true));
    window.addEventListener("blur", () => endDrag(false));

    // edge handles: left grows left (right edge anchored), right grows right (left
    // edge anchored via RO ro-delta), bottom grows down (top edge anchored via bo-delta).
    let suppressWidthAnchor = false;
    let edgeDrag = null;
    function makeEdgeHandle(side) {
        const h = document.createElement("div");
        h.className = `aimdo-edge-handle aimdo-edge-${side}`;
        h.title = "Drag to resize";
        h.addEventListener("mousedown", (e) => {
            if (e.button !== 0) return;
            e.preventDefault();
            e.stopPropagation();
            // captured at drag start so the RO ro-shift mid-drag can't move the cap.
            const r = panel.getBoundingClientRect();
            const { minLeft, maxRight } = getSideChromeBounds(r.top, r.bottom);
            const maxWidth = side === "right" ? maxRight - r.left : r.right - minLeft;
            edgeDrag = { side, startX: e.clientX, startWidth: r.width, maxWidth };
            if (side === "left") suppressWidthAnchor = true;
        });
        panel.appendChild(h);
    }
    makeEdgeHandle("left");
    makeEdgeHandle("right");

    // collapsed: can't shrink below header + miniBar. expanded: keep modelsDiv top visible.
    function computeMinHeight(panelRect) {
        if (collapsed) return miniBar.getBoundingClientRect().bottom - panelRect.top + 2;
        if (refs && refs.modelsDiv) {
            return Math.max(40, refs.modelsDiv.getBoundingClientRect().top - panelRect.top + 8);
        }
        return 80;
    }

    let bottomDrag = null;
    function makeBottomHandle() {
        const h = document.createElement("div");
        // inset from the side handles so corners go to the ew-resize handles
        h.className = "aimdo-edge-handle aimdo-edge-bottom";
        h.title = "Drag to resize";
        h.addEventListener("mousedown", (e) => {
            if (e.button !== 0) return;
            e.preventDefault();
            e.stopPropagation();
            const r = panel.getBoundingClientRect();
            const minHeight = computeMinHeight(r);
            const maxHeight = window.innerHeight - r.top;
            bottomDrag = { startY: e.clientY, startHeight: r.height, minHeight, maxHeight };
        });
        panel.appendChild(h);
    }
    makeBottomHandle();

    // bottom-right corner handle: diagonal width+height drag. The grip gradient + hover
    // opacity live in aimdo_viz.css now.
    let cornerDrag = null;
    function makeCornerHandle() {
        const h = document.createElement("div");
        h.className = "aimdo-corner-handle";
        h.title = "Drag to resize";
        h.addEventListener("mousedown", (e) => {
            if (e.button !== 0) return;
            e.preventDefault();
            e.stopPropagation();
            const r = panel.getBoundingClientRect();
            const { minLeft, maxRight } = getSideChromeBounds(r.top, r.bottom);
            cornerDrag = {
                startX: e.clientX, startY: e.clientY,
                startWidth: r.width, startHeight: r.height,
                maxWidth: maxRight - r.left,
                maxHeight: window.innerHeight - r.top,
                minHeight: computeMinHeight(r),
            };
        });
        panel.appendChild(h);
    }
    makeCornerHandle();

    document.addEventListener("mousemove", (e) => {
        if (!edgeDrag) return;
        const delta = e.clientX - edgeDrag.startX;
        const newWidth = edgeDrag.side === "left" ? edgeDrag.startWidth - delta : edgeDrag.startWidth + delta;
        panel.style.width = Math.max(200, Math.min(edgeDrag.maxWidth, newWidth)) / panelScale + "px";
    });
    document.addEventListener("mousemove", (e) => {
        if (!bottomDrag) return;
        const delta = e.clientY - bottomDrag.startY;
        const newHeight = Math.max(bottomDrag.minHeight, Math.min(bottomDrag.maxHeight, bottomDrag.startHeight + delta));
        panel.style.height = (newHeight / panelScale) + "px";
        // relax the 50vh auto-grow cap once user sets an explicit height.
        applyConstraints();
    });
    document.addEventListener("mousemove", (e) => {
        if (!cornerDrag) return;
        const dx = e.clientX - cornerDrag.startX;
        const dy = e.clientY - cornerDrag.startY;
        const newWidth = Math.max(200, Math.min(cornerDrag.maxWidth, cornerDrag.startWidth + dx));
        const newHeight = Math.max(cornerDrag.minHeight, Math.min(cornerDrag.maxHeight, cornerDrag.startHeight + dy));
        panel.style.width = (newWidth / panelScale) + "px";
        panel.style.height = (newHeight / panelScale) + "px";
        applyConstraints();
    });
    document.addEventListener("mouseup", () => {
        if (edgeDrag) {
            edgeDrag = null;
            suppressWidthAnchor = false;
        }
        if (bottomDrag) bottomDrag = null;
        if (cornerDrag) cornerDrag = null;
    });

    // menu chrome lives in aimdo_viz.css (.aimdo-menu). Submenus differ only by
    // position which we set inline at openSubmenu time.
    function makeMenu() {
        const m = document.createElement("div");
        m.className = "aimdo-menu";
        return m;
    }
    const rootMenu = makeMenu();
    const scaleSubmenu = makeMenu();
    const pollSubmenu = makeMenu();
    const displaySubmenu = makeMenu();
    const miniSubmenu = makeMenu();
    const themeSubmenu = makeMenu();
    const dockWidthSubmenu = makeMenu();
    const gpuSubmenu = makeMenu();
    const allSubmenus = [scaleSubmenu, pollSubmenu, displaySubmenu, miniSubmenu, themeSubmenu, dockWidthSubmenu, gpuSubmenu];
    function closeAllSubmenus() { for (const m of allSubmenus) m.style.display = "none"; }

    // submenu overlaps parent by 1px so mouse transit doesn't trigger mouseleave-close.
    // anchorMenu defaults to rootMenu (top-level submenus) but can be another submenu
    // for nested cases; keepOpen lists ancestors that must NOT be closed when this opens.
    function openSubmenu(parentItem, submenu, anchorMenu, keepOpen) {
        anchorMenu = anchorMenu || rootMenu;
        keepOpen = keepOpen || [];
        for (const m of allSubmenus) {
            if (m === submenu || keepOpen.includes(m)) continue;
            m.style.display = "none";
        }
        submenu.style.zoom = panelScale;
        submenu.style.display = "block";
        const anchorR = anchorMenu.getBoundingClientRect();
        const itemR = parentItem.getBoundingClientRect();
        const subR = submenu.getBoundingClientRect();
        let left = anchorR.right - 1;
        if (left + subR.width > window.innerWidth) left = Math.max(2, anchorR.left - subR.width + 1);
        submenu.style.left = (left / panelScale) + "px";
        submenu.style.top = (Math.min(itemR.top, window.innerHeight - subR.height - 4) / panelScale) + "px";
    }

    // factory for a checkbox-style toggle item bound to a module-level flag.
    function makeToggleItem(label, getValue, setValue, stateKey) {
        const item = document.createElement("div");
        item.className = "aimdo-menu-item";
        const render = () => {
            item.innerHTML = `<span class="aimdo-check">${getValue() ? "✓" : ""}</span>${label}`;
        };
        render();
        item.addEventListener("click", (e) => {
            e.stopPropagation();
            if (item.classList.contains("is-disabled")) return;
            setValue(!getValue());
            saveState({ [stateKey]: getValue() });
            render();
            // visible change kicks in on the next poll tick (<= pollInterval ms).
        });
        return { item, render };
    }

    // parent item that opens a submenu on hover; chevron hints at the nesting.
    function makeSubmenuParent(label, submenu, anchorMenu, keepOpen) {
        const item = document.createElement("div");
        item.className = "aimdo-menu-parent";
        item.innerHTML = `<span>${label}</span><span class="aimdo-chevron">▸</span>`;
        item.addEventListener("mouseenter", () => openSubmenu(item, submenu, anchorMenu, keepOpen));
        return item;
    }

    // --- Scale submenu
    const scalePresets = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2];
    const scaleItems = new Map();
    function renderScaleItems() {
        for (const [s, item] of scaleItems) {
            const on = Math.abs(s - panelScale) < 1e-6;
            item.innerHTML = `<span class="aimdo-check">${on ? "✓" : ""}</span>${Math.round(s * 100)}%`;
        }
    }
    function setScale(s) {
        const r = panel.getBoundingClientRect();
        const w = r.width, h = r.height;
        const hadExplicitHeight = panel.style.height !== "";
        panelScale = s;
        panel._scale = s;
        panel.style.zoom = s;
        panel.style.width = (w / s) + "px";
        if (hadExplicitHeight) panel.style.height = (h / s) + "px";
        applyConstraints();
        applyOffsets();
        saveState({ scale: s });
    }
    for (const s of scalePresets) {
        const item = document.createElement("div");
        item.className = "aimdo-menu-item";
        item.addEventListener("click", (e) => {
            e.stopPropagation();
            setScale(s);
            renderScaleItems();
            rootMenu.style.display = "none";
            closeAllSubmenus();
        });
        scaleSubmenu.appendChild(item);
        scaleItems.set(s, item);
    }
    renderScaleItems();

    // --- Display submenu
    const colorBars = makeToggleItem("Color model bars",
        () => colorModelBars, v => { colorModelBars = v; }, "colorModelBars");
    const colorStroke = makeToggleItem("Color model stroke",
        () => colorModelStroke, v => { colorModelStroke = v; }, "colorModelStroke");
    const colorName = makeToggleItem("Color model name",
        () => colorModelName, v => { colorModelName = v; }, "colorModelName");
    const showLeg = makeToggleItem("Show legends",
        () => showLegends, v => { showLegends = v; }, "showLegends");
    const showTitleItem = makeToggleItem("Show title",
        () => showTitle, v => { showTitle = v; }, "showTitle");
    const showExecBtnItem = makeToggleItem("Execute button",
        () => showExecBtn,
        v => { showExecBtn = v; execBtn.style.display = v ? "" : "none"; },
        "showExecBtn");
    displaySubmenu.appendChild(colorBars.item);
    displaySubmenu.appendChild(colorStroke.item);
    displaySubmenu.appendChild(colorName.item);
    displaySubmenu.appendChild(showLeg.item);
    displaySubmenu.appendChild(showTitleItem.item);
    displaySubmenu.appendChild(showExecBtnItem.item);

    // --- Mini view submenu
    const showRam = makeToggleItem("RAM",
        () => showRamInMini, v => { showRamInMini = v; }, "showRamInMini");
    const showVram = makeToggleItem("VRAM",
        () => showVramInMini, v => { showVramInMini = v; }, "showVramInMini");
    const showCpu = makeToggleItem("CPU",
        () => showCpuInMini, v => { showCpuInMini = v; }, "showCpuInMini");
    // labeled "util" since these live under the nested GPU submenu now
    const showGpu = makeToggleItem("util",
        () => showGpuInMini, v => { showGpuInMini = v; }, "showGpuInMini");
    const showNames = makeToggleItem("Device names",
        () => showHwNames, v => { showHwNames = v; }, "showHwNames");
    const showNumbers = makeToggleItem("Numbers",
        () => miniShowNumbers, v => { miniShowNumbers = v; }, "miniShowNumbers");
    const showUnits = makeToggleItem("Units",
        () => miniShowUnits, v => { miniShowUnits = v; }, "miniShowUnits");
    const showType = makeToggleItem("Type labels",
        () => miniShowType, v => { miniShowType = v; }, "miniShowType");
    const showGpuTemp = makeToggleItem("temp",
        () => miniShowGpuTemp, v => { miniShowGpuTemp = v; }, "miniShowGpuTemp");
    const showGpuPower = makeToggleItem("power",
        () => miniShowGpuPower, v => { miniShowGpuPower = v; }, "miniShowGpuPower");
    miniSubmenu.appendChild(showRam.item);
    miniSubmenu.appendChild(showVram.item);
    miniSubmenu.appendChild(showCpu.item);
    // GPU's util / temp / power get their own submenu since they're closely related —
    // keeps the Mini-view list flat and groups the three multibar toggles together.
    // Each is independent: any combination can be on/off, including just temp+power.
    gpuSubmenu.appendChild(showGpu.item);
    gpuSubmenu.appendChild(showGpuTemp.item);
    gpuSubmenu.appendChild(showGpuPower.item);
    miniSubmenu.appendChild(makeSubmenuParent("GPU", gpuSubmenu, miniSubmenu, [miniSubmenu]));
    miniSubmenu.appendChild(showNames.item);
    miniSubmenu.appendChild(showType.item);
    miniSubmenu.appendChild(showNumbers.item);
    miniSubmenu.appendChild(showUnits.item);

    // --- Polling interval submenu (single-select like Scale)
    const pollPresets = [100, 250, 500, 1000, 2000, 5000];
    const pollItems = new Map();
    function renderPollItems() {
        for (const [ms, item] of pollItems) {
            const on = ms === pollInterval;
            const label = ms < 1000 ? `${ms} ms` : `${ms / 1000} s`;
            item.innerHTML = `<span class="aimdo-check">${on ? "✓" : ""}</span>${label}`;
        }
    }
    for (const ms of pollPresets) {
        const item = document.createElement("div");
        item.className = "aimdo-menu-item";
        item.addEventListener("click", (e) => {
            e.stopPropagation();
            pollInterval = ms;
            saveState({ pollInterval });
            renderPollItems();
            rootMenu.style.display = "none";
            closeAllSubmenus();
        });
        pollSubmenu.appendChild(item);
        pollItems.set(ms, item);
    }
    renderPollItems();

    // --- Theme submenu (single-select; live-applies)
    // applyPalette pushes the active palette into CSS variables on :root, so
    // most chrome repaints itself. We only need to manually clear places that
    // bake colors into innerHTML / canvas at render time and pick the new
    // palette up on their next tick.
    function applyTheme(name) {
        if (!THEME_NAMES.includes(name)) return;
        currentTheme = name;
        applyPalette(name);
        if (refs && refs.bottomLegend) {
            refs.bottomLegend.remove();
            refs.bottomLegend = null;
        }
    }
    const themeItems = new Map();
    function renderThemeItems() {
        for (const [name, item] of themeItems) {
            const on = name === currentTheme;
            const label = name[0].toUpperCase() + name.slice(1);
            item.innerHTML = `<span class="aimdo-check">${on ? "✓" : ""}</span>${label}`;
        }
    }
    for (const name of THEME_NAMES) {
        const item = document.createElement("div");
        item.className = "aimdo-menu-item";
        item.addEventListener("click", (e) => {
            e.stopPropagation();
            applyTheme(name);
            saveState({ theme: name });
            renderThemeItems();
            rootMenu.style.display = "none";
            closeAllSubmenus();
        });
        themeSubmenu.appendChild(item);
        themeItems.set(name, item);
    }
    renderThemeItems();

    // --- Dock width submenu (section width when docked into the topbar)
    const dockWidthSliderRow = document.createElement("div");
    dockWidthSliderRow.style.cssText = `padding:6px 10px;display:flex;flex-direction:column;gap:4px;min-width:180px;`;
    dockWidthSliderRow.addEventListener("click", (e) => e.stopPropagation());
    const dockWidthLabel = document.createElement("div");
    dockWidthLabel.style.cssText = `display:flex;justify-content:space-between;font-size:10px;color:var(--aimdo-textDim);`;
    dockWidthLabel.innerHTML = `<span>Section width</span><span class="aimdo-dw-val">${dockSectionWidth}px</span>`;
    const dockWidthSlider = document.createElement("input");
    dockWidthSlider.type = "range";
    dockWidthSlider.min = "60";
    dockWidthSlider.max = "400";
    dockWidthSlider.step = "5";
    dockWidthSlider.value = String(dockSectionWidth);
    dockWidthSlider.style.cssText = `width:100%;accent-color:var(--aimdo-vram);cursor:pointer;`;
    const dockWidthValSpan = dockWidthLabel.querySelector(".aimdo-dw-val");
    dockWidthSlider.addEventListener("input", () => {
        dockSectionWidth = parseInt(dockWidthSlider.value, 10);
        dockWidthValSpan.textContent = dockSectionWidth + "px";
        applyDockSectionWidth();
        saveState({ dockSectionWidth });
    });
    dockWidthSliderRow.appendChild(dockWidthLabel);
    dockWidthSliderRow.appendChild(dockWidthSlider);
    dockWidthSubmenu.appendChild(dockWidthSliderRow);
    function renderDockWidthItems() {
        dockWidthSlider.value = String(dockSectionWidth);
        dockWidthValSpan.textContent = dockSectionWidth + "px";
    }

    // --- Root menu items
    rootMenu.appendChild(makeSubmenuParent("Scale", scaleSubmenu));
    rootMenu.appendChild(makeSubmenuParent("Polling interval", pollSubmenu));
    rootMenu.appendChild(makeSubmenuParent("Display", displaySubmenu));
    rootMenu.appendChild(makeSubmenuParent("Mini view", miniSubmenu));
    rootMenu.appendChild(makeSubmenuParent("Theme", themeSubmenu));
    rootMenu.appendChild(makeSubmenuParent("Dock width", dockWidthSubmenu));

    // dock / undock toggle — present only when the actionbar is available so we don't
    // offer a no-op when ComfyUI's new menu is disabled.
    const dockItem = document.createElement("div");
    dockItem.className = "aimdo-menu-item";
    function renderDockItem() {
        const canDock = !!getActionbarContainer();
        dockItem.style.display = (isDocked || canDock) ? "" : "none";
        dockItem.innerHTML = `<span class="aimdo-check">${isDocked ? "✓" : ""}</span>${isDocked ? "Undock to floating" : "Dock to top"}`;
    }
    dockItem.addEventListener("click", (e) => {
        e.stopPropagation();
        if (isDocked) undock(); else dock();
        renderDockItem();
        rootMenu.style.display = "none";
        closeAllSubmenus();
    });
    renderDockItem();
    rootMenu.appendChild(dockItem);

    // reset peak VRAM marker + clear history graph; this used to live on the header
    const resetItem = document.createElement("div");
    resetItem.className = "aimdo-menu-item";
    resetItem.innerHTML = `<span class="aimdo-check"></span>Reset history`;
    resetItem.title = "Reset peak VRAM marker and clear history graph";
    resetItem.addEventListener("click", (e) => {
        e.stopPropagation();
        resetHistory();
        rootMenu.style.display = "none";
        closeAllSubmenus();
    });
    rootMenu.appendChild(resetItem);

    function renderColorBarsItem() {
        renderScaleItems(); renderPollItems(); renderThemeItems(); renderDockWidthItems();
        colorBars.render(); colorStroke.render(); colorName.render(); showLeg.render();
        showRam.render(); showVram.render(); showCpu.render(); showGpu.render(); showNames.render();
        showTitleItem.render(); showExecBtnItem.render();
        showType.render(); showNumbers.render(); showUnits.render();
        showGpuTemp.render(); showGpuPower.render();
        renderDockItem();
    }

    document.body.appendChild(rootMenu);
    for (const m of allSubmenus) document.body.appendChild(m);

    panel.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        e.stopPropagation();
        renderColorBarsItem();
        closeAllSubmenus();
        rootMenu.style.zoom = panelScale;
        rootMenu.style.display = "block";
        const mRect = rootMenu.getBoundingClientRect();
        const mw = mRect.width || 120 * panelScale;
        const mh = mRect.height || 120 * panelScale;
        rootMenu.style.left = (Math.min(e.clientX, window.innerWidth - mw - 4) / panelScale) + "px";
        rootMenu.style.top = (Math.min(e.clientY, window.innerHeight - mh - 4) / panelScale) + "px";
    });
    document.addEventListener("click", (e) => {
        if (rootMenu.contains(e.target)) return;
        for (const m of allSubmenus) if (m.contains(e.target)) return;
        rootMenu.style.display = "none";
        closeAllSubmenus();
    });
    // close the docked-expanded body overlay on outside click. body is a DOM child of panel
    // so panel.contains catches clicks inside the overlay; the popped-out menus live in
    // document.body, so we whitelist them too — otherwise picking a context-menu item
    // would also dismiss the overlay the menu was launched from.
    document.addEventListener("click", (e) => {
        if (!isDocked || !dockExpanded) return;
        if (panel.contains(e.target)) return;
        if (rootMenu.contains(e.target)) return;
        if (unloadMenu.contains(e.target)) return;
        for (const m of allSubmenus) if (m.contains(e.target)) return;
        dockExpanded = false;
        panel.classList.remove("aimdo-docked-expanded");
        toggleBtn.textContent = "+";
    });
    // moving the mouse out of the menu structure into empty space should also
    // close the submenu so it doesn't linger over unrelated UI.
    rootMenu.addEventListener("mouseleave", (e) => {
        const target = e.relatedTarget;
        if (target && (target instanceof Node) && allSubmenus.some(m => m.contains(target))) return;
        closeAllSubmenus();
    });
    for (const m of allSubmenus) {
        m.addEventListener("mouseleave", (e) => {
            const target = e.relatedTarget;
            if (target && (target instanceof Node) && (rootMenu.contains(target) || allSubmenus.some(x => x.contains(target)))) return;
            m.style.display = "none";
        });
    }

    document.body.appendChild(panel);
    applyOffsets();

    // restore docked placement once ComfyUI has built the topbar. Until the container
    // exists we wait — the panel sits floating in its persisted spot in the meantime.
    // Capped at ~3s so a user who disabled the new menu doesn't keep us polling.
    // Any user-initiated drag or explicit undock during the wait cancels the poll so
    // we don't yank the panel out from under them.
    if (isDocked) {
        isDocked = false;  // dock() guards on this; reset so it actually runs
        autoDockPending = true;
        let frames = 0;
        const tryDock = () => {
            if (!autoDockPending) return;
            if (dock(dockSide)) return;
            if (++frames > 180) { autoDockPending = false; saveState({ docked: false }); return; }
            requestAnimationFrame(tryDock);
        };
        requestAnimationFrame(tryDock);
    }

    // width changes shift ro to anchor one edge (unless suppressed by the left handle).
    // height changes always anchor the top edge by shifting bo by -Δh.
    let lastPanelWidth = null;
    let lastPanelHeight = null;
    if (typeof ResizeObserver !== "undefined") {
        new ResizeObserver(() => {
            if (isPoppedOut()) return;  // PiP resize doesn't persist back to main-page state
            if (isDocked) { lastPanelWidth = null; lastPanelHeight = null; positionDockedBody(); return; }
            const r = panel.getBoundingClientRect();
            const w = r.width, h = r.height;
            if (lastPanelWidth !== null && w !== lastPanelWidth) {
                if (!suppressWidthAnchor) rightOffset -= (w - lastPanelWidth);
                saveState({ width: w, rightOffset, bottomOffset });
            }
            if (lastPanelHeight !== null && h !== lastPanelHeight) {
                bottomOffset -= (h - lastPanelHeight);
                // persist height only on explicit drag, and key it by current mode so
                // expanded vs collapsed heights don't overwrite each other.
                if (bottomDrag || cornerDrag) {
                    const key = collapsed ? "heightCollapsed" : "height";
                    saveState({ [key]: h, bottomOffset });
                } else saveState({ bottomOffset });
            }
            lastPanelWidth = w;
            lastPanelHeight = h;
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
            new ResizeObserver(() => { applyOffsets(); positionDockedBody(); }).observe(el);
        }
        applyOffsets();
    }
    attachCanvasObserver();
    body._titleSpan = titleSpan;
    body._miniBar = miniBar;
    body._panel = panel;
    body._updateExecBtnState = updateExecBtnState;
    return body;
}

// persistent DOM refs to avoid re-querying / re-creating
let refs = null;

function ensureStructure(body) {
    if (refs) return refs;

    body.innerHTML = "";

    const contentDiv = document.createElement("div");
    contentDiv.id = "aimdo-content";
    contentDiv.style.cssText = "flex-shrink: 0;";
    contentDiv.addEventListener("click", (e) => {
        const t = e.target.closest(".aimdo-gpu-util");
        if (!t) return;
        gpuLineVisible = !gpuLineVisible;
        saveState({ gpuLineVisible });
        if (refs) drawGraph(refs.graphCtx, refs.graphCanvas.width, refs.graphCanvas.height);
        t.style.opacity = gpuLineVisible ? "1" : "0.4";
    });
    body.appendChild(contentDiv);

    const graphHeader = document.createElement("div");
    graphHeader.style.cssText = `display:flex;justify-content:space-between;font-size:9px;color:var(--aimdo-textDim);margin-bottom:2px;flex-shrink:0;`;
    graphHeader.innerHTML = `<span class="graph-time-left"></span><span class="graph-hover-info"></span><span class="graph-time-right"></span>`;
    body.appendChild(graphHeader);

    const graphCanvas = document.createElement("canvas");
    graphCanvas.width = 300;
    graphCanvas.height = graphHeight;
    graphCanvas.style.cssText = `width:100%;height:${graphHeight}px;border-radius:3px;background:var(--aimdo-graphBg);flex-shrink:0;cursor:crosshair;`;
    body.appendChild(graphCanvas);

    // redraw without waiting for the next poll — used by the scrub-drag handler.
    const redrawGraph = () => {
        if (!refs || !refs.graphCtx) return;
        const panelScale = (body._panel && body._panel._scale) || 1;
        const gRect = graphCanvas.getBoundingClientRect();
        if (gRect.width > 0 && gRect.height > 0) {
            drawGraph(refs.graphCtx, gRect.width / panelScale, gRect.height / panelScale);
        }
        updateGraphTimes();
    };

    graphCanvas.addEventListener("mousemove", (e) => {
        if (scrubDrag) return;  // hover line during drag is distracting
        const rect = graphCanvas.getBoundingClientRect();
        const scale = (body._panel && body._panel._scale) || 1;
        const x = (e.clientX - rect.left) / scale;
        const w = rect.width / scale;
        const stepX = w / (GRAPH_POINTS - 1);
        const slotIdx = Math.round(x / stepX);
        const len = history.len;
        const visible = Math.min(GRAPH_POINTS, len - history.viewOffset);
        const visibleIdx = slotIdx - (GRAPH_POINTS - visible);
        if (visible < 1 || visibleIdx < 0 || visibleIdx >= visible) {
            graphHover.x = null;
            graphHover.idx = null;
        } else {
            const startIdx = len - visible - history.viewOffset;
            graphHover.idx = startIdx + visibleIdx;
            graphHover.x = slotIdx * stepX;
        }
        redrawGraph();
    });
    graphCanvas.addEventListener("mouseleave", () => {
        graphHover.x = null;
        graphHover.idx = null;
        redrawGraph();
    });

    // drag scrubbing: dragging the full visible window's worth scrolls by GRAPH_POINTS.
    // Drag right pulls older points into view; releasing at viewOffset 0 re-enables
    // follow-live so new data slides into the window automatically.
    let scrubDrag = null;
    graphCanvas.addEventListener("mousedown", (e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        const visualScale = (body._panel && body._panel._scale) || 1;
        scrubDrag = { startX: e.clientX, startOffset: history.viewOffset, scale: visualScale };
        history.followLive = false;
        graphCanvas.style.cursor = "grabbing";
    });
    document.addEventListener("mousemove", (e) => {
        if (!scrubDrag) return;
        const rect = graphCanvas.getBoundingClientRect();
        const dxPx = (e.clientX - scrubDrag.startX) / scrubDrag.scale;
        const ptsPerPx = GRAPH_POINTS / Math.max(1, rect.width / scrubDrag.scale);
        const maxOffset = Math.max(0, history.len - GRAPH_POINTS);
        history.viewOffset = Math.max(0, Math.min(maxOffset, Math.round(scrubDrag.startOffset + dxPx * ptsPerPx)));
        redrawGraph();
    });
    document.addEventListener("mouseup", () => {
        if (!scrubDrag) return;
        scrubDrag = null;
        graphCanvas.style.cursor = "grab";
        if (history.viewOffset <= 1) {
            history.viewOffset = 0;
            history.followLive = true;
        }
        redrawGraph();
    });

    // drag handle below the graph — adjusts canvas height; modelsDiv takes the rest.
    const graphResize = document.createElement("div");
    graphResize.className = "aimdo-graph-resize";
    graphResize.title = "Drag to resize graph";
    let graphDrag = null;
    graphResize.addEventListener("mousedown", (e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        e.stopPropagation();
        graphDrag = { startY: e.clientY, startHeight: graphCanvas.getBoundingClientRect().height };
        graphResize.classList.add("is-dragging");
    });
    document.addEventListener("mousemove", (e) => {
        if (!graphDrag) return;
        const visualScale = (body._panel && body._panel._scale) || 1;
        const delta = (e.clientY - graphDrag.startY) / visualScale;
        const next = Math.max(20, Math.min(600, graphDrag.startHeight / visualScale + delta));
        graphHeight = Math.round(next);
        graphCanvas.style.height = graphHeight + "px";
    });
    document.addEventListener("mouseup", () => {
        if (graphDrag) {
            graphDrag = null;
            graphResize.classList.remove("is-dragging");
            saveState({ graphHeight });
        }
    });
    body.appendChild(graphResize);

    const modelsDiv = document.createElement("div");
    modelsDiv.id = "aimdo-models";
    modelsDiv.style.cssText = "flex: 1 1 auto; min-height: 0; overflow-y: auto;";
    body.appendChild(modelsDiv);

    refs = {
        contentDiv,
        graphHeader,
        graphCanvas,
        graphResize,
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

function updateGraphTimes() {
    if (!refs || !refs.graphHeader) return;
    refs.graphHeader.style.color = "var(--aimdo-textDim)";
    const leftEl = refs.graphHeader.querySelector(".graph-time-left");
    const hoverEl = refs.graphHeader.querySelector(".graph-hover-info");
    const rightEl = refs.graphHeader.querySelector(".graph-time-right");
    const len = history.len;
    if (len < 2) {
        leftEl.textContent = "";
        hoverEl.textContent = "";
        rightEl.textContent = "";
        return;
    }
    const visible = Math.min(GRAPH_POINTS, len - history.viewOffset);
    if (visible < 1) {
        leftEl.textContent = "";
        hoverEl.textContent = "";
        rightEl.textContent = "";
        return;
    }
    const startIdx = len - visible - history.viewOffset;
    const endIdx = len - 1 - history.viewOffset;
    leftEl.textContent = formatClock(historyGet(history.times, startIdx));
    rightEl.textContent = history.followLive ? "live" : formatClock(historyGet(history.times, endIdx));
    if (graphHover.idx != null) {
        const used = history.total_vram - historyGet(history.free_vram, graphHover.idx);
        const parts = [
            formatClock(historyGet(history.times, graphHover.idx)),
            formatBytes(used),
        ];
        if (gpuLineVisible) parts.push(Math.round(historyGet(history.gpu_util, graphHover.idx)) + "%");
        hoverEl.textContent = parts.join(" · ");
    } else {
        hoverEl.textContent = "";
    }
}

function applyRowCollapsed(row) {
    row.bar.style.display = row.collapsed ? "none" : "flex";
    row.legend.style.display = (row.collapsed || !showLegends) ? "none" : "flex";
    row.vbarsDiv.style.display = row.collapsed ? "none" : "";
    row.chevron.textContent = row.collapsed ? "▸" : "▾";
}

// build (or reuse) a model row, mutating only what changed
function renderModelRow(r, m, data) {
    const wantsWm = m.dynamic && data.aimdo_active;
    let row = r.modelRows[m.index];
    if (!row) {
        const el = document.createElement("div");
        el.className = "aimdo-model-row";
        const head = document.createElement("div");
        head.className = "aimdo-model-head";
        const nameWrap = document.createElement("span");
        nameWrap.className = "aimdo-model-name";
        nameWrap.title = "Click to collapse/expand";
        const chevron = document.createElement("span");
        chevron.className = "aimdo-model-chevron";
        const nameSpan = document.createElement("span");
        nameWrap.appendChild(chevron);
        nameWrap.appendChild(nameSpan);
        const right = document.createElement("span");
        right.className = "aimdo-model-right";
        const sizeSpan = document.createElement("span");
        right.appendChild(sizeSpan);
        const unloadBtn = document.createElement("span");
        unloadBtn.className = "aimdo-unload-btn";
        unloadBtn.dataset.index = m.index;
        unloadBtn.textContent = "x";
        unloadBtn.title = "Unload this model";
        right.appendChild(unloadBtn);
        head.appendChild(nameWrap);
        head.appendChild(right);
        el.appendChild(head);
        const bar = document.createElement("div");
        bar.className = "aimdo-model-bar";
        el.appendChild(bar);
        const legend = document.createElement("div");
        legend.className = "aimdo-model-legend";
        el.appendChild(legend);
        const vbarsDiv = document.createElement("div");
        el.appendChild(vbarsDiv);
        row = { el, chevron, nameSpan, sizeSpan, right, unloadBtn, bar, barSegs: [], legend, vbarsDiv, vbarRefs: [], wmBtn: null, lastDynamic: null, lastVbarSig: "", collapsed: false };
        nameWrap.addEventListener("click", () => {
            row.collapsed = !row.collapsed;
            modelCollapsed[m.name] = row.collapsed;
            saveState({ modelCollapsed });
            applyRowCollapsed(row);
        });
        row.collapsed = !!modelCollapsed[m.name];
        applyRowCollapsed(row);
        r.modelRows[m.index] = row;
    }

    row.nameSpan.textContent = m.name + (m.dynamic ? "" : " (static)");
    const typeColor = MODEL_TYPE_COLOR[m.type];
    const vramColor = (colorModelBars && typeColor) || C.vram;
    row.nameSpan.style.color = (colorModelName && typeColor) || C.text;
    row.el.style.borderColor = (colorModelStroke && typeColor) ? hexToRgba(typeColor, 0.4) : "transparent";
    if (!row.collapsed) row.legend.style.display = showLegends ? "flex" : "none";
    row.sizeSpan.textContent = formatBytes(m.total_size);

    if (wantsWm && !row.wmBtn) {
        const wm = document.createElement("span");
        wm.className = "aimdo-reset-wm-btn";
        wm.dataset.index = m.index;
        wm.textContent = "wm";
        wm.title = "reset watermark";
        row.right.insertBefore(wm, row.unloadBtn);
        row.wmBtn = wm;
    } else if (!wantsWm && row.wmBtn) {
        row.wmBtn.remove();
        row.wmBtn = null;
    }

    const barColors = m.dynamic ? [C.vram, C.pinned, C.loadedRam, C.unloaded] : [C.vram, C.pinned, C.loadedRam, C.pinned];
    if (row.lastDynamic !== m.dynamic || row.barSegs.length !== barColors.length) {
        row.bar.innerHTML = "";
        row.barSegs = [];
        for (const color of barColors) {
            const seg = document.createElement("div");
            seg.style.cssText = `background:${color};height:100%;`;
            row.bar.appendChild(seg);
            row.barSegs.push(seg);
        }
        row.lastDynamic = m.dynamic;
    }
    // re-apply each tick so theme changes reach the segments created once on dynamic-change rebuild.
    row.barSegs.forEach((seg, i) => { seg.style.background = i === 0 ? vramColor : barColors[i]; });

    if (m.dynamic) {
        const pinnedRam = m.pinned_ram || 0;
        const loadedRam = m.loaded_ram || 0;
        const unloadedSize = Math.max(0, m.total_size - m.vbar_loaded - pinnedRam - loadedRam);
        const total = m.total_size || 1;
        row.barSegs[0].style.width = (m.vbar_loaded / total * 100) + "%";
        row.barSegs[0].title = "VRAM: " + formatBytes(m.vbar_loaded);
        row.barSegs[1].style.width = (pinnedRam / total * 100) + "%";
        row.barSegs[1].title = "pinned RAM: " + formatBytes(pinnedRam);
        row.barSegs[2].style.width = (loadedRam / total * 100) + "%";
        row.barSegs[2].title = "loaded RAM: " + formatBytes(loadedRam);
        row.barSegs[3].style.width = (unloadedSize / total * 100) + "%";
        row.barSegs[3].title = "unloaded: " + formatBytes(unloadedSize);
        row.legend.innerHTML =
            `<span><span style="color:${vramColor};">&#9632;</span> VRAM ${formatBytes(m.vbar_loaded)}</span>` +
            (pinnedRam > 0 ? `<span><span style="color:var(--aimdo-pinned);">&#9632;</span> pinned ${formatBytes(pinnedRam)}</span>` : "") +
            (loadedRam > 0 ? `<span><span style="color:var(--aimdo-loadedRam);">&#9632;</span> loaded ${formatBytes(loadedRam)}</span>` : "") +
            `<span><span style="color:var(--aimdo-unloaded);">&#9632;</span> unloaded ${formatBytes(unloadedSize)}</span>`;
    } else {
        const inRam = Math.max(0, m.total_size - m.loaded_size);
        const pinnedRam = m.pinned_ram || 0;
        const loadedRam = m.loaded_ram || 0;
        const otherRam = Math.max(0, inRam - pinnedRam - loadedRam);
        const total = m.total_size || 1;
        row.barSegs[0].style.width = (m.loaded_size / total * 100) + "%";
        row.barSegs[0].title = "VRAM: " + formatBytes(m.loaded_size);
        row.barSegs[1].style.width = (pinnedRam / total * 100) + "%";
        row.barSegs[1].title = "pinned RAM: " + formatBytes(pinnedRam);
        row.barSegs[2].style.width = (loadedRam / total * 100) + "%";
        row.barSegs[2].title = "loaded RAM: " + formatBytes(loadedRam);
        row.barSegs[3].style.width = (otherRam / total * 100) + "%";
        row.barSegs[3].title = "RAM: " + formatBytes(otherRam);
        row.legend.innerHTML =
            `<span><span style="color:${vramColor};">&#9632;</span> VRAM ${formatBytes(m.loaded_size)}</span>` +
            (pinnedRam > 0 ? `<span><span style="color:var(--aimdo-pinned);">&#9632;</span> pinned ${formatBytes(pinnedRam)}</span>` : "") +
            (loadedRam > 0 ? `<span><span style="color:var(--aimdo-loadedRam);">&#9632;</span> loaded ${formatBytes(loadedRam)}</span>` : "") +
            (otherRam > 0 ? `<span><span style="color:var(--aimdo-pinned);">&#9632;</span> RAM ${formatBytes(otherRam)}</span>` : "");
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
                lbl.style.cssText = `font-size:10px;color:var(--aimdo-textDim);margin-top:3px;`;
                lbl.textContent = vb.device;
                row.vbarsDiv.appendChild(lbl);
            }
            const pgrid = document.createElement("div");
            pgrid.style.cssText = "margin-top:2px;";
            row.vbarsDiv.appendChild(pgrid);
            const stats = document.createElement("div");
            stats.style.cssText = `color:var(--aimdo-textDim);font-size:10px;margin-top:2px;`;
            row.vbarsDiv.appendChild(stats);
            row.vbarRefs.push({ vi, pgrid, stats });
        }
        row.lastVbarSig = sig;
    }
    return row;
}

function renderData(body, data) {
    if (!data.enabled) {
        body.innerHTML = `<div style="color:var(--aimdo-textDim);">not available</div>`;
        refs = null;
        return;
    }

    const r = ensureStructure(body);
    const pw = body._panel.getBoundingClientRect().width;
    body._titleSpan.style.display = showTitle ? "" : "none";
    body._titleSpan.textContent =
        pw >= 320 && data.aimdo_active ? "Memory (aimdo)" :
        pw >= 240 ? "Memory" : "";
    data.gpu_util = smoothGpuUtil(data.gpu_util);
    // throttle history snapshots so a fast poll rate (e.g. 100 ms) doesn't fill
    // the buffer in 2 minutes; UI keeps updating at the full poll cadence.
    if (Date.now() - (history.lastPush || 0) >= HISTORY_TICK_MS) {
        pushHistory(data);
        history.lastPush = Date.now();
    }

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
    const loadedRamTotal = data.loaded_ram || 0;
    const pythonOther = Math.max(0, processRam - pinnedRamTotal - loadedRamTotal);
    const ramOther = Math.max(0, ramUsed - processRam);
    const pinnedRamPct = (pinnedRamTotal / ramTotal * 100).toFixed(0);
    const loadedRamPct = (loadedRamTotal / ramTotal * 100).toFixed(0);
    const pythonOtherPct = (pythonOther / ramTotal * 100).toFixed(0);
    const ramOtherPct = (ramOther / ramTotal * 100).toFixed(0);

    const mb = body._miniBar;
    const _u = miniShowUnits;
    const _n = miniShowNumbers;
    // "%" isn't a unit — units off should still show ratios as percentages, just
    // without GB/MB/°C/W. Quantities with a denominator (RAM/VRAM/power) become
    // "used/total" percentages; CPU/GPU util are already percentages so the "%"
    // stays on regardless; temp has no natural denominator so we drop the °C.
    // leading zero on single-digit values keeps width stable, matching the CPU/GPU util format
    const asPct = (num, total) => {
        if (total <= 0) return "?";
        const p = Math.round(num / total * 100);
        return (p < 10 ? "0" : "") + p + "%";
    };
    mb.querySelector(".mini-vram-usage").textContent = !_n ? "" :
        _u ? `${formatBytes(used)} / ${formatBytes(data.total_vram)}`
           : asPct(used, data.total_vram);
    mb.querySelector(".mini-vram-bar").innerHTML =
        `<div style="background:var(--aimdo-vram);height:100%;width:${aimdoPct}%;"></div>` +
        `<div style="background:var(--aimdo-torch);height:100%;width:${torchPct}%;"></div>` +
        `<div style="background:var(--aimdo-torchCache);height:100%;width:${torchCachePct}%;"></div>` +
        `<div style="background:var(--aimdo-other);height:100%;width:${otherPct}%;"></div>`;
    mb.querySelector(".mini-ram-usage").textContent = !_n ? "" :
        _u ? `${formatBytes(ramUsed)} / ${formatBytes(ramTotal)}`
           : asPct(ramUsed, ramTotal);
    mb.querySelector(".mini-ram-bar").innerHTML =
        `<div style="background:var(--aimdo-pinned);height:100%;width:${pinnedRamPct}%;"></div>` +
        `<div style="background:var(--aimdo-loadedRam);height:100%;width:${loadedRamPct}%;"></div>` +
        `<div style="background:var(--aimdo-python);height:100%;width:${pythonOtherPct}%;"></div>` +
        `<div style="background:var(--aimdo-other);height:100%;width:${ramOtherPct}%;"></div>`;

    // toggleable "Type" label hides RAM / VRAM / CPU / GPU prefixes (plus any device suffix)
    mb.querySelector(".mini-ram-label").style.display = miniShowType ? "" : "none";
    mb.querySelector(".mini-vram-label").style.display = miniShowType ? "" : "none";

    mb.querySelector(".mini-ram-section").style.display = showRamInMini ? "" : "none";
    mb.querySelector(".mini-vram-section").style.display = showVramInMini ? "" : "none";
    const cpuSection = mb.querySelector(".mini-cpu-section");
    if (data.cpu_util != null && showCpuInMini) {
        cpuSection.style.display = "";
        const cpuColor = gpuUtilColor(data.cpu_util);
        const cpuPct = Math.round(data.cpu_util);
        mb.querySelector(".mini-cpu-usage").innerHTML = _n
            ? `<span style="color:${cpuColor};">${(cpuPct < 10 ? "0" : "") + cpuPct}%</span>`
            : "";
        const cpuFill = mb.querySelector(".mini-cpu-fill");
        cpuFill.style.background = cpuColor;
        cpuFill.style.width = `${cpuPct}%`;
        const cpuLabel = mb.querySelector(".mini-cpu-label");
        cpuLabel.textContent = (showHwNames && data.cpu_name) ? `CPU (${shortenCpuName(data.cpu_name)})` : "CPU";
        cpuLabel.title = data.cpu_name || "";
        cpuLabel.style.display = miniShowType ? "" : "none";
    } else {
        cpuSection.style.display = "none";
    }
    const gpuSection = mb.querySelector(".mini-gpu-section");
    // each bar is independently toggleable; section only hides when ALL three are off
    // (or unavailable). util is no longer special — it can be off while temp/power show.
    const _showUtil = data.gpu_util != null && showGpuInMini;
    const _showTemp = data.gpu_temp != null && miniShowGpuTemp;
    const _showPower = data.gpu_power != null && data.gpu_power_limit != null && miniShowGpuPower;
    const _activeBars = (_showUtil ? 1 : 0) + (_showTemp ? 1 : 0) + (_showPower ? 1 : 0);
    if (_activeBars > 0) {
        gpuSection.style.display = "";
        // compact 8px / 3px styling kicks in only when there's >1 bar to fit
        const isSingleBar = _activeBars === 1;
        gpuSection.classList.toggle("is-multibar", !isSingleBar);

        // title row — "GPU" or "GPU (RTX 4090)" on the left, value on the right when
        // only one bar is visible (then the layout matches RAM/VRAM/CPU above).
        const gpuLabel = mb.querySelector(".mini-gpu-label");
        gpuLabel.textContent = (showHwNames && data.gpu_name) ? `GPU (${shortenGpuName(data.gpu_name)})` : "GPU";
        gpuLabel.title = data.gpu_name || "";
        gpuLabel.style.display = miniShowType ? "" : "none";
        // keep the row visible if either the label is on, or there's a single-bar value to show
        mb.querySelector(".mini-gpu-row").style.display = (miniShowType || isSingleBar) ? "" : "none";

        // util row
        mb.querySelector(".mini-util-row").style.display = _showUtil ? "" : "none";
        if (_showUtil) {
            const gpuColor = gpuUtilColor(data.gpu_util);
            const gpuFill = mb.querySelector(".mini-gpu-fill");
            gpuFill.style.background = gpuColor;
            gpuFill.style.width = `${data.gpu_util}%`;
            mb.querySelector(".mini-gpu-usage").innerHTML = _n
                ? `<span style="color:${gpuColor};">${(data.gpu_util < 10 ? "0" : "") + data.gpu_util}%</span>`
                : "";
        }

        // temp row — 100°C is full scale; units-off uses "%" since the bar already
        // treats 100°C as the denominator (75°C → 75% of the bar full).
        mb.querySelector(".mini-temp-row").style.display = _showTemp ? "" : "none";
        if (_showTemp) {
            const tempColor = gpuTempColor(data.gpu_temp);
            const tempFill = mb.querySelector(".mini-temp-fill");
            tempFill.style.background = tempColor;
            tempFill.style.width = `${Math.min(100, data.gpu_temp)}%`;
            mb.querySelector(".mini-temp-usage").innerHTML = _n
                ? `<span style="color:${tempColor};">${data.gpu_temp}${_u ? "&deg;C" : "%"}</span>`
                : "";
        }

        // power row — fill is draw/limit; value is W or % depending on the Units toggle
        mb.querySelector(".mini-power-row").style.display = _showPower ? "" : "none";
        if (_showPower) {
            const powerColor = gpuPowerColor(data.gpu_power, data.gpu_power_limit);
            const powerFill = mb.querySelector(".mini-power-fill");
            powerFill.style.background = powerColor;
            const powerPct = data.gpu_power_limit > 0
                ? Math.min(100, data.gpu_power / data.gpu_power_limit * 100)
                : 0;
            powerFill.style.width = `${powerPct}%`;
            const powerText = _u
                ? formatPower(data.gpu_power, data.gpu_power_limit)
                : asPct(data.gpu_power, data.gpu_power_limit);
            mb.querySelector(".mini-power-usage").innerHTML = _n
                ? `<span style="color:${powerColor};">${powerText}</span>`
                : "";
        }

        // single-bar mode: lift the visible value into the title row so the section
        // reads like RAM/VRAM/CPU above (label + value on top, bar below).
        let headerHtml = "";
        if (isSingleBar) {
            if (_showUtil) headerHtml = mb.querySelector(".mini-gpu-usage").innerHTML;
            else if (_showTemp) headerHtml = mb.querySelector(".mini-temp-usage").innerHTML;
            else if (_showPower) headerHtml = mb.querySelector(".mini-power-usage").innerHTML;
        }
        mb.querySelector(".mini-gpu-header-value").innerHTML = headerHtml;
    } else {
        gpuSection.style.display = "none";
    }

    r.contentDiv.innerHTML = `<div style="margin-bottom:4px;">
        <div style="display:flex;justify-content:space-between;gap:6px;margin-bottom:2px;">
            <span>RAM</span>
            <span>${formatBytes(ramUsed)} / ${formatBytes(ramTotal)}</span>
        </div>
        <div style="background:var(--aimdo-barBg);border-radius:3px;height:8px;overflow:hidden;display:flex;">
            <div style="background:var(--aimdo-pinned);height:100%;width:${pinnedRamPct}%;" title="pinned: ${formatBytes(pinnedRamTotal)}"></div>
            <div style="background:var(--aimdo-loadedRam);height:100%;width:${loadedRamPct}%;" title="loaded: ${formatBytes(loadedRamTotal)}"></div>
            <div style="background:var(--aimdo-python);height:100%;width:${pythonOtherPct}%;" title="python: ${formatBytes(pythonOther)}"></div>
            <div style="background:var(--aimdo-other);height:100%;width:${ramOtherPct}%;" title="other: ${formatBytes(ramOther)}"></div>
        </div>
        ${showLegends ? `<div style="display:flex;gap:8px;font-size:10px;color:var(--aimdo-textDim);margin-top:2px;">
            <span><span style="color:var(--aimdo-pinned);">&#9632;</span> pinned ${formatBytes(pinnedRamTotal)}</span>
            <span><span style="color:var(--aimdo-loadedRam);">&#9632;</span> loaded ${formatBytes(loadedRamTotal)}</span>
            <span><span style="color:var(--aimdo-python);">&#9632;</span> python ${formatBytes(pythonOther)}</span>
            <span><span style="color:var(--aimdo-other);">&#9632;</span> other ${formatBytes(ramOther)}</span>
        </div>` : ""}
    </div>
    <div style="margin-bottom:4px;">
        <div style="display:flex;justify-content:space-between;gap:6px;margin-bottom:2px;">
            <span title="${escHtml(data.gpu_name || "")}">VRAM${(showHwNames && data.gpu_name) ? ` (${escHtml(shortenGpuName(data.gpu_name))})` : ""}</span>
            <span>${formatBytes(used)} / ${formatBytes(data.total_vram)}</span>
        </div>
        <div style="background:var(--aimdo-barBg);border-radius:3px;height:8px;overflow:hidden;display:flex;">
            <div style="background:var(--aimdo-vram);height:100%;width:${aimdoPct}%;" title="models: ${formatBytes(aimdo)}"></div>
            <div style="background:var(--aimdo-torch);height:100%;width:${torchPct}%;" title="torch: ${formatBytes(torchActive)}"></div>
            <div style="background:var(--aimdo-torchCache);height:100%;width:${torchCachePct}%;" title="cache: ${formatBytes(torchCache)}"></div>
            <div style="background:var(--aimdo-other);height:100%;width:${otherPct}%;" title="other: ${formatBytes(otherUsed)}"></div>
        </div>
        ${showLegends ? `<div style="display:flex;gap:8px;font-size:10px;color:var(--aimdo-textDim);margin-top:2px;">
            ${aimdo > 0 ? `<span><span style="color:var(--aimdo-vram);">&#9632;</span> models ${formatBytes(aimdo)}</span>` : ""}
            ${torchActive > 0 ? `<span><span style="color:var(--aimdo-torch);">&#9632;</span> torch ${formatBytes(torchActive)}</span>` : ""}
            ${torchCache > 0 ? `<span><span style="color:var(--aimdo-torchCache);">&#9632;</span> cache ${formatBytes(torchCache)}</span>` : ""}
            <span><span style="color:var(--aimdo-other);">&#9632;</span> other ${formatBytes(otherUsed)}</span>
        </div>` : ""}
        <div style="display:flex;gap:10px;font-size:10px;color:var(--aimdo-textDim);margin-top:2px;">
            <span>peak: ${formatBytes(peakVramUsed)}</span>
            <span>cache: ${formatBytes(data.torch_reserved - data.torch_active)}</span>
            ${data.gpu_util != null ? `<span class="aimdo-gpu-util" title="Click to toggle GPU line on graph" style="color:${gpuUtilColor(data.gpu_util)};cursor:pointer;opacity:${gpuLineVisible ? 1 : 0.4};">GPU ${data.gpu_util < 10 ? "0" : ""}${data.gpu_util}%</span>` : ""}
            ${data.gpu_temp != null ? `<span style="color:${gpuTempColor(data.gpu_temp)};">${data.gpu_temp}&deg;C</span>` : ""}
            ${data.gpu_power != null && data.gpu_power_limit != null ? `<span title="GPU power draw / cap" style="color:${gpuPowerColor(data.gpu_power, data.gpu_power_limit)};">${formatPower(data.gpu_power, data.gpu_power_limit)}</span>` : ""}
            ${execState.running ? `<span style="color:var(--aimdo-running);">&#9679; ${execState.node || "running"}${execState.progress ? " " + execState.progress : ""}</span>` : `<span>&#9679; idle</span>`}
        </div>
    </div>`;

    // sync canvas backing to device pixels: visual viewport px × devicePixelRatio.
    // Drawing then happens in logical (panel-local) CSS px via the totalScale transform,
    // so cells/lines scale with panelScale while staying crisp on HiDPI.
    const panelScaleNow = body._panel._scale || 1;
    const dpr = window.devicePixelRatio || 1;
    const totalScale = panelScaleNow * dpr;
    const gRect = r.graphCanvas.getBoundingClientRect();
    if (gRect.width > 0 && gRect.height > 0) {
        const backingW = Math.max(1, Math.round(gRect.width * dpr));
        const backingH = Math.max(1, Math.round(gRect.height * dpr));
        if (r.graphCanvas.width !== backingW) r.graphCanvas.width = backingW;
        if (r.graphCanvas.height !== backingH) r.graphCanvas.height = backingH;
        r.graphCtx.setTransform(totalScale, 0, 0, totalScale, 0, 0);
        drawGraph(r.graphCtx, gRect.width / panelScaleNow, gRect.height / panelScaleNow);
        updateGraphTimes();
    }

    // models section — incremental DOM updates: keep rows across polls, only mutate text/widths
    if (data.models.length === 0 && !r.noModelsMsg) {
        r.noModelsMsg = document.createElement("div");
        r.noModelsMsg.textContent = "No models loaded";
        r.noModelsMsg.style.cssText = `color:var(--aimdo-textDim);margin-top:6px;`;
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

    if (!r.bottomLegend) {
        r.bottomLegend = document.createElement("div");
        r.bottomLegend.style.cssText = `display:flex;flex-wrap:wrap;gap:8px;font-size:10px;color:var(--aimdo-textDim);margin-top:4px;border-bottom:1px solid var(--aimdo-border);padding-bottom:4px;`;
        r.bottomLegend.innerHTML =
            `<span><span style="color:var(--aimdo-vram);">&#9632;</span> VRAM</span>` +
            `<span><span style="color:var(--aimdo-pinned);">&#9632;</span> pinned</span>` +
            `<span><span style="color:var(--aimdo-loadedRam);">&#9632;</span> loaded</span>` +
            `<span><span style="color:var(--aimdo-unloaded);">&#9632;</span> unloaded</span>` +
            `<span><span style="color:var(--aimdo-torch);">&#9632;</span> torch</span>` +
            `<span><span style="color:var(--aimdo-totalLine);">&#9472;</span> total</span>` +
            `<span><span style="color:var(--aimdo-gpuUtil);">&#9472;</span> GPU %</span>`;
        r.modelsDiv.insertBefore(r.bottomLegend, r.modelsDiv.firstChild);
    }
    r.bottomLegend.style.display = showLegends ? "flex" : "none";

    for (const m of data.models) {
        const isNew = !r.modelRows[m.index];
        const row = renderModelRow(r, m, data);
        if (isNew) r.modelsDiv.appendChild(row.el);
    }

    // draw page grids and update vbar stat text
    for (const m of data.models) {
        const row = r.modelRows[m.index];
        if (!row || !row.vbarRefs.length || row.collapsed) continue;
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
            const vramColor = (colorModelBars && MODEL_TYPE_COLOR[m.type]) || C.vram;
            // swatch carries the category color; text inherits readable textDim.
            ref.stats.innerHTML =
                `<span><span style="color:${vramColor};">&#9632;</span> ${vramPages} VRAM (${formatBytes(vramPages * PAGE)})</span>` +
                ` <span><span style="color:var(--aimdo-unloaded);">&#9632;</span> ${ramPages} unloaded (${formatBytes(ramPages * PAGE)})</span>`;

            let canvas = r.pageCanvases[vkey];
            if (!canvas) {
                canvas = document.createElement("canvas");
                canvas.style.cssText = "width:100%;border-radius:2px;";
                r.pageCanvases[vkey] = canvas;
                r.pageCtxs[vkey] = canvas.getContext("2d");
            }
            if (canvas.parentElement !== ref.pgrid) ref.pgrid.appendChild(canvas);
            const pgVisualW = ref.pgrid.getBoundingClientRect().width
                || r.modelsDiv.getBoundingClientRect().width
                || 300 * panelScaleNow;
            const pgCssW = pgVisualW / panelScaleNow;
            drawPageGrid(r.pageCtxs[vkey], pgCssW, vb.residency, st ? st.changeAge : new Uint8Array(vb.residency.length), panelScaleNow, colorModelBars ? MODEL_TYPE_COLOR[m.type] : undefined);
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
        // wait for aimdo_viz.css so applyPalette can read CSS variables back into C —
        // canvas drawing needs real hex strings, not unresolved var() references.
        await cssLoaded;
        const body = createPanel();

        api.addEventListener("execution_start", () => {
            execState.running = true;
            execState.node = null;
            execState.progress = null;
            pushExecEvent("start");
            body._updateExecBtnState?.();
        });
        api.addEventListener("executing", ({ detail }) => {
            const wasRunning = execState.running;
            execState.running = detail != null;
            execState.node = null;
            execState.progress = null;
            if (wasRunning && !execState.running) pushExecEvent("end");
            body._updateExecBtnState?.();
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
