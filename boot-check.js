// temp: run index.html's inline scripts under a stubbed DOM to catch boot-time
// exceptions exactly where a browser would hit them. 3 frames of the loop run.
const fs = require("fs"), vm = require("vm");

function ctx2d() {
  const grad = { addColorStop() {} };
  const bad = [];
  const seen = { arcs: 0, minX: 1e9, maxX: -1e9, minY: 1e9, maxY: -1e9 };
  const fin = (v) => typeof v === "number" && Number.isFinite(v);
  const wrap = (name, checks) => function (...a) {
    const msg = checks(a);
    if (msg) bad.push(name + "(" + a.slice(0, 6).join(",") + "): " + msg);
    return undefined;
  };
  const base = {
    canvas: null,
    createRadialGradient: (a, b, c, d, e, f) => {
      if (![a, b, c, d, e, f].every(fin)) bad.push("createRadialGradient non-finite");
      return grad;
    },
    createLinearGradient: () => grad,
    measureText: () => ({ width: 8 }),
    arc: wrap("arc", (a) => {
      if (![a[0], a[1], a[2]].every(fin)) return "NON-FINITE (invisible in browser)";
      if (a[2] < 0) return "IndexSizeError (negative radius) THROWS";
      seen.arcs++;
      seen.minX = Math.min(seen.minX, a[0]); seen.maxX = Math.max(seen.maxX, a[0]);
      seen.minY = Math.min(seen.minY, a[1]); seen.maxY = Math.max(seen.maxY, a[1]);
    }),
    fillRect: wrap("fillRect", (a) =>
      a.length === 4 && !a.every(fin) ? "NON-FINITE" : undefined),
    drawImage: wrap("drawImage", (a) =>
      a.slice(1).some(v => !fin(v) && typeof v === "number") ? "NON-FINITE coord" : undefined),
    setTransform: wrap("setTransform", (a) =>
      !a.every(fin) ? "NON-FINITE (everything downstream invisible)" : undefined),
    translate: wrap("translate", (a) => !a.every(fin) ? "NON-FINITE" : undefined),
    _bad: bad, _seen: seen,
  };
  return new Proxy(base, {
    get(t, p) {
      if (p in t) return t[p];
      if (typeof p === "string") return t[p] = function () {};
      return undefined;
    },
    set(t, p, v) {
      if ((p === "globalAlpha" || p === "lineWidth") && !Number.isFinite(v))
        t._bad.push("set " + p + "=" + v + " (browser clamps/ignores)");
      t[p] = v; return true;
    },
  });
}
const VIEW = (() => {
  const [w, h] = (process.env.SIZE || "1600x800").split("x").map(Number);
  return { w, h };
})();
function el(tag) {
  const e = {
    tagName: (tag || "div").toUpperCase(),
    style: {}, children: [], textContent: "", value: "", checked: false, title: "",
    width: 300, height: 150,
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    addEventListener() {}, appendChild(c) { e.children.push(c); },
    setAttribute() {}, getAttribute: () => null,
    getBoundingClientRect: () => ({ left: 0, top: 0, right: VIEW.w, bottom: VIEW.h, width: VIEW.w, height: VIEW.h }),
    clientWidth: VIEW.w, clientHeight: VIEW.h,
    setPointerCapture() {},
  };
  e.getContext = () => (e._c || (e._c = ctx2d()));
  return e;
}
const byId = {};
const documentStub = {
  getElementById(id) { return byId[id] || (byId[id] = el(id === "cv" ? "canvas" : "div")); },
  createElement(t) { return el(t); },
  body: el("body"), documentElement: el("html"),
  fullscreenElement: null, addEventListener() {},
};
const rafQueue = [];
const windowStub = {
  devicePixelRatio: Number(process.env.DPR || 2), innerWidth: VIEW.w, innerHeight: VIEW.h,
  outerWidth: VIEW.w, outerHeight: VIEW.h + 90,
  matchMedia: () => ({ matches: false }), addEventListener() {},
  requestAnimationFrame(cb) { rafQueue.push(cb); return rafQueue.length; },
};
const store = {};
// env knobs: MODE=1|2|3 pre-seeds the prefs like a returning user would have;
// SIZE=1600x800 sets the stage; DPR caps devicePixelRatio
if (process.env.MODE) store["sph-fluid-opts-1"] = JSON.stringify({ mode: +process.env.MODE });
const [sw, sh] = (process.env.SIZE || "1600x800").split("x").map(Number);
const sandbox = {
  window: windowStub, document: documentStub, navigator: { standalone: false },
  localStorage: {
    getItem: (k) => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
  },
  requestAnimationFrame: windowStub.requestAnimationFrame,
  performance, console, DeviceMotionEvent: undefined, setTimeout, clearTimeout,
};
sandbox.window.localStorage = sandbox.localStorage;
vm.createContext(sandbox);

const html = fs.readFileSync("index.html", "utf8");
const simSrc = fs.readFileSync("sim.js", "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);

// boot step 1: sim.js as a plain script
try { vm.runInContext(simSrc, sandbox, { filename: "sim.js" }); }
catch (e) { console.log("sim.js LOAD ERROR:", e.message); process.exit(1); }
// boot step 2: inline main script
try { vm.runInContext(scripts[0], sandbox, { filename: "inline-main" }); }
catch (e) { console.log("BOOT ERROR (main script):", e.stack.split("\n").slice(0, 4).join("\n")); process.exit(1); }
// boot step 3: second inline (SW guard)
if (scripts[1]) {
  try { vm.runInContext(scripts[1], sandbox, { filename: "inline-sw" }); }
  catch (e) { console.log("BOOT ERROR (sw script):", e.message); process.exit(1); }
}
// pump frames like a real browser: first callback lands ~700ms after boot
// (parse+load spike), then 16ms ticks
try {
  const t0 = sandbox.performance.now();
  for (let n = 0; n < 4 && rafQueue.length; n++) {
    const cbs = rafQueue.splice(0);
    for (const cb of cbs) cb(t0 + (n === 0 ? 700 : 700 + n * 16));
  }
} catch (e) {
  console.log("FRAME ERROR:", e.stack.split("\n").slice(0, 4).join("\n"));
  process.exit(1);
}
const cvs = Object.values(byId).filter(x => x._c);
let problems = [];
for (const c of cvs) for (const b of c._c._bad) problems.push(c === byId.cv ? "main" : "offscreen", b);
const s = (byId.cv._c || {})._seen || {};
console.log("arc calls on main ctx:", s.arcs,
  s.arcs ? `x:[${s.minX.toFixed(0)}..${s.maxX.toFixed(0)}] y:[${s.minY.toFixed(0)}..${s.maxY.toFixed(0)}]` : "");
if (problems.length) { console.log("CANVAS PROBLEMS:", problems.length); problems.slice(0, 8).forEach(p => console.log("  ", p)); process.exit(1); }
console.log("BOOT + 4 FRAMES OK, no non-finite canvas ops");
