/*
 * sim.js — 2D Smoothed Particle Hydrodynamics core (Müller 2003 style).
 *
 * Pure physics: no DOM, no canvas. Works in the browser (exposes globalThis.SPHSim)
 * and in Node (module.exports) so test.js can require the exact same core.
 *
 * Kernels (2D, smoothing radius h):
 *   poly6    : W(r)      = POLY6 * (h^2 - r^2)^3        (density, value only)
 *   spiky    : grad W(r) = SPIKY_GRAD * (h - r)^2 * r_vec/r   (pressure force)
 *   viscosity: lap  W(r) = VISC_LAP * (h - r)           (viscosity force)
 *
 * Update order per substep (semi-implicit / symplectic Euler):
 *   1. rebuild uniform spatial hash grid (cell size = h)
 *   2. density rho_i = sum_j m * poly6(r_ij)
 *   3. pressure p_i  = k * (rho_i - restDensity), floored at 0
 *                    (or at -eps*k*rho0 when opts.tension = eps > 0)
 *   4. accel a_i = pressure + viscosity + gravity + boundary + pointer
 *   5. v += a*dt ;  x += v*dt
 */
(function (global, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else global.SPHSim = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var TAU = Math.PI * 2;

  // ---------------------------------------------------------------------------
  // Default tuning constants (pixels, seconds). Units are made CONSISTENT by
  // anchoring on a rest particle spacing:
  //
  //   spacing = 9 px          -> target nearest-neighbour distance at rest.
  //   h = 26 px               -> smoothing radius; h/spacing ~ 2.9 samples the
  //                              kernel over ~26 neighbours (good support).
  //   mass = 25               -> with poly6 this makes restDensity ~= 0.3.
  //   restDensity             -> NOT hand-tuned: computed at construction as
  //                              mass * (sum of the poly6 kernel over a square
  //                              lattice at `spacing`). Pressure at the rest
  //                              configuration is then exactly zero. (The old
  //                              value 24 was measured from the fully-crushed
  //                              pile-up, which made every real fluid state
  //                              pressure-free: the particles fell through
  //                              each other and pancaked into a single line.)
  //   stiffness k = 4.5e6     -> EOS sound speed sqrt(k) ~= 2120 px/s, well
  //                              above the ~800 px/s dam-break free-fall speed
  //                              (Mach < 0.4), so hydrostatic compression at
  //                              the pool floor is <1%: the pool is a BODY,
  //                              not a line. The old k=60 gave sound speed
  //                              sqrt(60) ~ 8 px/s -> Mach ~100 -> crush.
  //   dt = 1/400              -> CFL: c*dt/h = 2120/400/26 ~= 0.2, safe.
  //   boundStiff = 8000       -> floor spring must hold up the whole column
  //                              (~g * h_pool/spacing of accel at the bottom);
  //                              350 let the bottom crush into the wall zone.
  // Overridable via options.
  // ---------------------------------------------------------------------------
  var DEFAULTS = {
    width: 800,
    height: 600,
    count: 1500,
    h: 26,            // smoothing radius (== spatial hash cell size)
    spacing: 9,       // rest nearest-neighbour distance (px)
    mass: 25,         // particle mass; sets restDensity via the lattice sum
    // restDensity: computed (see comment) unless provided in opts
    stiffness: 4.5e6, // gas constant k: p = k*(rho-rho0)
    viscosity: 600.0, // dynamic viscosity mu. NOTE: mu is a MULTIPLIER in the
                      // viscosity force; with rho0~0.31 this is a kinematic
                      // viscosity nu = mu/rho0 ~ 1900 px^2/s, enough to kill a
                      // box-scale slosh in a few seconds at this particle
                      // count (too few particles to cascade to turbulence).
    gravity: 900,     // gravity magnitude (px/s^2)
    dt: 1 / 400,      // fixed substep timestep (s); step() accumulates real time
    boundStiff: 8000, // wall spring stiffness (accel per px of penetration)
    boundDamp: 90.0,  // wall damping (accel per (px/s) of inward velocity)
    boundMargin: 20,  // wall spring/damper activation margin (px)
    maxVel: 3000,     // velocity clamp (safety, px/s; ~ sound speed)
    pointerRadius: 120,   // brush radius (px); live-adjustable at runtime
    attractForce: 90000,  // peak accel at pointer centre, falls linearly to 0
    repelForce: 90000,    //   at the brush edge. Split so attract and repel
                          //   can be tuned independently.
    seed: 12345,      // deterministic RNG seed for the dam-break block
    blockFrac: 0.6,   // dam-break block width as a fraction of the box width
    tension: 0,       // negative-pressure floor, as a fraction eps of k*rho0:
                      //   p = max(-eps*k*rho0, k*(rho-rho0))
                      // 0 keeps the classic clamp-at-zero behaviour. A small
                      // eps (~Sim.TENSION_EPS) gives under-density fluid a
                      // weak mutual attraction: cavities heal, spray droplets
                      // ball up. Overshoot (roughly > 0.05) invites "grape"
                      // clustering at the free surface — keep it small.
  };

  // Proven-safe recommended strength for the tension option.
  Sim.TENSION_EPS = 0.01;

  // ---------------------------------------------------------------------------
  // Small deterministic PRNG (mulberry32) so the dam-break block is reproducible
  // and tests are stable.
  // ---------------------------------------------------------------------------
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function Sim(opts) {
    opts = opts || {};
    var d = DEFAULTS;
    this.width = num(opts.width, d.width);
    this.height = num(opts.height, d.height);
    this.count = num(opts.count, d.count);
    this.h = num(opts.h, d.h);
    this.spacing = num(opts.spacing, d.spacing);
    this.mass = num(opts.mass, d.mass);
    this.stiffness = num(opts.stiffness, d.stiffness);
    this.viscosity = num(opts.viscosity, d.viscosity);
    this.gravity = num(opts.gravity, d.gravity);
    this.dt = num(opts.dt, d.dt);
    this.boundStiff = num(opts.boundStiff, d.boundStiff);
    this.boundDamp = num(opts.boundDamp, d.boundDamp);
    this.boundMargin = num(opts.boundMargin, d.boundMargin);
    this.maxVel = num(opts.maxVel, d.maxVel);
    this.pointerRadius = num(opts.pointerRadius, d.pointerRadius);
    // opts.pointerForce stays as a legacy alias that sets both halves at once.
    this.attractForce = num(opts.attractForce, num(opts.pointerForce, d.attractForce));
    this.repelForce = num(opts.repelForce, num(opts.pointerForce, d.repelForce));
    this.seed = num(opts.seed, d.seed);
    this.blockFrac = num(opts.blockFrac, d.blockFrac);
    this.tension = num(opts.tension, d.tension);
    if (this.tension < 0) this.tension = 0;          // no runaway repulsion
    if (this.tension > 0.05) this.tension = 0.05;    // grape-cluster guard

    // Gravity direction (unit vector). down by default.
    this.gx = 0; this.gy = this.gravity;

    // Interaction pointer state.
    this.pointer = { x: 0, y: 0, mode: 0, active: false }; // mode: -1 repel, +1 attract

    // Precompute kernel coefficients for the current h.
    this._kernels();

    // Rest density: unless overridden, the EXACT kernel sum of an infinite
    // square lattice at `spacing`. This is what makes the equation of state
    // honest: a fluid sitting at its rest spacing has p == 0, anything denser
    // pushes back, anything less is free surface. (Hand-picking this number
    // unrelated to the kernels is what made every state pressure-free before.)
    this.restDensity = num(opts.restDensity, this.mass * this._latticeSum());

    // Particle storage (structure of arrays).
    this.N = this.count;
    this.px = new Float64Array(this.N);
    this.py = new Float64Array(this.N);
    this.vx = new Float64Array(this.N);
    this.vy = new Float64Array(this.N);
    this.ax = new Float64Array(this.N);
    this.ay = new Float64Array(this.N);
    this.rho = new Float64Array(this.N);
    this.p = new Float64Array(this.N);

    // Spatial hash grid.
    this.cell = this.h;
    this.cols = Math.max(1, Math.ceil(this.width / this.cell));
    this.rows = Math.max(1, Math.ceil(this.height / this.cell));
    this.gridHead = new Int32Array(this.cols * this.rows); // head index per cell, -1 = empty
    this.gridNext = new Int32Array(this.N);               // linked list next per particle

    this.time = 0;
    this.reset();
  }

  function num(a, b) { return (typeof a === "number" && isFinite(a)) ? a : b; }

  Sim.prototype._kernels = function () {
    var h = this.h, h2 = h * h, h5 = Math.pow(h, 5), h8 = Math.pow(h, 8);
    this.kPoly6 = 4.0 / (Math.PI * h8);        // poly6  (2D)
    // Spiky kernel gradient magnitude w.r.t. particle i's position. W_spiky
    // decreases with r, so dW/dr < 0; the gradient wrt x_i is dW/dr * (-r_vec/r),
    // which points FROM i TOWARD j with POSITIVE magnitude 30/(pi h^5)(h-r)^2/r.
    // The pressure acceleration is a_i = -sum m (pi/rho_i^2 + pj/rho_j^2) * gradW,
    // so with a positive kSpiky the minus sign makes the force push i AWAY from j.
    this.kSpiky = 30.0 / (Math.PI * h5);       // spiky gradient (2D), positive
    this.kVisc = 40.0 / (Math.PI * h5);        // viscosity laplacian (2D)
    this.h2 = h2;
  };

  // Sum of the poly6 kernel W over a square lattice of unit mass points at
  // spacing `spacing`, including the self term, truncated at the kernel radius
  // h. Multiplying by mass gives the density a particle has when the fluid
  // sits exactly at its rest packing.
  Sim.prototype._latticeSum = function () {
    var s0 = this.spacing, h2 = this.h2, kw = this.kPoly6;
    var L = Math.ceil(this.h / s0);
    var sum = 0;
    for (var i = -L; i <= L; i++) {
      for (var j = -L; j <= L; j++) {
        var dx = i * s0, dy = j * s0, r2 = dx * dx + dy * dy;
        if (r2 < h2) { var t = h2 - r2; sum += kw * t * t * t; }
      }
    }
    return sum;
  };

  // Gravity direction setter. dir in {"down","left","right","up"} or a unit {x,y}.
  Sim.prototype.setGravityDirection = function (dir) {
    var g = this.gravity;
    if (typeof dir === "string") {
      switch (dir) {
        case "down":  this.gx = 0; this.gy = g; break;
        case "up":    this.gx = 0; this.gy = -g; break;
        case "left":  this.gx = -g; this.gy = 0; break;
        case "right": this.gx = g; this.gy = 0; break;
        default:      this.gx = 0; this.gy = g;
      }
    } else if (dir && typeof dir.x === "number") {
      this.gx = dir.x * g; this.gy = dir.y * g;
    }
    return this;
  };

  Sim.prototype.cycleGravity = function () {
    var order = ["down", "left", "right", "up"];
    var cur = (this.gx === 0 && this.gy > 0) ? "down"
            : (this.gx === 0 && this.gy < 0) ? "up"
            : (this.gx < 0) ? "left" : "right";
    var i = order.indexOf(cur);
    return this.setGravityDirection(order[(i + 1) % 4]);
  };

  // Dam-break: a rectangular block packed at the REST SPACING (a few % jitter
  // only, to break lattice symmetry), bottom-aligned against the floor and
  // hugging the left wall, ~45% of the box width. Seeding at rest spacing
  // means the block starts with p ~= 0 everywhere and collapses as a body.
  Sim.prototype.reset = function () {
    var N = this.N, w = this.width, h = this.height;
    var rnd = mulberry32(this.seed);
    var s0 = this.spacing, mg = this.boundMargin + s0 * 0.5;

    var cols = Math.max(1, Math.floor((w * this.blockFrac) / s0));
    var rows = Math.ceil(N / cols);
    // If the block would not fit vertically, widen it (rare; only for big N).
    var rowsMax = Math.max(1, Math.floor((h - 2 * mg) / s0));
    if (rows > rowsMax) {
      cols = Math.min(Math.ceil(N / rowsMax), Math.floor((w - 2 * mg) / s0));
      cols = Math.max(1, cols);
      rows = Math.ceil(N / cols);
    }

    var i = 0, c, r;
    for (r = 0; r < rows && i < N; r++) {
      for (c = 0; c < cols && i < N; c++) {
        this.px[i] = mg + (c + 0.5) * s0 + (rnd() - 0.5) * s0 * 0.3;
        this.py[i] = h - mg - (r + 0.5) * s0 + (rnd() - 0.5) * s0 * 0.3;
        this.vx[i] = 0; this.vy[i] = 0;
        this.ax[i] = 0; this.ay[i] = 0;
        this.rho[i] = 0; this.p[i] = 0;
        i++;
      }
    }
    this.time = 0;
    this.pointer.active = false;
    return this;
  };

  // ---------------------------------------------------------------------------
  // Change particle SIZE (resolution) mid-run. Everything physical is anchored
  // on the rest spacing, so a size change rescales the whole chain:
  //
  //   spacing s -> s'        the anchor itself (clamped to a sane range)
  //   h         -> h*r       keep h/s ~2.9 so the kernel still samples ~26
  //                          neighbours and the grid cell stays == h
  //   mass      -> m*r^2     the lattice sum scales as 1/s^2, so this keeps
  //                          rest density (and the fluid's whole feel) the
  //                          same at any resolution
  //   stiffness -> k*r^2     sound speed must scale with h to hold the same
  //                          CFL margin at the fixed dt. Consequence: finer
  //                          fluid compresses a bit more under the pool floor
  //                          (~2.5% at r=0.6) — an honest trade for stability.
  //   boundStiff-> /r        wall penetration should stay ~1 particle, and
  //                          g*h_pool/(s*bStiff) constant requires ~1/s.
  //                          omega*dt rises to ~0.23 at the finest setting:
  //                          still comfortably inside symplectic stability.
  //
  // The UI is responsible for also adjusting N (fluid VOLUME = N*s^2) and
  // calling reset() to repack the world at the new size.
  // ---------------------------------------------------------------------------
  Sim.prototype.setScale = function (spacing) {
    spacing = num(spacing, this.spacing);
    if (spacing < 4.5) spacing = 4.5;
    if (spacing > 16) spacing = 16;
    var r = spacing / this.spacing;
    if (Math.abs(r - 1) < 1e-9) return this;

    this.spacing = spacing;
    this.h *= r;
    this.mass *= r * r;
    this.stiffness *= r * r;
    this.boundStiff /= r;
    this._kernels();
    this.restDensity = this.mass * this._latticeSum();   // ~unchanged by design

    this.cell = this.h;
    this.cols = Math.max(1, Math.ceil(this.width / this.cell));
    this.rows = Math.max(1, Math.ceil(this.height / this.cell));
    this.gridHead = new Int32Array(this.cols * this.rows);
    this._nbOff = null; this._nbList = null;             // rebuilt lazily
    return this;
  };

  // ---------------------------------------------------------------------------
  // Change the particle count mid-run (the UI slider calls this).
  //
  // Growing: new particles rain in as a jittered lattice laid along the TOP of
  // the box at 1.25x rest spacing (roomy, so the fall-out is gentle rain, not
  // a pile-up explosion). Shrinking: the HIGHEST particles are dropped — they
  // are spray / free-surface, the least representative fluid — and the settled
  // body keeps its exact current state.
  //
  // Total capacity is box area / spacing^2 (~5900 at the defaults); keep well
  // under it or the pool fills the box and there is no room for anything.
  // ---------------------------------------------------------------------------
  Sim.prototype.setCount = function (n) {
    n = Math.floor(n);
    if (!(n >= 1) || n === this.N) return this;

    var i, k, fields = ["px", "py", "vx", "vy", "ax", "ay", "rho", "p"];
    var oldN = this.N, old = {};
    for (k = 0; k < fields.length; k++) old[fields[k]] = this[fields[k]];

    // Pick which old particles survive.
    var keep = null;
    if (n < oldN) {
      var py = old.py;
      var idx = new Array(oldN);
      for (i = 0; i < oldN; i++) idx[i] = i;
      idx.sort(function (a, b) { return py[a] - py[b]; });  // top of box first
      keep = idx.slice(oldN - n);                           // keep the bottom
    }

    // Reallocate the structure of arrays.
    for (k = 0; k < fields.length; k++) this[fields[k]] = new Float64Array(n);
    this.gridNext = new Int32Array(n);
    this._nbOff = null; this._nbList = null;                // resized lazily

    var kept = Math.min(n, oldN);
    for (i = 0; i < kept; i++) {
      var src = keep ? keep[i] : i;
      this.px[i] = old.px[src]; this.py[i] = old.py[src];
      this.vx[i] = old.vx[src]; this.vy[i] = old.vy[src];
      this.ax[i] = old.ax[src]; this.ay[i] = old.ay[src];
      this.rho[i] = old.rho[src]; this.p[i] = old.p[src];
    }

    // New particles: rain in from the top, left to right, row by row.
    var rnd = mulberry32((this.seed + n * 7919) | 0);
    var mg = this.boundMargin, s0 = this.spacing * 1.25;
    var perRow = Math.max(1, Math.floor((this.width - 2 * mg) / s0));
    for (i = kept; i < n; i++) {
      var a = i - kept;
      this.px[i] = mg + s0 * ((a % perRow) + 0.5) + (rnd() - 0.5) * s0 * 0.3;
      this.py[i] = mg + s0 * (Math.floor(a / perRow) + 0.5) + (rnd() - 0.5) * s0 * 0.3;
      this.vx[i] = 0; this.vy[i] = 0;
      this.ax[i] = 0; this.ay[i] = 0;
      this.rho[i] = 0; this.p[i] = 0;
    }

    this.N = this.count = n;
    return this;
  };

  // ---------------------------------------------------------------------------
  // Spatial hash grid: cell size == smoothing radius. Each particle is inserted
  // into a linked list per cell. Neighbour query walks the 3x3 cell block.
  // This is O(N) to build and O(N * k) to query (k ~ constant neighbours).
  // ---------------------------------------------------------------------------
  Sim.prototype._cellIndex = function (x, y) {
    var c = (x / this.cell) | 0;
    var r = (y / this.cell) | 0;
    if (c < 0) c = 0; else if (c >= this.cols) c = this.cols - 1;
    if (r < 0) r = 0; else if (r >= this.rows) r = this.rows - 1;
    return r * this.cols + c;
  };

  Sim.prototype._buildGrid = function () {
    var N = this.N, gridHead = this.gridHead, gridNext = this.gridNext;
    var Ncells = this.cols * this.rows;
    for (var k = 0; k < Ncells; k++) gridHead[k] = -1;
    for (var i = 0; i < N; i++) {
      var ci = this._cellIndex(this.px[i], this.py[i]);
      gridNext[i] = gridHead[ci];
      gridHead[ci] = i;
    }
  };

  // Collect neighbours of particle i (within h) into an externally provided
  // Int32Array `out`; returns the count. Used by force computation AND by the
  // brute-force-equivalence test (test.js compares this to an O(N^2) scan).
  Sim.prototype.neighbours = function (i, out) {
    var N = this.N, h = this.h, h2 = h * h;
    var px = this.px, py = this.py;
    var x0 = px[i], y0 = py[i];
    var ci = this._cellIndex(x0, y0);
    var col = (ci % this.cols) | 0;
    var row = (ci / this.cols) | 0;
    var c = this.cols, r = this.rows;
    var n = 0;
    for (var dr = -1; dr <= 1; dr++) {
      var rr = row + dr;
      if (rr < 0 || rr >= r) continue;
      for (var dc = -1; dc <= 1; dc++) {
        var cc = col + dc;
        if (cc < 0 || cc >= c) continue;
        var j = this.gridHead[rr * c + cc];
        while (j !== -1) {
          var dx = px[j] - x0, dy = py[j] - y0;
          if (dx * dx + dy * dy <= h2) out[n++] = j;
          j = this.gridNext[j];
        }
      }
    }
    return n;
  };

  // ---------------------------------------------------------------------------
  // Build the cached CSR neighbour table (nbOff / nbList) from the current grid.
  // Called once per substep after _buildGrid(); the density and force passes
  // then read from this table instead of re-walking the grid. Also exposed so
  // tests can rebuild it from the current positions and verify it.
  // ---------------------------------------------------------------------------
  Sim.prototype._buildNeighbourTable = function () {
    var N = this.N;
    var nbOff = this._nbOff, nbList = this._nbList, nbCap = this._nbCap;
    if (!nbOff || nbOff.length < N + 1) {
      nbOff = this._nbOff = new Int32Array(N + 1);
      nbList = this._nbList = new Int32Array(N * 32);   // ~26 neighbours expected
      nbCap = this._nbCap = N * 32;
    }
    var tmp = this._nbTmp;
    if (!tmp || tmp.length < N) tmp = this._nbTmp = new Int32Array(N);
    nbOff[0] = 0;
    var total = 0;
    for (var i = 0; i < N; i++) {
      var n = this.neighbours(i, tmp);
      if (total + n > nbCap) { // grow if a frame is denser than the initial estimate
        var newCap = nbCap + n + nbCap * 0.5;
        var nl = new Int32Array(newCap);
        nl.set(nbList.subarray(0, total));
        nbList = this._nbList = nl;
        nbCap = this._nbCap = newCap;
      }
      for (var a = 0; a < n; a++) nbList[total++] = tmp[a];
      nbOff[i + 1] = total;
    }
  };

  // ---------------------------------------------------------------------------
  // One fixed substep of the simulation.
  // ---------------------------------------------------------------------------
  Sim.prototype._substep = function (dt) {
    var N = this.N;
    var px = this.px, py = this.py, vx = this.vx, vy = this.vy;
    var ax = this.ax, ay = this.ay, rho = this.rho, p = this.p;
    var m = this.mass, h = this.h, h2 = this.h2;
    var kPoly6 = this.kPoly6, kSpiky = this.kSpiky, kVisc = this.kVisc;
    var stiff = this.stiffness, rho0 = this.restDensity, mu = this.viscosity;
    // Pressure floor: 0 when tension is off; -eps*k*rho0 when on. Scaled by
    // k*rho0 so a setScale() rescale keeps the physical strength proportional.
    var pFloor = -this.tension * stiff * rho0;
    var gx = this.gx, gy = this.gy;
    var bStiff = this.boundStiff, bDamp = this.boundDamp;
    var maxV = this.maxVel;
    var W = this.width, H = this.height;
    var pr = this.pointerRadius, pr2 = pr * pr, aF = this.attractForce, rF = this.repelForce;
    var pm = this.pointer.mode, pActive = this.pointer.active, pxp = this.pointer.x, pyp = this.pointer.y;

    this._buildGrid();

    // ---- Build the neighbour table ONCE (CSR) and reuse it for both the
    // density pass and the force pass. Positions don't change during a
    // substep, so one 3x3 grid walk per particle is all we need. This halves
    // the neighbour-search cost versus walking the grid in each pass.
    this._buildNeighbourTable();
    var nbOff = this._nbOff, nbList = this._nbList;
    var i, j;

    // ---- Pass 1: density (poly6) + pressure ---------------------------------
    for (i = 0; i < N; i++) {
      var x0 = px[i], y0 = py[i];
      var lo = nbOff[i], hi = nbOff[i + 1];
      var sum = 0;
      for (var a = lo; a < hi; a++) {
        j = nbList[a];
        var dx = px[j] - x0, dy = py[j] - y0;
        var r2 = dx * dx + dy * dy;
        if (r2 < h2) {
          var t = h2 - r2;
          sum += m * kPoly6 * t * t * t;
        }
      }
      rho[i] = sum;
      var pi = stiff * (rho[i] - rho0);
      // Negative-pressure floor. At tension=0 this is the classic clamp that
      // sidesteps tensile instability; at tension>0 under-dense fluid instead
      // attracts weakly and monotonically (the floor bounds the pull).
      if (pi < pFloor) pi = pFloor;
      p[i] = pi;
    }

    // ---- Pass 2: forces / accelerations -------------------------------------
    for (i = 0; i < N; i++) {
      var x0 = px[i], y0 = py[i];
      var vxi = vx[i], vyi = vy[i];
      var ri = rho[i];
      // Guard against zero density (shouldn't happen, but keep it robust).
      if (ri < 1e-6) ri = 1e-6;
      var invRi2 = 1.0 / (ri * ri);

      var fpx = 0, fpy = 0;   // pressure accel
      var fvx = 0, fvy = 0;   // viscosity accel

      var lo = nbOff[i], hi = nbOff[i + 1];
      for (var a = lo; a < hi; a++) {
        j = nbList[a];
        if (j === i) continue; // self-term: spiky grad at r=0 is 0 anyway; skip
        var dx = px[j] - x0, dy = py[j] - y0;
        var r2 = dx * dx + dy * dy;
        if (r2 >= h2 || r2 < 1e-12) continue;
        var r = Math.sqrt(r2);

        // ---- pressure (spiky gradient) ----
        var hMinusR = h - r;
        // gw = magnitude of the spiky gradient wrt i's position along r_ij,
        // positive (kSpiky > 0). The '-=' below turns it into a repulsive accel.
        var gw = kSpiky * hMinusR * hMinusR / r;
        var rj = rho[j]; if (rj < 1e-6) rj = 1e-6;
        var press = m * (p[i] * invRi2 + p[j] / (rj * rj)) * gw;
        fpx -= press * dx;
        fpy -= press * dy;

        // ---- viscosity (laplacian) ----
        var lap = kVisc * hMinusR; // positive
        fvx += mu * m * (vx[j] - vxi) / rj * lap;
        fvy += mu * m * (vy[j] - vyi) / rj * lap;
      }

      var axi = fpx + fvx + gx;
      var ayi = fpy + fvy + gy;

      // ---- pointer interaction (attract / repel) ----
      if (pActive && pm !== 0) {
        var pdx = pxp - x0, pdy = pyp - y0;
        var pd2 = pdx * pdx + pdy * pdy;
        if (pd2 < pr2 && pd2 > 1e-6) {
          var pd = Math.sqrt(pd2);
          var fall = 1.0 - pd / pr;        // linear falloff to 0 at the edge
          var pmag = fall * (pm > 0 ? aF : -rF);  // attract pulls, repel pushes
          axi += (pmag * pdx) / pd;
          ayi += (pmag * pdy) / pd;
        }
      }

      // ---- boundary spring-damper (soft walls, no hard clamp) ----
      // A particle within `boundMargin` of a wall feels a restoring spring
      // (proportional to penetration) plus a damper (proportional to its
      // velocity INTO the wall). Damping only fires for inward velocity, so it
      // removes energy without creating an outward "glue". This drains the
      // bulk slosh mode that pure SPH viscosity cannot, so the pool settles.
      var margin = this.boundMargin;
      if (x0 < margin) {
        axi += bStiff * (margin - x0);
        if (vxi < 0) axi -= bDamp * vxi;     // damp inward (leftward) velocity
      } else if (x0 > W - margin) {
        axi -= bStiff * (x0 - (W - margin));
        if (vxi > 0) axi -= bDamp * vxi;
      }
      if (y0 < margin) {
        ayi += bStiff * (margin - y0);
        if (vyi < 0) ayi -= bDamp * vyi;
      } else if (y0 > H - margin) {
        ayi -= bStiff * (y0 - (H - margin));
        if (vyi > 0) ayi -= bDamp * vyi;
      }

      ax[i] = axi;
      ay[i] = ayi;
    }

    // ---- Pass 3: semi-implicit (symplectic) Euler integration ---------------
    for (i = 0; i < N; i++) {
      var nvx = vx[i] + ax[i] * dt;
      var nvy = vy[i] + ay[i] * dt;
      // velocity clamp as a safety net against rare blow-ups
      var sp2 = nvx * nvx + nvy * nvy;
      if (sp2 > maxV * maxV) {
        var s = maxV / Math.sqrt(sp2);
        nvx *= s; nvy *= s;
      }
      vx[i] = nvx;
      vy[i] = nvy;
      px[i] += nvx * dt;
      py[i] += nvy * dt;
      // hard outer bound ONLY as a last resort: if a particle escapes the box
      // (e.g. a tunneling spike), reflect it back to keep the sim recoverable.
      // Normal operation never reaches this because the spring-damper keeps it in.
      if (px[i] < 0) { px[i] = 0; if (vx[i] < 0) vx[i] = -vx[i] * 0.5; }
      else if (px[i] > W) { px[i] = W; if (vx[i] > 0) vx[i] = -vx[i] * 0.5; }
      if (py[i] < 0) { py[i] = 0; if (vy[i] < 0) vy[i] = -vy[i] * 0.5; }
      else if (py[i] > H) { py[i] = H; if (vy[i] > 0) vy[i] = -vy[i] * 0.5; }
    }

    this.time += dt;
  };

  // Advance the sim by a real-time wall interval (seconds). Uses a fixed
  // substep dt and an accumulator so it is stable regardless of frame rate.
  // Returns the number of substeps actually taken (for fps / energy accounting).
  Sim.prototype.step = function (frameDt) {
    if (frameDt === undefined) frameDt = this.dt;
    var acc = (this._acc || 0) + Math.min(frameDt, 0.05); // clamp long stalls
    var taken = 0;
    while (acc >= this.dt && taken < 20) {
      this._substep(this.dt);
      acc -= this.dt;
      taken++;
    }
    this._acc = acc;
    return taken;
  };

  // Total kinetic energy: 0.5 * m * sum(v^2). Used by tests and the HUD.
  Sim.prototype.kineticEnergy = function () {
    var N = this.N, vx = this.vx, vy = this.vy, m = this.mass, ke = 0;
    for (var i = 0; i < N; i++) ke += 0.5 * m * (vx[i] * vx[i] + vy[i] * vy[i]);
    return ke;
  };

  // True if every particle is inside [0,W]x[0,H].
  Sim.prototype.allInBounds = function () {
    var N = this.N, px = this.px, py = this.py, W = this.width, H = this.height;
    for (var i = 0; i < N; i++) {
      if (px[i] < -1e-9 || px[i] > W + 1e-9 || py[i] < -1e-9 || py[i] > H + 1e-9) return false;
    }
    return true;
  };

  // Max speed, for diagnostics / stability.
  Sim.prototype.maxSpeed = function () {
    var N = this.N, vx = this.vx, vy = this.vy, m = 0;
    for (var i = 0; i < N; i++) {
      var s = Math.sqrt(vx[i] * vx[i] + vy[i] * vy[i]);
      if (s > m) m = s;
    }
    return m;
  };

  return Sim;
});
