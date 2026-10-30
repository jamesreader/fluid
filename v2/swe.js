/*
 * swe.js — the surf-zone shallow-water solver, CPU reference.
 *
 * The GPU runs the same scheme in fragment passes (ocean.js, FS_SWE_*);
 * this copy exists so the numerics can be tuned and tested in Node.
 *
 * Staggered grid of n x m cells, dx metres: bed b and water depth h at
 * cell centres, velocity u on x-faces (between cell i and i+1) and w on
 * z-faces. One step:
 *
 *   1. advect u, w semi-Lagrangian (trace back along the face velocity)
 *   2. h -= dt div(F), F = u * h_upwind      (mass-conserving, h >= 0)
 *   3. u -= g dt d(eta)/dx on wet faces      (eta = b + h; dry faces stop)
 *   4. bottom friction, speed limit
 *   5. relax toward a target state (the analytic swell) where `relax` > 0
 *
 * Breaking: a front is breaking where the surface drops steeply in the
 * direction the water moves and the crest stands well above still water:
 *   steep = -grad(eta) . u/|u| , crest = eta - eta_still
 * frontScore() reports both so the particle emitter (GPU) and the tests
 * use the same rule.
 */
(function (global, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else global.SWE = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var G = 9.81;
  var DRY = 0.02;          // m: below this a cell is dry
  var FRICTION = 0.003;    // quadratic bottom friction coefficient

  function Grid(n, m, dx) {
    this.n = n; this.m = m; this.dx = dx;
    var c = n * m;
    this.b = new Float32Array(c);      // bed height (m, + up)
    this.h = new Float32Array(c);      // water depth
    this.u = new Float32Array(c);      // x-face velocity (face i+1/2)
    this.w = new Float32Array(c);      // z-face velocity (face j+1/2)
    this.relax = new Float32Array(c);  // 0..1 per second-ish weight toward the target
    this.tEta = new Float32Array(c);   // target surface (m)
    this.tU = new Float32Array(c);     // target face velocities
    this.tW = new Float32Array(c);
    this._u = new Float32Array(c); this._w = new Float32Array(c); this._h = new Float32Array(c);
  }
  Grid.prototype.idx = function (i, j) { return j * this.n + i; };
  Grid.prototype.eta = function (k) { return this.b[k] + this.h[k]; };
  // Still water: surface at 0 over the bed.
  Grid.prototype.fillStill = function () { for (var k = 0; k < this.h.length; k++) this.h[k] = Math.max(0, -this.b[k]); };

  function clampI(v, a, b) { return v < a ? a : v > b ? b : v; }
  // Bilinear sample of a face field: x-faces live at (i + 1, j + 0.5) in
  // cell-corner coordinates, z-faces at (i + 0.5, j + 1).
  Grid.prototype._sample = function (f, x, z) {
    var n = this.n, m = this.m;
    var i = Math.floor(x), j = Math.floor(z), fx = x - i, fz = z - j;
    var i0 = clampI(i, 0, n - 1), i1 = clampI(i + 1, 0, n - 1), j0 = clampI(j, 0, m - 1), j1 = clampI(j + 1, 0, m - 1);
    return (f[j0 * n + i0] * (1 - fx) + f[j0 * n + i1] * fx) * (1 - fz) + (f[j1 * n + i0] * (1 - fx) + f[j1 * n + i1] * fx) * fz;
  };
  // Velocity vector at a point in cell units (cell centres at i + 0.5).
  Grid.prototype.velAt = function (x, z) {
    return [this._sample(this.u, x - 1, z - 0.5), this._sample(this.w, x - 0.5, z - 1)];
  };

  Grid.prototype.step = function (dt) {
    var n = this.n, m = this.m, dx = this.dx, b = this.b, h = this.h, u = this.u, w = this.w;
    var nu = this._u, nw = this._w, nh = this._h, i, j, k;
    // 1. advection of the face velocities
    for (j = 0; j < m; j++) for (i = 0; i < n; i++) {
      k = j * n + i;
      // x-face at (i + 1, j + 0.5)
      var vu = this.velAt(i + 1, j + 0.5);
      nu[k] = this._sample(u, i + 1 - vu[0] * dt / dx - 1, j + 0.5 - vu[1] * dt / dx - 0.5);
      var vw = this.velAt(i + 0.5, j + 1);
      nw[k] = this._sample(w, i + 0.5 - vw[0] * dt / dx - 0.5, j + 1 - vw[1] * dt / dx - 1);
    }
    // 2. height from upwind fluxes
    for (j = 0; j < m; j++) for (i = 0; i < n; i++) {
      k = j * n + i;
      var fxR = i < n - 1 ? nu[k] * (nu[k] > 0 ? h[k] : h[k + 1]) : 0;
      var fxL = i > 0 ? nu[k - 1] * (nu[k - 1] > 0 ? h[k - 1] : h[k]) : 0;
      var fzT = j < m - 1 ? nw[k] * (nw[k] > 0 ? h[k] : h[k + n]) : 0;
      var fzB = j > 0 ? nw[k - n] * (nw[k - n] > 0 ? h[k - n] : h[k]) : 0;
      nh[k] = Math.max(0, h[k] - dt / dx * (fxR - fxL + fzT - fzB));
    }
    // 3-4. pressure gradient on wet faces, friction, speed limit
    var vmax = 0.5 * dx / dt;
    for (j = 0; j < m; j++) for (i = 0; i < n; i++) {
      k = j * n + i;
      var e0 = b[k] + nh[k];
      if (i < n - 1) {
        var e1 = b[k + 1] + nh[k + 1], hf = Math.max(e0, e1) - Math.max(b[k], b[k + 1]);
        var vu2 = hf > DRY ? nu[k] - G * dt / dx * (e1 - e0) : 0;
        vu2 /= 1 + dt * FRICTION * Math.abs(vu2) / Math.max(hf, 0.05);
        nu[k] = clampI(vu2, -vmax, vmax);
      } else nu[k] = 0;
      if (j < m - 1) {
        var e2 = b[k + n] + nh[k + n], hz = Math.max(e0, e2) - Math.max(b[k], b[k + n]);
        var vw2 = hz > DRY ? nw[k] - G * dt / dx * (e2 - e0) : 0;
        vw2 /= 1 + dt * FRICTION * Math.abs(vw2) / Math.max(hz, 0.05);
        nw[k] = clampI(vw2, -vmax, vmax);
      } else nw[k] = 0;
    }
    // 5. relaxation toward the target (forcing zone)
    for (k = 0; k < n * m; k++) {
      var r = this.relax[k];
      if (r > 0) {
        var a = 1 - Math.exp(-r * dt);
        nh[k] += (Math.max(0, this.tEta[k] - b[k]) - nh[k]) * a;
        nu[k] += (this.tU[k] - nu[k]) * a;
        nw[k] += (this.tW[k] - nw[k]) * a;
      }
    }
    this.h = nh; this._h = h;
    this.u = nu; this._u = u;
    this.w = nw; this._w = w;
  };

  // Front steepness (surface drop per metre along the flow) and crest
  // height above still water, at cell (i, j).
  Grid.prototype.frontScore = function (i, j) {
    var n = this.n, m = this.m, k = j * n + i;
    if (this.h[k] < DRY || i < 1 || j < 1 || i > n - 2 || j > m - 2) return { steep: 0, crest: 0, speed: 0 };
    var v = this.velAt(i + 0.5, j + 0.5), sp = Math.hypot(v[0], v[1]);
    if (sp < 1e-3) return { steep: 0, crest: 0, speed: 0 };
    var ex = (this.eta(k + 1) - this.eta(k - 1)) / (2 * this.dx), ez = (this.eta(k + n) - this.eta(k - n)) / (2 * this.dx);
    return { steep: -(ex * v[0] + ez * v[1]) / sp, crest: this.eta(k) - Math.max(0, this.b[k]), speed: sp };
  };
  Grid.prototype.mass = function () { var s = 0; for (var k = 0; k < this.h.length; k++) s += this.h[k]; return s * this.dx * this.dx; };

  return { Grid: Grid, G: G, DRY: DRY, FRICTION: FRICTION };
});
