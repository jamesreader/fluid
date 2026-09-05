#!/usr/bin/env node
"use strict";
/*
 * test.js — quality gate for the SPH fluid sim.
 *
 * Loads sim.js (same file the browser page loads via <script src="sim.js">)
 * and asserts seven things:
 *
 *   (a) 5s settling: after a dam-break runs for 5 sim-seconds of
 *       step(1/60) frames, the kinetic energy is below the settle
 *       threshold (the pool has visibly pooled and calmed).
 *   (b) boundary: no particle ever leaves the box during the run, and the
 *       sim's own allInBounds() check passes at the end.
 *   (c) neighbour search: the spatial-hash grid result for a random
 *       100-particle sample matches a brute-force O(n^2) neighbour
 *       enumeration exactly (same neighbours, same set).
 *   (d) fluid BODY (regression guard for the "blue line" bug): after
 *       settling the pool is a body with mass — bulk density ~= rest
 *       density, the pool is many particle-spacing layers thick, and
 *       particles have a healthy number of neighbours. A sim whose
 *       particles pass through each other and pancake into a single
 *       line on the floor fails this even if it passes (a).
 *   (e) setCount(): growing the particle count mid-run (rain-in from the top)
 *       and shrinking it (drops the highest particles) both keep the sim
 *       finite and bounded, and N follows the request.
 *   (f) setScale(): rescaling particle size keeps rest density invariant,
 *       preserves the h/spacing ratio, and stays finite and bounded at the
 *       fine and coarse extremes.
 *   (g) tension: the negative-pressure floor heals cavities noticeably
 *       faster than the zero clamp, without grape-clustering the fluid.
 *
 * Exits 0 on pass, non-zero with a FAIL message otherwise.
 */

var SPHSim = require("./sim.js");

var W = 800, H = 600, N = 1500;
var KE_THRESHOLD = 3e8;   // KE ceiling after 5s of settling (see report)
var PASS = true;

function fail(msg) { console.error("FAIL: " + msg); PASS = false; }
function pass(msg) { console.log("ok:   " + msg); }

/* ------------------------------------------------------------------ (a) */
var sim = new SPHSim({ width: W, height: H, count: N });

// Run 5 sim-seconds at a 60fps frame cadence, watching bounds each frame.
var frames = 5 * 60;
var outOfBounds = 0;
var maxSpeedSeen = 0;
for (var f = 0; f < frames; f++) {
  sim.step(1 / 60);
  if (!sim.allInBounds()) { outOfBounds++; }
  if (f % 30 === 0) maxSpeedSeen = Math.max(maxSpeedSeen, sim.maxSpeed());
}
var ke5 = sim.kineticEnergy();

console.log("--- (a) 5s settling ---");
console.log("    KE after 5s   = " + ke5.toExponential(3));
console.log("    maxSpeedSeen  = " + maxSpeedSeen.toFixed(1) + " px/s");
console.log("    frames OOB    = " + outOfBounds + " / " + frames);
if (isFinite(ke5) && ke5 < KE_THRESHOLD) {
  pass("(a) KE after 5s = " + ke5.toExponential(2) + " < " + KE_THRESHOLD.toExponential(1));
} else {
  fail("(a) KE after 5s = " + ke5 + " not < " + KE_THRESHOLD + " (did not settle?)");
}

/* ------------------------------------------------------------------ (b) */
console.log("--- (b) boundary ---");
var inBoundsNow = sim.allInBounds();
// Also do an explicit per-particle sweep of the final state.
var manualOOB = 0;
for (var i = 0; i < sim.N; i++) {
  if (sim.px[i] < -1e-6 || sim.px[i] > W + 1e-6 ||
      sim.py[i] < -1e-6 || sim.py[i] > H + 1e-6) manualOOB++;
}
console.log("    final allInBounds() = " + inBoundsNow);
console.log("    manual OOB count    = " + manualOOB);
console.log("    frames OOB during   = " + outOfBounds);
if (inBoundsNow && manualOOB === 0 && outOfBounds === 0) {
  pass("(b) no particle leaves the box (final + throughout run)");
} else {
  fail("(b) boundary violation: inBounds=" + inBoundsNow +
       " manualOOB=" + manualOOB + " framesOOB=" + outOfBounds);
}

/* ------------------------------------------------------- (c) neighbours */
console.log("--- (c) neighbour search vs brute force ---");
// Fresh random sample of 100 particles, placed on a jittered grid inside the
// box with spacing (~18px) well under h=26 so each particle has several real
// neighbours within the smoothing radius (not just itself). In-bounds positions
// keep the sim's own grid state consistent.
var M = 100;
var rng = (function () { var s = 987654321; return function () {
  s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296;
}; })();
var small = new SPHSim({ width: W, height: H, count: M });
var spacing = 18;                            // < h=26 -> genuine neighbours
var ci = 0;
for (var yy = 0; yy < 10 && ci < M; yy++) {
  for (var xx = 0; xx < 10 && ci < M; xx++) {
    small.px[ci] = 100 + (xx + 0.5) * spacing + (rng() - 0.5) * 6;
    small.py[ci] = 100 + (yy + 0.5) * spacing + (rng() - 0.5) * 6;
    small.vx[ci] = 0; small.vy[ci] = 0;
    ci++;
  }
}
// Build the grid + CSR table from current positions (fresh, not stale).
small._buildGrid();
small._buildNeighbourTable();

var h2 = small.h * small.h;
var mismatches = 0, totalChecks = 0, missingTotal = 0, extraTotal = 0;
// Brute force: for each i, the set of j with r^2 = |x_j - x_i|^2 <= h^2.
// NOTE: the sim's neighbours() (and the CSR table built from it) include the
// particle ITSELF (r^2=0 <= h^2) — this is intentional, because the density
// pass sums over exactly this set (a particle's own mass contributes to its
// density). The force pass separately skips j===i. So the brute force here
// must include self to match, hence NO b2===a skip.
for (var a = 0; a < M; a++) {
  var brute = new Set();
  for (var b2 = 0; b2 < M; b2++) {
    var dx = small.px[b2] - small.px[a];
    var dy = small.py[b2] - small.py[a];
    if (dx * dx + dy * dy <= h2) brute.add(b2);
  }
  // CSR result for a.
  var off0 = small._nbOff[a], off1 = small._nbOff[a + 1];
  var gridSet = new Set();
  for (var e = off0; e < off1; e++) gridSet.add(small._nbList[e]);
  // The grid returns neighbours for a 3x3 block; the brute set uses the
  // same r^2<=h^2 test, so they must match exactly.
  var missing = 0, extra = 0;
  brute.forEach(function (j) { if (!gridSet.has(j)) missing++; });
  gridSet.forEach(function (j) { if (!brute.has(j)) extra++; });
  if (missing || extra) {
    mismatches++;
    missingTotal += missing; extraTotal += extra;
    if (mismatches <= 3) {
      console.error("    mismatch at i=" + a + " missing=" + missing + " extra=" + extra);
    }
  }
  totalChecks++;
}
console.log("    particles checked = " + totalChecks);
console.log("    mismatches        = " + mismatches + " (missing=" + missingTotal + ", extra=" + extraTotal + ")");
if (mismatches === 0) {
  pass("(c) grid neighbours == brute force on 100-particle sample (0 mismatches)");
} else {
  fail("(c) " + mismatches + "/" + totalChecks + " particles disagree with brute force");
}

/* ---------------------------------------------------------- (d) body */
console.log("--- (d) settled state is a fluid body (not a pancake line) ---");
// Reuse the 5s-settled `sim` from (a).
sim._buildGrid(); sim._buildNeighbourTable();
var rhoSum = 0, nbSum = 0;
var ys = new Float64Array(sim.N);
for (var i2 = 0; i2 < sim.N; i2++) {
  rhoSum += sim.rho[i2];
  nbSum += sim._nbOff[i2 + 1] - sim._nbOff[i2];
  ys[i2] = sim.py[i2];
}
var rhoRel = rhoSum / sim.N / sim.restDensity;
var nbRel = nbSum / sim.N;
var ysSorted = Array.prototype.slice.call(ys).sort(function (a, b) { return a - b; });
var thickness = ysSorted[Math.floor(sim.N * 0.9)] - ysSorted[Math.floor(sim.N * 0.1)];
console.log("    rhoAvg/rho0    = " + rhoRel.toFixed(3) + "   (want 0.8 .. 1.5)");
console.log("    neighbours avg = " + nbRel.toFixed(1) + "   (want 8 .. 40)");
console.log("    pool thickness = " + thickness.toFixed(1) + " px   (want >= " + 8 * sim.spacing + ")");
if (rhoRel >= 0.8 && rhoRel <= 1.5 && nbRel >= 8 && nbRel <= 40 && thickness >= 8 * sim.spacing) {
  pass("(d) settled pool is a fluid BODY: rho~rho0, " + nbRel.toFixed(0) +
       " neighbours, " + thickness.toFixed(0) + "px thick");
} else {
  fail("(d) not a fluid body: rhoRel=" + rhoRel.toFixed(2) + " nb=" + nbRel.toFixed(1) +
       " thickness=" + thickness.toFixed(1) + " (particles overlapping/pancaked?)");
}

/* --------------------------------------------------- (e) setCount resize */
console.log("--- (e) mid-run particle count changes ---");
var sc = new SPHSim({ width: W, height: H, count: 1500 });
for (var g0 = 0; g0 < 120; g0++) sc.step(1 / 60);           // 2s, block collapsing
sc.setCount(3000);                                           // rain in +1500
for (var g1 = 0; g1 < 240; g1++) sc.step(1 / 60);           // 4s absorbing the rain
var growOK = (sc.N === 3000) && sc.allInBounds();
for (var i4 = 0; growOK && i4 < sc.N; i4++) {
  if (!isFinite(sc.px[i4]) || !isFinite(sc.py[i4]) || !isFinite(sc.vx[i4])) growOK = false;
}
var keAfterGrow = sc.kineticEnergy();
var nAfterGrow = sc.N;
sc.setCount(600);                                            // drop the highest
for (var g2 = 0; g2 < 180; g2++) sc.step(1 / 60);           // 3s re-settle
var shrinkOK = (sc.N === 600) && sc.allInBounds();
for (var i5 = 0; shrinkOK && i5 < sc.N; i5++) {
  if (!isFinite(sc.px[i5]) || !isFinite(sc.py[i5]) || !isFinite(sc.vx[i5])) shrinkOK = false;
}
console.log("    grow 1500->3000: N=" + nAfterGrow + " finite+bounded=" + growOK +
            " KE=" + keAfterGrow.toExponential(1));
if (growOK) pass("(e) grow: rain-in stayed finite and in bounds");
else fail("(e) grow: unstable or escaped after 1500->3000");
if (shrinkOK) pass("(e) shrink: 600 survivors stayed finite and in bounds");
else fail("(e) shrink: unstable or escaped after ->600");

/* ----------------------------------------------------- (f) setScale size */
console.log("--- (f) particle size rescale ---");
var sz = new SPHSim({ width: W, height: H, count: 1500 });
var rho0 = sz.restDensity, ratioOK = true;
sz.setScale(5.5);
ratioOK = Math.abs(sz.h / sz.spacing - 26 / 9) < 1e-6;
var rhoHold = sz.restDensity > 0.95 * rho0 && sz.restDensity < 1.05 * rho0;
for (var h0 = 0; h0 < 240; h0++) sz.step(1 / 60);
var fineOK = ratioOK && rhoHold && sz.allInBounds();
for (var i7 = 0; fineOK && i7 < sz.N; i7++) {
  if (!isFinite(sz.px[i7]) || !isFinite(sz.py[i7])) fineOK = false;
}
sz.setScale(16);                                   // coarse extreme
for (var h1 = 0; h1 < 180; h1++) sz.step(1 / 60);
var coarseOK = sz.allInBounds() && Math.abs(sz.h / sz.spacing - 26 / 9) < 1e-6;
for (var i8 = 0; coarseOK && i8 < sz.N; i8++) {
  if (!isFinite(sz.px[i8]) || !isFinite(sz.py[i8])) coarseOK = false;
}
console.log("    fine (5.5px): rho0 preserved=" + rhoHold + " h/s kept=" + ratioOK + " stable+bounded=" + fineOK);
if (fineOK) pass("(f) fine rescale: same density, same sampling, stable");
else fail("(f) fine rescale broke something (rho=" + rhoHold + " ratio=" + ratioOK + " bounds=" + fineOK + ")");
if (coarseOK) pass("(f) coarse rescale: stable and bounded");
else fail("(f) coarse rescale unstable or escaped");

/* ----------------------------------------------------- (g) tension floor */
console.log("--- (g) surface tension heals cavities ---");
function carveAndCount(tension, markFrames) {
  var s = new SPHSim({ width: W, height: H, count: 1500, tension: tension });
  for (var f = 0; f < 300; f++) s.step(1 / 60);          // settle 5s
  var cx = 0, cy = 0, i;
  for (i = 0; i < s.N; i++) { cx += s.px[i]; cy += s.py[i]; }
  cx /= s.N; cy /= s.N;
  var order = new Uint32Array(s.N);
  for (i = 0; i < s.N; i++) order[i] = i;
  order.sort(function (a, b) {                            // deterministic: by distance to centroid
    var da = (s.px[a] - cx) * (s.px[a] - cx) + (s.py[a] - cy) * (s.py[a] - cy);
    var db = (s.px[b] - cx) * (s.px[b] - cx) + (s.py[b] - cy) * (s.py[b] - cy);
    return da - db;
  });
  for (var q = 0; q < 90; q++) {                          // teleport a plug away: cavity left behind
    var k2 = order[q];
    s.px[k2] = 40 + (q % 30) * 11.25; s.py[k2] = 30 + ((q / 30) | 0) * 11.25;
    s.vx[k2] = 0; s.vy[k2] = 0;
  }
  var out = { iso: 0, rhoAvg: 0, rhoMax: 0, bounds: true };
  for (var t = 0; t < markFrames; t++) {
    s.step(1 / 60);
    s._buildGrid(); s._buildNeighbourTable();
    var iso = 0, rhoSum = 0, rhoMax = 0;
    for (i = 0; i < s.N; i++) {
      if (s._nbOff[i + 1] - s._nbOff[i] <= 5 && s.rho[i] < 0.6 * s.restDensity) iso++;
      rhoSum += s.rho[i];
      if (s.rho[i] > rhoMax) rhoMax = s.rho[i];
    }
    if (t === markFrames - 1) {
      out.iso = iso; out.rhoAvg = rhoSum / s.N / s.restDensity;
      out.rhoMax = rhoMax / s.restDensity; out.bounds = s.allInBounds();
    }
  }
  return out;
}
var noT = carveAndCount(0, 90), withT = carveAndCount(SPHSim.TENSION_EPS, 90);
console.log("    at +1.5s   off: iso=" + noT.iso + " rhoMax=" + noT.rhoMax.toFixed(2) +
            "   on:  iso=" + withT.iso + " rhoMax=" + withT.rhoMax.toFixed(2));
if (withT.iso < noT.iso && withT.iso <= 2) pass("(g) tension heals cavities faster than the zero clamp");
else fail("(g) tension did not improve cavity healing (off=" + noT.iso + " on=" + withT.iso + ")");
if (withT.bounds && withT.rhoAvg > 0.85 && withT.rhoAvg < 1.25 && withT.rhoMax < 2.0)
  pass("(g) no grape clustering at eps=" + SPHSim.TENSION_EPS);
else fail("(g) grape clustering or instability (rhoAvg=" + withT.rhoAvg.toFixed(2) +
          " rhoMax=" + withT.rhoMax.toFixed(2) + " bounds=" + withT.bounds + ")");

/* ------------------------------------------------------------------ */
console.log("");
if (PASS) {
  console.log("ALL TESTS PASSED");
  process.exit(0);
} else {
  console.error("SOME TESTS FAILED");
  process.exit(1);
}
