#!/usr/bin/env node
"use strict";
/*
 * ocean-test.js — checks for the CPU side of the 3D ocean (ocean.js).
 *
 * The GPU passes can't run under Node, but everything they are fed can:
 *
 *   (a) FFT index math: the Stockham gather the fragment shader uses
 *       (exported as fftStockham) equals a direct inverse DFT.
 *   (b) spectrum energy: summing the discretised directional spectrum over
 *       the three cascades' wavevector grids reproduces the 1D JONSWAP(+TMA)
 *       variance, i.e. the dw/dk and 1/k Jacobians and the cascade hand-over
 *       bands neither lose nor double-count waves (Hs within 5%).
 *   (c) Hermitian packing: h0 texels carry conj(h0(-k)) of their mirror, so
 *       every field the GPU synthesises is real.
 *   (d) wind sanity: stronger wind or longer fetch makes bigger waves, a
 *       6 m/s breeze stays in the 0.3-0.9 m range, and the optional swell adds
 *       energy that the grid still integrates correctly.
 *   (e) the island (world.js): one island with a summit and a beach band,
 *       finite shore fields, swell arriving at the surf beach in order, and a
 *       reef that keeps the lagoon calmer than the open surf beach.
 *   (f) reef life (life.js): coral and animal meshes are valid; fish stay
 *       finite, under water and off the reef, and scatter from a diver.
 *   (g) play.js: floating objects settle at their Archimedes draught and
 *       drift with the swell, rocks sink, and the boat planes, turns,
 *       leaves a wake and throws spray.
 *   (g2) the surfboard paddles to ~2 m/s, a breaker-size wave picks it up,
 *       the rider stands and rides the face faster than 5 m/s.
 *
 * Exits 0 on pass, non-zero with a FAIL message otherwise.
 */

var Ocean = require("./ocean.js");

var PASS = true;
function fail(msg) { console.error("FAIL: " + msg); PASS = false; }
function pass(msg) { console.log("ok:   " + msg); }
function params(o) { return Object.assign({}, Ocean.DEFAULTS, o || {}); }

/* ------------------------------------------------------------------ (a) */
console.log("--- (a) Stockham FFT vs direct DFT ---");
[8, 64, 256].forEach(function (N) {
  var re = [], im = [];
  for (var i = 0; i < N; i++) { re.push(Math.sin(i * 1.7) + 0.3 * Math.cos(i * i)); im.push(Math.cos(i * 0.3) - 0.2 * Math.sin(3 * i)); }
  var out = Ocean.fftStockham(re, im), err = 0;
  for (var n = 0; n < N; n++) {
    var sr = 0, si = 0;
    for (var k = 0; k < N; k++) {
      var a = 2 * Math.PI * k * n / N;
      sr += re[k] * Math.cos(a) - im[k] * Math.sin(a);
      si += re[k] * Math.sin(a) + im[k] * Math.cos(a);
    }
    err = Math.max(err, Math.abs(sr - out[0][n]), Math.abs(si - out[1][n]));
  }
  if (err < 1e-9) pass("(a) N=" + N + " max error " + err.toExponential(1));
  else fail("(a) N=" + N + " Stockham disagrees with the DFT by " + err);
});

/* ------------------------------------------------------------------ (b) */
console.log("--- (b) spectrum energy ---");
[[6, 25, 4.5], [3, 10, 4], [12, 100, 8]].forEach(function (c) {
  var p = params({ wind: c[0], fetch: c[1], depth: c[2] });
  var hsGrid = 4 * Math.sqrt(Ocean.spectrumVariance(p)), hs1D = 4 * Math.sqrt(Ocean.spectrumM0(p));
  var r = hsGrid / hs1D;
  console.log("    U=" + c[0] + " m/s F=" + c[1] + " km h=" + c[2] + " m: Hs grid " + hsGrid.toFixed(3) + " m, 1D " + hs1D.toFixed(3) + " m");
  if (r > 0.95 && r < 1.05) pass("(b) Hs ratio " + r.toFixed(3));
  else fail("(b) Hs from the cascades is " + r.toFixed(3) + "x the 1D integral at U=" + c[0]);
});

/* ------------------------------------------------------------------ (c) */
console.log("--- (c) Hermitian packing ---");
[0, 1, 2].forEach(function (c) {
  var N = Ocean.FFT_N, xi = Ocean.gaussians(N * N, 1234 + c), worst = 0, nonzero = 0;
  var s = Ocean.buildSpectrum(c, params(), xi).data;
  for (var m = 0; m < N; m++) {
    for (var n = 0; n < N; n++) {
      var j = (m * N + n) * 4, jm = (((N - m) % N) * N + ((N - n) % N)) * 4;
      if (s[j] || s[j + 1]) nonzero++;
      worst = Math.max(worst, Math.abs(s[j + 2] - s[jm]), Math.abs(s[j + 3] + s[jm + 1]));
    }
  }
  // cascade 1's band (9 m..3 m waves on a 36 m tile) is a narrow ring of ~400 modes
  if (worst === 0 && nonzero > 300) pass("(c) cascade " + c + ": " + nonzero + " modes, each texel holds conj(h0(-k)) of its mirror");
  else fail("(c) cascade " + c + " packing mismatch " + worst + " (" + nonzero + " modes)");
});

/* ------------------------------------------------------------------ (d) */
console.log("--- (d) wind sanity ---");
(function () {
  // wind sea alone (swell 0), then the swell on top
  function hs(o) { return 4 * Math.sqrt(Ocean.spectrumVariance(params(Object.assign({ swell: 0 }, o)))); }
  var base = hs({}), windy = hs({ wind: 12 }), longFetch = hs({ fetch: 150 });
  // the optional FFT swell (shallow-water TMA damps it at the 4.5 m spectrum depth)
  var ps = params({ swell: 1.2 }), both = 4 * Math.sqrt(Ocean.spectrumVariance(ps)), expect = 4 * Math.sqrt(Ocean.spectrumM0(ps));
  console.log("    with a 1.2 m swell: Hs " + both.toFixed(2) + " m (1D integral " + expect.toFixed(2) + " m, wind sea alone " + base.toFixed(2) + " m)");
  if (Math.abs(both / expect - 1) < 0.05 && both > base * 1.1) pass("(d) the swell adds energy and the grid matches the 1D integral");
  else fail("(d) swell + wind sea Hs " + both.toFixed(2) + " m vs 1D " + expect.toFixed(2) + " m");
  console.log("    Hs default " + base.toFixed(2) + " m, wind 12 m/s " + windy.toFixed(2) + " m, fetch 150 km " + longFetch.toFixed(2) + " m");
  if (base > 0.3 && base < 0.9) pass("(d) default breeze Hs " + base.toFixed(2) + " m");
  else fail("(d) default Hs " + base.toFixed(2) + " m outside 0.3-0.9 m");
  if (windy > base * 1.4 && longFetch > base * 1.3) pass("(d) more wind / fetch -> bigger waves");
  else fail("(d) wave height doesn't grow with wind and fetch");
})();

/* ------------------------------------------------------------------ (e) */
console.log("--- (e) the island ---");
(function () {
  var World = require("./world.js");
  var t0 = Date.now(), w = World.build({ swell: 1.2, swellDir: 35 });
  console.log("    built in " + (Date.now() - t0) + " ms " + JSON.stringify(w.timings));
  var N = w.N, land = 0, top = -1e9, beach = 0, finite = true;
  for (var k = 0; k < N * N; k++) {
    var h = w.height[k];
    if (h >= 0) land++;
    if (h > top) top = h;
    if (h > 0 && h < 2.5) beach++;
    if (!isFinite(w.arrival[k]) || !isFinite(w.waveH[k]) || !isFinite(w.stage[k])) finite = false;
  }
  var landKm2 = land * w.DX * w.DX / 1e6;
  console.log("    land " + landKm2.toFixed(2) + " km^2, summit " + top.toFixed(0) + " m, beach band " + (beach * w.DX * w.DX / 1e4).toFixed(1) + " ha");
  if (landKm2 > 0.2 && landKm2 < 1.5 && top > 150 && beach > 1000) pass("(e) one island with a summit and a beach band");
  else fail("(e) island shape off: land " + landKm2 + " km^2, summit " + top);
  if (finite) pass("(e) shore fields are finite everywhere");
  else fail("(e) NaN/Inf in arrival, wave height or stage");
  if (World.heightAt(w, 5000, 5000) === w.OCEAN_FLOOR) pass("(e) outside the tile is open ocean floor");
  else fail("(e) heightAt outside the tile is " + World.heightAt(w, 5000, 5000));
  // walk in from the open sea to the surf beach (windward side)
  function at(arr, x, z) { return World.bilinear(arr, x, z, 0); }
  var ax = Math.cos(World.WINDWARD), az = Math.sin(World.WINDWARD), prev = -Infinity, monotone = true, shore = 0;
  for (var r = 900; r > 0; r -= 4) {
    var x = ax * r, z = az * r;
    if (World.heightAt(w, x, z) >= 0) { shore = r; break; }
    var A = at(w.arrival, x, z);
    if (A < prev - 0.5) monotone = false;
    prev = A;
  }
  if (monotone && shore > 0) pass("(e) swell reaches the surf beach later the closer it gets (shore at r=" + shore + " m)");
  else fail("(e) arrival time not increasing toward the surf beach");
  // biggest surf on the open side, small waves inside the reef
  var surf = 0, lagoon = 0;
  for (r = shore + 200; r > shore; r -= 2) surf = Math.max(surf, at(w.waveH, ax * r, az * r));
  var la = World.WINDWARD + Math.PI;
  for (r = 380; r < 600; r += 2) { if (World.heightAt(w, Math.cos(la) * r, Math.sin(la) * r) > -2) continue; lagoon = Math.max(lagoon, at(w.waveH, Math.cos(la) * r, Math.sin(la) * r)); }
  console.log("    max wave height: surf beach " + surf.toFixed(2) + " m, lee lagoon " + lagoon.toFixed(2) + " m");
  if (surf > 0.8 && lagoon < 0.5 * surf) pass("(e) reef shelters the lagoon (" + lagoon.toFixed(2) + " m vs " + surf.toFixed(2) + " m surf)");
  else fail("(e) surf " + surf.toFixed(2) + " m vs lagoon " + lagoon.toFixed(2) + " m");
})();

/* ------------------------------------------------------------------ (f) */
console.log("--- (f) reef life ---");
(function () {
  var World = require("./world.js"), Life = require("./life.js");
  var w = World.build({});
  var meshes = Object.assign({}, World.coralMeshes(7), { fish: World.fishMesh(), turtle: World.turtleMesh(), ray: World.rayMesh() });
  var badMesh = Object.keys(meshes).filter(function (k) {
    var m = meshes[k], nv = m.vertices.length / 9;
    for (var i = 0; i < m.vertices.length; i++) if (!isFinite(m.vertices[i])) return true;
    for (i = 0; i < m.indices.length; i++) if (m.indices[i] >= nv) return true;
    return false;
  });
  if (badMesh.length === 0) pass("(f) " + Object.keys(meshes).length + " reef meshes are finite with valid indices");
  else fail("(f) broken meshes: " + badMesh.join(", "));
  var e = new Life.Ecosystem(w, 1), cam = [392, -2.6, 304], above = 0, under = 0, bad = 0;
  for (var f = 0; f < 600; f++) {
    e.update(1 / 30, cam);
    e.schools.forEach(function (s) {
      if (!s.alive) return;
      for (var i = 0; i < s.n; i++) {
        var x = s.pos[i * 3], y = s.pos[i * 3 + 1], z = s.pos[i * 3 + 2];
        if (!isFinite(x + y + z)) bad++;
        if (y > -0.29) above++;
        if (y < World.heightAt(w, x, z) + 0.07) under++;
      }
    });
  }
  console.log("    " + e.fishCount() + " fish in " + e.schools.filter(function (s) { return s.alive; }).length + " schools, " +
    e.turtles.filter(function (t) { return t.alive; }).length + " turtles, " + e.rays.filter(function (r) { return r.alive; }).length + " rays");
  if (bad === 0 && above === 0 && under === 0 && e.fishCount() > 100) pass("(f) fish stay finite, under the surface and above the reef for 20 s");
  else fail("(f) fish: " + bad + " non-finite, " + above + " above surface, " + under + " inside the reef");
  // a diver swimming into a school scatters it
  var s = e.schools.filter(function (q) { return q.alive && q.n > 30; })[0], c = [0, 0, 0];
  for (var i = 0; i < s.n; i++) { c[0] += s.pos[i * 3] / s.n; c[1] += s.pos[i * 3 + 1] / s.n; c[2] += s.pos[i * 3 + 2] / s.n; }
  function spread() { var d = 0; for (var j = 0; j < s.n; j++) d += Math.hypot(s.pos[j * 3] - c[0], s.pos[j * 3 + 1] - c[1], s.pos[j * 3 + 2] - c[2]); return d / s.n; }
  var d0 = spread();
  for (f = 0; f < 30; f++) e.update(1 / 30, c);
  var d1 = spread();
  if (d1 > d0 + 1) pass("(f) a school scatters from the diver (" + d0.toFixed(1) + " -> " + d1.toFixed(1) + " m in 1 s)");
  else fail("(f) school did not scatter: " + d0.toFixed(2) + " -> " + d1.toFixed(2) + " m");
})();

/* ------------------------------------------------------------------ (g) */
console.log("--- (g) floating things and the boat ---");
(function () {
  var World = require("./world.js"), Play = require("./play.js");
  var w = World.build({});
  // the cap-volume formula integrates to the sphere
  var r = 0.3, n = 2000, v = 0;
  for (var i = 0; i < n; i++) { var y = -r + (i + 0.5) / n * 2 * r; v += Math.PI * (r * r - y * y) * (2 * r / n); }
  var vf = v / (4 / 3 * Math.PI * r * r * r), cf = Play.capFraction(2 * r, r);
  if (Math.abs(vf - 1) < 1e-3 && Math.abs(cf - 1) < 1e-9 && Math.abs(Play.capFraction(r, r) - 0.5) < 1e-9) pass("(g) submerged-cap volume is right (half at the equator, all at the top)");
  else fail("(g) capFraction " + cf + " / " + Play.capFraction(r, r));
  // drop each kind into flat water: it should settle at its Archimedes draught
  var pg = new Play.Playground(w), t = 0, x0 = 300, z0 = 330;
  Play.KINDS.forEach(function (K, k) { pg.throwObject(k, [x0 + k * 2, 1.5, z0], [0, 0, 0]); });
  function sea(x, z, t) { return 0.4 * Math.sin(0.35 * x - 1.85 * t); }
  for (var f = 0; f < 900; f++) {
    t += 1 / 60;
    var pts = pg.probePoints();
    pg.setProbes(pts.map(function (p) { return t < 6 ? 0 : sea(p[0], p[1], t); }));
    pg.update(1 / 60, t, [x0, 2, z0], 0);
    if (Math.abs(t - 5.9) < 1 / 120) {
      var ok = pg.objects.slice(0, 2).every(function (o) { return Math.abs(o.wet - o.K.density / 1025) < 0.04; });
      console.log("    settled wet fractions " + pg.objects.map(function (o) { return o.K.name + " " + o.wet.toFixed(3) + " (expect " + Math.min(1, o.K.density / 1025).toFixed(3) + ")"; }).join(", "));
      if (ok && pg.objects[2].pos[1] < -2) pass("(g) ball and coconut float at their Archimedes draught; the rock sinks");
      else fail("(g) floating draughts off");
    }
  }
  var drift = pg.objects[0].pos[0] - x0;
  console.log("    beach ball drift after 9 s of 0.8 m swell: " + drift.toFixed(2) + " m");
  if (drift > 0.3) pass("(g) the swell carries a floating ball along with it");
  else fail("(g) ball did not drift with the waves (" + drift.toFixed(2) + " m)");
  // probe slots belong to objects for life: the 17th throw must not shuffle them
  var pq = new Play.Playground(w);
  for (var q = 0; q < Play.MAX_OBJ + 1; q++) pq.throwObject(1, [q * 10, 1, 0], [0, 0, 0]);
  var qp = pq.probePoints(), slots = pq.objects.map(function (o) { return o.slot; });
  var own = pq.objects.every(function (o) { var p0 = qp[1 + 3 * o.slot]; return p0[0] === o.pos[0] && p0[1] === o.pos[2]; });
  if (own && new Set(slots).size === slots.length) pass("(g) each object keeps its own water probe when the oldest is replaced");
  else fail("(g) probe slots shuffled: " + slots.join(","));
  // boat: planes up to speed, turns, stays finite, throws spray and wake
  var b = pg.spawnBoat(280, 300, 0), maxV = 0, finite = true, drops = 0, spray = 0;
  b.throttle = 1;
  for (f = 0; f < 900; f++) {
    t += 1 / 60;
    var pts2 = pg.probePoints();
    pg.setProbes(pts2.map(function () { return 0; }));
    if (f === 600) b.steer = 1;
    pg.update(1 / 60, t, b.pos, 0);
    drops += pg.takeDrops().length;
    maxV = Math.max(maxV, Math.hypot(b.vel[0], b.vel[2]));
    spray = Math.max(spray, pg.pN);
    if (![b.pos[0], b.pos[1], b.pos[2], b.pitch, b.roll, b.hdg].every(isFinite)) finite = false;
  }
  console.log("    boat top speed " + (maxV * 1.944).toFixed(0) + " kn, heading after turning " + (b.hdg * 180 / Math.PI).toFixed(0) + " deg, wake drops " + drops + ", spray up to " + spray + " drops in the air");
  if (finite && maxV > 8 && maxV < 22 && b.hdg > 0.5 && drops > 1000 && spray > 50) pass("(g) the boat planes, turns, leaves a wake and throws spray");
  else fail("(g) boat: finite " + finite + ", top " + maxV.toFixed(1) + " m/s, heading " + b.hdg.toFixed(2) + ", drops " + drops + ", spray " + pg.pN);
})();

// --------------------------------------------------------------- (g2)
(function () {
  console.log("--- (g2) the surfboard ---");
  var Play = require("./play.js");
  var pg = new Play.Playground({ heightAt: function () { return -6; } }, function () { return -6; });
  // a 1.5 m breaker-steep hump running at 6.5 m/s through 6 m of water
  function eta(x, t) { var u = (x - (6.5 * t - 30)) / 4.5; return 1.5 * Math.exp(-u * u); }
  pg.spawnBoard(0, 0, 0);
  var t = 0, dt = 1 / 60, paddle2 = 0, stoodAt = -1, top = 0, finite = true;
  for (var i = 0; i < 60 * 12; i++) {
    var pts = pg.probePoints(), H = new Float32Array(Play.MAX_PROBES);
    for (var k = 0; k < Play.MAX_PROBES; k++) H[k] = eta(pts[k][0], t);
    pg.setProbes(H);
    pg.updateBoard(dt, pg.board.standing ? 0 : 1, 0);
    t += dt;
    var b = pg.board, sp = Math.hypot(b.vel[0], b.vel[2]);
    if (t < 3) paddle2 = sp;
    if (b.standing && stoodAt < 0) stoodAt = t;
    if (b.standing) top = Math.max(top, sp);
    if (![b.pos[0], b.pos[1], b.pos[2], b.pitch, b.roll, b.hdg].every(isFinite)) finite = false;
  }
  console.log("    paddling " + paddle2.toFixed(1) + " m/s after 3 s, up on the wave at " + stoodAt.toFixed(1) + " s, riding up to " + top.toFixed(1) + " m/s");
  if (finite && paddle2 > 1.5 && paddle2 < 3 && stoodAt > 0 && top > 5) pass("(g2) the board paddles, catches the wave, stands and rides it");
  else fail("(g2) board: finite " + finite + ", paddle " + paddle2 + ", stood " + stoodAt + ", top " + top);
})();

console.log("");
if (PASS) {
  console.log("ALL TESTS PASSED");
  process.exit(0);
} else {
  console.log("SOME TESTS FAILED");
  process.exit(1);
}
