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
 *       6 m/s breeze stays in the 0.3-0.9 m range, and the swell adds its
 *       energy on top.
 *   (e) the island (world.js): one island with a summit and a beach band,
 *       finite shore fields, swell arriving at the surf beach in order, and a
 *       reef that keeps the lagoon calmer than the open surf beach.
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
  var both = hs({ swell: 1.2 }), expect = Math.sqrt(base * base + 1.2 * 1.2);
  console.log("    with a 1.2 m swell: Hs " + both.toFixed(2) + " m (independent seas: " + expect.toFixed(2) + " m)");
  if (Math.abs(both / expect - 1) < 0.08) pass("(d) swell adds its energy to the wind sea");
  else fail("(d) swell + wind sea Hs " + both.toFixed(2) + " m, expected about " + expect.toFixed(2) + " m");
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

console.log("");
if (PASS) {
  console.log("ALL TESTS PASSED");
  process.exit(0);
} else {
  console.log("SOME TESTS FAILED");
  process.exit(1);
}
