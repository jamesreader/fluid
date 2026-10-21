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
 *   (d) wind sanity: stronger wind or longer fetch makes bigger waves, and
 *       a 6 m/s lagoon breeze stays in the 0.3-0.9 m range.
 *   (e) bathymetry: tiles seamlessly, spans shallow bars to deep channels,
 *       and the CPU floor sampler agrees with the stored shape.
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
  function hs(o) { return 4 * Math.sqrt(Ocean.spectrumVariance(params(o))); }
  var base = hs({}), windy = hs({ wind: 12 }), longFetch = hs({ fetch: 150 });
  console.log("    Hs default " + base.toFixed(2) + " m, wind 12 m/s " + windy.toFixed(2) + " m, fetch 150 km " + longFetch.toFixed(2) + " m");
  if (base > 0.3 && base < 0.9) pass("(d) default breeze Hs " + base.toFixed(2) + " m");
  else fail("(d) default Hs " + base.toFixed(2) + " m outside 0.3-0.9 m");
  // growth is sub-linear here on purpose: in 4.5 m of water the TMA factor caps long waves
  if (windy > base * 1.4 && longFetch > base * 1.3) pass("(d) more wind / fetch -> bigger waves");
  else fail("(d) wave height doesn't grow with wind and fetch");
})();

/* ------------------------------------------------------------------ (e) */
console.log("--- (e) bathymetry ---");
(function () {
  var b = Ocean.makeBathymetry(256, 4242), n = b.n, p = params(), T = Ocean.BATHY_TILE;
  var seam = 0, inner = 0;
  for (var i = 0; i < n; i++) {
    seam = Math.max(seam, Math.abs(b.shape[i * n] - b.shape[i * n + n - 1]), Math.abs(b.shape[i] - b.shape[(n - 1) * n + i]));
    inner = Math.max(inner, Math.abs(b.shape[i * n + 1] - b.shape[i * n]));
  }
  if (seam <= inner * 1.5) pass("(e) tiles seamlessly (edge step " + seam.toFixed(3) + " vs interior " + inner.toFixed(3) + ")");
  else fail("(e) seam at the tile edge: " + seam.toFixed(3) + " vs interior steps " + inner.toFixed(3));
  var d = [];
  for (var z = 0; z < 64; z++) for (var x = 0; x < 64; x++) d.push(Ocean.floorDepth(b, p, x * T / 64, z * T / 64));
  d.sort(function (a, c) { return a - c; });
  var lo = d[Math.floor(d.length * 0.02)], hi = d[Math.floor(d.length * 0.98)];
  console.log("    depth 2%.." + "98%: " + lo.toFixed(1) + " .. " + hi.toFixed(1) + " m");
  if (lo < 2.5 && hi > 9 && d[0] >= 0.6) pass("(e) shallow bars and deep channels, never above 0.6 m");
  else fail("(e) depth range " + lo.toFixed(1) + ".." + hi.toFixed(1) + " m");
  var texel = T / n, k = 37 * n + 91;   // texel centre
  var want = Math.max(0.6, p.depth * Math.exp(2 * p.relief * b.shape[k]));
  var got = Ocean.floorDepth(b, p, (91 + 0.5) * texel, (37 + 0.5) * texel);
  var wrapped = Ocean.floorDepth(b, p, (91 + 0.5) * texel - 3 * T, (37 + 0.5) * texel + T);
  if (Math.abs(got - want) < 1e-9 && Math.abs(wrapped - want) < 1e-6) pass("(e) CPU sampler matches texel centres and wraps");
  else fail("(e) floorDepth " + got + " / " + wrapped + " vs " + want);
})();

console.log("");
if (PASS) {
  console.log("ALL TESTS PASSED");
  process.exit(0);
} else {
  console.log("SOME TESTS FAILED");
  process.exit(1);
}
