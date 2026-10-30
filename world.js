/*
 * world.js — the island the ocean washes against, built on the CPU.
 *
 * One 2048 m square tile (1024^2 texels, 2 m each) centred on the origin holds
 * everything: a volcanic island with white beaches and rocky headlands, a
 * turquoise lagoon dotted with coral heads, a barrier reef with sandy motus
 * that opens to the swell on the windward side, and the outer reef slope
 * falling to the deep ocean. Outside the tile the sea floor is 50 m down.
 *
 * From the heightfield it derives what the shore waves need, all on the same
 * grid:
 *   - signed distance to the coastline (+ offshore, - inland), and
 *   - swell travel time to the shore: the eikonal |grad tau| = 1 / c(depth)
 *     solved by fast sweeping, with c from the finite-depth dispersion of a
 *     9.5 s swell. Level sets of tau are the wave crests, so the crests slow,
 *     bunch up and bend around the island to meet every beach face-on
 *     (refraction for free).
 *   - local wave height: Green's-law shoaling, exposure to the swell
 *     direction, and a depth limit carried along each ray from offshore, so a
 *     wave that broke on the reef arrives in the lagoon small.
 *   - breaking stage: 0..1 steepening before the break point, 1..2 from the
 *     plunge to a fully developed bore, 2 on the swash zone of the beach.
 *
 * Pure JS (no DOM, no GL): the browser uses it as globalThis.World, Node
 * tests require it.
 */
(function (global, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else global.World = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var G = 9.81, TAU = Math.PI * 2;
  var N = 1024, SIZE = 2048, DX = SIZE / N, HALF = SIZE / 2;
  var OCEAN_FLOOR = -50;
  var SWELL_T = 9.5;                       // swell period (s)
  var SWASH_C = 2.2;                       // bore speed up the beach (m/s)
  var ISLAND_R = 360;                      // mean coast radius (m)
  var REEF_GAP = 0.62;                     // half-width (rad) of the open reef sector around WINDWARD
  var WINDWARD = 215 * Math.PI / 180;      // the surf beach faces this way
  var BAY = 65 * Math.PI / 180;            // sheltered bay on the lee side
  var POINT = 150 * Math.PI / 180;         // rocky point

  // ------------------------------------------------------------- noise
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function makeNoise(seed) {          // 2D gradient noise, roughly [-1, 1]
    var r = mulberry32(seed), perm = new Uint16Array(512), g = new Float32Array(512);
    for (var i = 0; i < 256; i++) perm[i] = i;
    for (i = 255; i > 0; i--) { var j = (r() * (i + 1)) | 0, t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    for (i = 0; i < 256; i++) { perm[i + 256] = perm[i]; var a = r() * TAU; g[2 * i] = Math.cos(a); g[2 * i + 1] = Math.sin(a); }
    function grad(ix, iz, x, z) { var h = perm[(perm[ix & 255] + iz) & 511]; return g[2 * h] * x + g[2 * h + 1] * z; }
    return function (x, z) {
      var ix = Math.floor(x), iz = Math.floor(z), fx = x - ix, fz = z - iz;
      var ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10), uz = fz * fz * fz * (fz * (fz * 6 - 15) + 10);
      var a = grad(ix, iz & 255, fx, fz), b = grad(ix + 1, iz & 255, fx - 1, fz);
      var c = grad(ix, (iz + 1) & 255, fx, fz - 1), d = grad(ix + 1, (iz + 1) & 255, fx - 1, fz - 1);
      var p = a + ux * (b - a), q = c + ux * (d - c);
      return (p + uz * (q - p)) * 1.4;
    };
  }
  function fbm(n, x, z, oct) {
    var s = 0, a = 0.5, f = 1;
    for (var i = 0; i < oct; i++) { s += a * n(x * f + i * 13.7, z * f - i * 7.3); a *= 0.5; f *= 2.03; }
    return s;
  }
  function smoothstep(a, b, x) { var t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); }
  function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
  function angDist(a, b) { var d = Math.abs(((a - b) % TAU + TAU) % TAU); return Math.min(d, TAU - d); }

  // ------------------------------------------------------------ terrain
  // Height (m, + up) and an underwater coral mask, texel by texel.
  function buildTerrain(seed) {
    seed = seed || 7;
    var nWarp = makeNoise(seed), nCoast = makeNoise(seed + 1), nHill = makeNoise(seed + 2);
    var nFloor = makeNoise(seed + 3), nReef = makeNoise(seed + 4), nTone = makeNoise(seed + 5);
    var height = new Float32Array(N * N), coral = new Float32Array(N * N), tone = new Float32Array(N * N);
    var rnd = mulberry32(seed + 99);
    // Twin peaks off centre (a steep main summit and a lower shoulder);
    // motus (sandy islets) sit on the reef ring.
    var peak = [-55, 10], peak2 = [95, -70], motus = [];
    [40, 95, 150, 300, 340].forEach(function (deg) {
      var a = deg * Math.PI / 180;
      motus.push({ a: a, len: 0.07 + rnd() * 0.06, w: 22 + rnd() * 14, h: 1.6 + rnd() * 0.8 });
    });
    // Coral heads ("bommies") scattered through the lagoon.
    var bommies = [];
    for (var b = 0; b < 140; b++) {
      var ba = rnd() * TAU, br = ISLAND_R + 70 + rnd() * 200;
      if (angDist(ba, WINDWARD) < REEF_GAP * 0.7) continue;
      bommies.push({ x: Math.cos(ba) * br, z: Math.sin(ba) * br, r: 4 + rnd() * 12, h: 1.2 + rnd() * 2.8 });
    }
    function bump(a, at, w) { var d = angDist(a, at); return Math.exp(-d * d / (2 * w * w)); }
    function coastBase(a) {
      var c = Math.cos(a), s = Math.sin(a), calm = 1 - 0.7 * bump(a, WINDWARD, 0.35);   // the surf beach runs straight
      return ISLAND_R * (1 + 0.14 * nCoast(c * 1.3 + 4, s * 1.3 + 4) + calm * (0.07 * nCoast(c * 3.1 - 2, s * 3.1 + 9) + 0.03 * nCoast(c * 7, s * 7)));
    }
    function coastR(a) { return coastBase(a) - 150 * bump(a, BAY, 0.2) + 120 * bump(a, POINT, 0.1); }
    function reefR(a) { var c = Math.cos(a), s = Math.sin(a); return coastBase(a) + 260 + 45 * nReef(c * 1.7 + 2, s * 1.7 - 3); }
    for (var j = 0; j < N; j++) {
      var z = -HALF + (j + 0.5) * DX;
      for (var i = 0; i < N; i++) {
        var x = -HALF + (i + 0.5) * DX, k = j * N + i;
        var wx = x + 55 * fbm(nWarp, x / 380, z / 380, 3), wz = z + 55 * fbm(nWarp, x / 380 + 5.2, z / 380 + 1.3, 3);
        // stretch the island along a NW-SE axis
        var ex = (wx * 0.94 - wz * 0.34) / 1.18, ez = (wx * 0.34 + wz * 0.94) / 0.9;
        var r = Math.sqrt(ex * ex + ez * ez), a = Math.atan2(wz, wx);
        var cr = coastR(a), d = cr - r;                    // ~distance inside the coast
        var c = Math.cos(a), s = Math.sin(a);
        // Rocky headlands where this noise is high: no beach, the hill meets the sea.
        // (none on the surf beach itself)
        var cliff = Math.max(smoothstep(0.3, 0.55, nCoast(c * 2.2 + 11, s * 2.2 - 6)), bump(a, POINT, 0.07)) * smoothstep(REEF_GAP * 0.5, REEF_GAP, angDist(a, WINDWARD));
        var open = 1 - smoothstep(REEF_GAP * 0.55, REEF_GAP, angDist(a, WINDWARD));   // 1 = no reef (surf side)
        var gap = smoothstep(0.42, 0.6, nReef(c * 5 + 1, s * 5 + 2));                  // channels through the reef
        var h;
        if (d > 0) {
          // land: concave beach rising to a berm, dunes, then jungle hills to the peak
          var beachW = 26 + 16 * nCoast(c * 4 + 3, s * 4 - 1);
          var beach = 2.3 * (1 - Math.exp(-d / 11)) + 0.6 * smoothstep(beachW * 0.8, beachW * 1.4, d);
          var inland = Math.max(0, d - beachW);
          var pr = Math.hypot(x - peak[0], z - peak[1]), pr2 = Math.hypot(x - peak2[0], z - peak2[1]);
          var summit = 205 * Math.exp(-Math.pow(pr / 115, 1.5)) + 95 * Math.exp(-Math.pow(pr2 / 120, 1.7));
          // fluted flanks: radial ridges and gullies cut by rain erosion
          var pa = Math.atan2(z - peak[1], x - peak[0]);
          var flute = Math.abs(Math.sin(pa * 7 + 2.5 * fbm(nHill, x / 90, z / 90, 3)));
          summit *= 1 - 0.22 * (1 - flute) * smoothstep(30, 110, pr) * (1 - smoothstep(220, 330, pr));
          var ridges = 1 - Math.abs(fbm(nHill, x / 170, z / 170, 4));
          var hills = (summit + 40 * ridges * ridges) * smoothstep(0, 170, inland) + 9 * fbm(nHill, x / 60, z / 60, 3) * smoothstep(0, 60, inland);
          h = beach + hills;
          // headland: steep rocky rise straight out of the water
          h = h * (1 - cliff) + cliff * (Math.min(d, 40) * 0.9 + 0.75 * hills + 2);
        } else {
          var e = -d;                                       // distance offshore
          var shelf = -e / (11 - 7.5 * cliff);              // beach slope under water (rocky coast: steeper)
          // lagoon floor 3.5..8 m with sand ridges
          var lagoon = -(5.2 + 1.8 * fbm(nFloor, x / 110, z / 110, 4) + 0.6 * (1 - Math.abs(nFloor(x / 35, z / 35))));
          var floorL = Math.max(shelf, lagoon);
          var rr = reefR(a), dr = r - rr;
          var crest = Math.exp(-dr * dr / (2 * 30 * 30)) * (1 - gap) * (1 - open);
          var outer = smoothstep(10, 140, dr);              // outer reef slope to the ocean
          var lagoonSide = floorL * (1 - crest) + (-0.7 + 0.4 * nReef(x / 25, z / 25)) * crest;
          var oceanH = OCEAN_FLOOR + 6 * fbm(nFloor, x / 200, z / 200, 3);
          var reefed = lagoonSide * (1 - outer) + oceanH * outer;
          // open (windward) sector: no reef, a long sandy slope with bars into deep water
          var openH = Math.max(-e * 0.05 - 0.00006 * e * e, oceanH) - 0.8 * Math.max(0, nFloor(x / 40, z / 40)) * smoothstep(20, 80, e);
          h = reefed * (1 - open) + openH * open;
          var crestCoral = crest * smoothstep(-6, -1.5, h);
          coral[k] = Math.max(crestCoral, smoothstep(5, 40, dr) * (1 - outer * 0.6) * (1 - open) * (1 - gap * 0.5));
          if (cliff > 0.3 && e < 60) coral[k] = Math.max(coral[k], smoothstep(60, 10, e) * cliff * 0.8);
        }
        // motus: sandy islets on the reef crest
        for (var m = 0; m < motus.length; m++) {
          var mo = motus[m], da = angDist(a, mo.a);
          if (da > mo.len * 2) continue;
          var along = da / mo.len, across = (r - reefR(mo.a)) / mo.w;
          var q = 1 - along * along - across * across;
          if (q > -0.6) {
            var mh = mo.h * (1 - Math.exp(-Math.max(q, 0) * 3)) + (q - 0.05) * 2.2;
            if (mh > h) { h = mh; coral[k] *= 0.3; }
          }
        }
        tone[k] = fbm(nTone, x / 30, z / 30, 3);
        height[k] = h;
      }
    }
    // coral heads in the lagoon (after the main pass so they sit on the floor)
    bommies.forEach(function (bo) {
      var i0 = Math.floor((bo.x - bo.r * 2 + HALF) / DX), i1 = Math.ceil((bo.x + bo.r * 2 + HALF) / DX);
      var j0 = Math.floor((bo.z - bo.r * 2 + HALF) / DX), j1 = Math.ceil((bo.z + bo.r * 2 + HALF) / DX);
      for (var jj = Math.max(0, j0); jj <= Math.min(N - 1, j1); jj++) {
        for (var ii = Math.max(0, i0); ii <= Math.min(N - 1, i1); ii++) {
          var px = -HALF + (ii + 0.5) * DX, pz = -HALF + (jj + 0.5) * DX, kk = jj * N + ii;
          if (height[kk] > -2.5) continue;
          var q = 1 - ((px - bo.x) * (px - bo.x) + (pz - bo.z) * (pz - bo.z)) / (bo.r * bo.r);
          if (q <= 0) continue;
          var lump = bo.h * Math.sqrt(q) * (0.8 + 0.4 * nReef(px / 3, pz / 3));
          height[kk] = Math.min(height[kk] + lump, -1.2);
          coral[kk] = Math.max(coral[kk], smoothstep(0, 0.35, q));
        }
      }
    });
    return { height: height, coral: coral, tone: tone, peak: peak };
  }

  // ------------------------------------------------- distance / travel time
  // Fast sweeping solver for |grad T| = cost / DX on the grid; cells with
  // frozen[k] keep their value. cost[k] is the time to cross one cell.
  function fastSweep(T, cost, frozen, rounds) {
    for (var it = 0; it < rounds; it++) {
      for (var sw = 0; sw < 4; sw++) {
        var di = sw & 1 ? -1 : 1, dj = sw & 2 ? -1 : 1;
        for (var jn = 0; jn < N; jn++) {
          var j = dj > 0 ? jn : N - 1 - jn, row = j * N;
          for (var in_ = 0; in_ < N; in_++) {
            var i = di > 0 ? in_ : N - 1 - in_, k = row + i;
            if (frozen[k]) continue;
            var a = Math.min(i > 0 ? T[k - 1] : Infinity, i < N - 1 ? T[k + 1] : Infinity);
            var b = Math.min(j > 0 ? T[k - N] : Infinity, j < N - 1 ? T[k + N] : Infinity);
            if (a > b) { var t = a; a = b; b = t; }
            if (a === Infinity) continue;
            var f = cost[k], v = b - a >= f ? a + f : 0.5 * (a + b + Math.sqrt(2 * f * f - (b - a) * (b - a)));
            if (v < T[k]) T[k] = v;
          }
        }
      }
    }
    return T;
  }

  // Phase speed of the swell at depth h (finite-depth dispersion).
  var C_TABLE = (function () {
    var w = TAU / SWELL_T, t = new Float32Array(512);
    for (var i = 0; i < 512; i++) {
      var h = Math.max(0.05, i * 0.1), k = w * w / G;
      for (var n = 0; n < 40; n++) {      // Newton on w^2 = g k tanh(k h)
        var th = Math.tanh(k * h), f = G * k * th - w * w, df = G * th + G * k * h * (1 - th * th);
        k -= f / df;
      }
      t[i] = w / k;
    }
    return t;
  })();
  function phaseSpeed(h) { var x = Math.min(h, 51.1) * 10, i = Math.floor(x), f = x - i; return C_TABLE[i] * (1 - f) + C_TABLE[Math.min(i + 1, 511)] * f; }
  // Green's law shoaling coefficient (ratio of group velocities), deep -> h.
  function shoaling(h) {
    var c = phaseSpeed(h), w = TAU / SWELL_T, k = w / c, kh = Math.min(k * h, 20);
    var n = 0.5 * (1 + 2 * kh / Math.sinh(2 * kh || 1e-6));
    if (kh < 1e-4) n = 1;
    var cg = n * c, cg0 = 0.5 * G / w;
    return Math.sqrt(cg0 / cg);
  }

  // Static fields that don't depend on the swell settings: land mask and
  // signed distance to the coast.
  function buildFields(terr) {
    var h = terr.height, n2 = N * N, k;
    var land = new Uint8Array(n2), water = new Uint8Array(n2);
    for (k = 0; k < n2; k++) { land[k] = h[k] >= 0 ? 1 : 0; water[k] = 1 - land[k]; }
    var unit = new Float32Array(n2).fill(DX);
    var dOff = new Float32Array(n2), dIn = new Float32Array(n2);
    for (k = 0; k < n2; k++) { dOff[k] = land[k] ? 0 : Infinity; dIn[k] = water[k] ? 0 : Infinity; }
    fastSweep(dOff, unit, land, 1);
    fastSweep(dIn, unit, water, 1);
    var sdf = new Float32Array(n2);
    for (k = 0; k < n2; k++) sdf[k] = land[k] ? -dIn[k] : dOff[k];
    return { land: land, sdf: sdf };
  }

  // Cells sorted by ascending key (bucket sort; keys are bounded).
  function orderBy(key, B) {
    var n2 = key.length, lo = Infinity, hi = -Infinity, k;
    for (k = 0; k < n2; k++) { var v = key[k]; if (v < lo) lo = v; if (v > hi) hi = v; }
    var span = (hi - lo) || 1, counts = new Uint32Array(B + 1), order = new Uint32Array(n2);
    for (k = 0; k < n2; k++) counts[Math.min(B - 1, Math.floor((key[k] - lo) / span * (B - 1)))]++;
    for (var q = 1; q <= B; q++) counts[q] += counts[q - 1];
    for (k = n2 - 1; k >= 0; k--) order[--counts[Math.min(B - 1, Math.floor((key[k] - lo) / span * (B - 1)))]] = k;
    return order;
  }

  // Swell arrival from the open ocean, for waves of height swellH heading
  // swellDeg. Deep water (> 35 m) is seeded with a plane wave; the eikonal
  // carries it over the reef, through the channels and around the motus, and
  // a second sweep runs it up the beaches as swash. Then each cell, in order
  // of arrival, inherits from its upstream neighbour how much of the wave is
  // left after breaking: a wave that broke on the reef crosses the lagoon
  // small and only shoals up again near the beach.
  function buildShore(terr, f, swellH, swellDeg) {
    var h = terr.height, n2 = N * N, k, i, j;
    var sx = Math.cos(swellDeg * Math.PI / 180), sz = Math.sin(swellDeg * Math.PI / 180);
    var cDeep = G * SWELL_T / TAU;
    var A = new Float32Array(n2), cost = new Float32Array(n2), frozen = new Uint8Array(n2);
    for (j = 0; j < N; j++) for (i = 0; i < N; i++) {
      k = j * N + i;
      var x = -HALF + (i + 0.5) * DX, z = -HALF + (j + 0.5) * DX;
      if (f.land[k]) { A[k] = Infinity; cost[k] = Infinity; frozen[k] = 1; continue; }
      var depth = -h[k];
      cost[k] = DX / phaseSpeed(Math.max(depth, 0.25));
      if (depth > 35) { A[k] = (x * sx + z * sz + 2 * HALF) / cDeep; frozen[k] = 1; } else A[k] = Infinity;
    }
    fastSweep(A, cost, frozen, 2);
    // swash: land cells arrive after their nearest wet neighbour, at bore speed
    var frozenL = new Uint8Array(n2), costL = new Float32Array(n2).fill(DX / SWASH_C);
    for (k = 0; k < n2; k++) { if (!f.land[k]) frozenL[k] = 1; else A[k] = Infinity; }
    fastSweep(A, costL, frozenL, 1);
    for (k = 0; k < n2; k++) if (!isFinite(A[k])) A[k] = 0;
    // propagation direction: up the arrival-time gradient
    var dirX = new Float32Array(n2), dirZ = new Float32Array(n2);
    for (j = 0; j < N; j++) for (i = 0; i < N; i++) {
      k = j * N + i;
      var gx = (A[j * N + Math.min(i + 1, N - 1)] - A[j * N + Math.max(i - 1, 0)]) / (2 * DX);
      var gz = (A[Math.min(j + 1, N - 1) * N + i] - A[Math.max(j - 1, 0) * N + i]) / (2 * DX);
      var l = Math.hypot(gx, gz);
      if (!(l > 0) || !isFinite(l)) { dirX[k] = sx; dirZ[k] = sz; continue; }
      dirX[k] = gx / l; dirZ[k] = gz / l;
    }
    // island shadow: land upwave blocks the swell, softened like diffraction
    var shadow = new Float32Array(n2), along = new Float32Array(n2);
    for (j = 0; j < N; j++) for (i = 0; i < N; i++) along[j * N + i] = i * sx + j * sz;
    var ord = orderBy(along, 4096);
    for (var o = 0; o < n2; o++) {
      k = ord[o]; i = k % N; j = (k / N) | 0;
      if (f.land[k]) { shadow[k] = 1; continue; }
      var pi = i - sx * 1.5, pj = j - sz * 1.5;
      if (pi < 0 || pj < 0 || pi > N - 1 || pj > N - 1) continue;
      var ii = Math.round(pi), jj = Math.round(pj);
      shadow[k] = shadow[jj * N + ii] * 0.997;
    }
    blur(shadow, 20); blur(shadow, 20);
    var H = new Float32Array(n2), stage = new Float32Array(n2);
    var ratio = new Float32Array(n2).fill(1), broke = new Float32Array(n2).fill(-1);   // arrival time of the break, -1 = not breaking
    var order = orderBy(A, 8192);
    for (o = 0; o < n2; o++) {
      k = order[o]; i = k % N; j = (k / N) | 0;
      var ui = Math.round(i - dirX[k] * 1.2), uj = Math.round(j - dirZ[k] * 1.2), up = -1;
      if (ui >= 0 && ui < N && uj >= 0 && uj < N) { up = uj * N + ui; if (!(A[up] < A[k])) up = -1; }
      if (!f.land[k]) {
        var d = Math.max(-h[k], 0.05);
        var Hu = swellH * (1 - 0.8 * shadow[k]) * shoaling(d);
        var r = up >= 0 ? ratio[up] : 1, b = up >= 0 ? broke[up] : -1;
        var Hc = r * Hu;
        if (b < 0 && Hc > 0.78 * d) b = A[k];                // break point
        var cap = b >= 0 ? 0.55 * d : 0.78 * d;              // breaker / bore height limit
        if (Hc > cap) { Hc = cap; r = Hc / Hu; }
        if (b >= 0 && Hc < 0.3 * d) b = -1;                  // reformed in deeper water
        ratio[k] = r; broke[k] = b;
        H[k] = Hc * clamp01((24 - d) / 12);
        stage[k] = b < 0 ? Math.min(Hc / (0.78 * d), 0.999) : 1 + clamp01((A[k] - b) / (0.55 * SWELL_T));
      } else {
        // the bore runs up the beach and dies within a few metres
        H[k] = up >= 0 && -f.sdf[k] < 40 ? H[up] * Math.exp(-DX / 5.5) : 0;
        stage[k] = 2;
      }
    }
    // Rays carry values cell to cell, which leaves wedge-shaped seams; a short
    // blur over the water hides them (land keeps its own swash values).
    var Hb = blur(Float32Array.from(H), 3), Sb = blur(Float32Array.from(stage), 2);
    for (k = 0; k < n2; k++) if (!f.land[k] && f.sdf[k] > 6) { H[k] = Hb[k]; stage[k] = Sb[k]; }
    // fade toward the tile edge so nothing pops at the border
    for (j = 0; j < N; j++) for (i = 0; i < N; i++) {
      var e = Math.min(i, j, N - 1 - i, N - 1 - j);
      if (e < 40) H[j * N + i] *= e / 40;
    }
    return { A: A, dirX: dirX, dirZ: dirZ, H: H, stage: stage, shadow: shadow };
  }

  // In-place separable box blur, radius r texels.
  function blur(a, r) {
    var tmp = new Float32Array(N), w = 2 * r + 1, i, j, s;
    for (j = 0; j < N; j++) {
      var row = j * N;
      s = 0;
      for (i = -r; i <= r; i++) s += a[row + Math.min(N - 1, Math.max(0, i))];
      for (i = 0; i < N; i++) {
        tmp[i] = s / w;
        s += a[row + Math.min(N - 1, i + r + 1)] - a[row + Math.max(0, i - r)];
      }
      for (i = 0; i < N; i++) a[row + i] = tmp[i];
    }
    for (i = 0; i < N; i++) {
      s = 0;
      for (j = -r; j <= r; j++) s += a[Math.min(N - 1, Math.max(0, j)) * N + i];
      for (j = 0; j < N; j++) {
        tmp[j] = s / w;
        s += a[Math.min(N - 1, j + r + 1) * N + i] - a[Math.max(0, j - r) * N + i];
      }
      for (j = 0; j < N; j++) a[j * N + i] = tmp[j];
    }
    return a;
  }

  // ------------------------------------------------------------ samplers
  function bilinear(arr, x, z, outside) {
    var u = (x + HALF) / DX - 0.5, v = (z + HALF) / DX - 0.5;
    if (u < 0 || v < 0 || u > N - 1 || v > N - 1) return outside;
    var i = Math.floor(u), j = Math.floor(v), fu = u - i, fv = v - j, k = j * N + i;
    var i1 = Math.min(i + 1, N - 1) - i, j1 = (Math.min(j + 1, N - 1) - j) * N;
    return (arr[k] * (1 - fu) + arr[k + i1] * fu) * (1 - fv) + (arr[k + j1] * (1 - fu) + arr[k + j1 + i1] * fu) * fv;
  }

  function build(opts) {
    opts = opts || {};
    var t0 = Date.now();
    var terr = buildTerrain(opts.seed);
    var t1 = Date.now();
    var f = buildFields(terr);
    var t2 = Date.now();
    var w = {
      N: N, SIZE: SIZE, DX: DX, HALF: HALF, OCEAN_FLOOR: OCEAN_FLOOR, SWELL_T: SWELL_T,
      height: terr.height, coral: terr.coral, tone: terr.tone, peak: terr.peak,
      sdf: f.sdf, land: f.land, fields: f,
      timings: { terrain: t1 - t0, fields: t2 - t1 },
      // helpers travel with the data so the renderer needs no global
      heightAt: heightAt, terrainTexture: terrainTexture, shoreTexture: shoreTexture, shoreKTexture: shoreKTexture,
    };
    setSwell(w, opts.swell == null ? 1.2 : opts.swell, opts.swellDir == null ? 35 : opts.swellDir);
    w.timings.shore = Date.now() - t2;
    return w;
  }
  function setSwell(w, swellH, swellDeg) {
    var s = buildShore(w, w.fields, swellH, swellDeg);
    w.arrival = s.A; w.dirX = s.dirX; w.dirZ = s.dirZ; w.shadow = s.shadow;
    w.waveH = s.H; w.stage = s.stage; w.swell = swellH; w.swellDir = swellDeg;
  }
  function heightAt(w, x, z) { return bilinear(w.height, x, z, OCEAN_FLOOR); }

  // GPU payloads (RGBA float rows, texel j*N+i).
  function terrainTexture(w) {
    var out = new Float32Array(N * N * 4);
    for (var k = 0; k < N * N; k++) {
      out[4 * k] = w.height[k]; out[4 * k + 1] = w.coral[k];
      out[4 * k + 2] = Math.max(-400, Math.min(400, w.sdf[k])); out[4 * k + 3] = w.tone[k];
    }
    return out;
  }
  function shoreTexture(w) {
    var out = new Float32Array(N * N * 4), kk = TAU / SWELL_T;
    for (var k = 0; k < N * N; k++) {
      var ph = w.arrival[k] * kk;
      out[4 * k] = Math.cos(ph); out[4 * k + 1] = Math.sin(ph);
      out[4 * k + 2] = w.waveH[k]; out[4 * k + 3] = w.stage[k];
    }
    return out;
  }
  // Wave vector (rad/m) of the shore swell, pointing the way it travels.
  function shoreKTexture(w) {
    var out = new Float32Array(N * N * 2);
    for (var k = 0; k < N * N; k++) {
      var depth = Math.max(-w.height[k], 0.25), c = w.land[k] ? SWASH_C : phaseSpeed(depth), kk = TAU / (SWELL_T * c);
      out[2 * k] = w.dirX[k] * kk; out[2 * k + 1] = w.dirZ[k] * kk;
    }
    return out;
  }

  return {
    N: N, SIZE: SIZE, DX: DX, HALF: HALF, OCEAN_FLOOR: OCEAN_FLOOR, SWELL_T: SWELL_T, SWASH_C: SWASH_C,
    WINDWARD: WINDWARD,
    build: build, setSwell: setSwell, heightAt: heightAt, bilinear: bilinear,
    terrainTexture: terrainTexture, shoreTexture: shoreTexture, shoreKTexture: shoreKTexture,
    phaseSpeed: phaseSpeed, shoaling: shoaling, fastSweep: fastSweep, makeNoise: makeNoise, mulberry32: mulberry32,
  };
});
