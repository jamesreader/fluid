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
          // foreshore about 1:11 at the waterline, flattening to a berm
          var beach = 2.4 * (1 - Math.exp(-d / 26)) + 0.6 * smoothstep(beachW * 1.2, beachW * 2.0, d);
          var inland = Math.max(0, d - beachW * 1.6);
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
    // The reef break: a coral ledge running out from the end of the barrier
    // reef at ~45 degrees to the incoming crests, shallow on top, with deep
    // water right up to its up-wave face. A wave catches the outer tip first
    // and the break peels down the ledge toward the beach.
    var reef = reefBreakFrame();
    for (var jj2 = 0; jj2 < N; jj2++) {
      var pz = -HALF + (jj2 + 0.5) * DX;
      for (var ii2 = 0; ii2 < N; ii2++) {
        var px = -HALF + (ii2 + 0.5) * DX, kk2 = jj2 * N + ii2;
        var rx = px - reef.p0[0], rz = pz - reef.p0[1];
        var along = rx * reef.dir[0] + rz * reef.dir[1], across = rx * reef.side[0] + rz * reef.side[1];
        if (along < -90 || along > reef.len + 70 || Math.abs(across) > 130) continue;
        var t = clamp01(along / reef.len);
        // a gently bowed, uneven ledge: the centre line wanders, the width
        // swells mid-way, the edges are ragged at a few metres
        var centre = 14 * Math.sin(t * Math.PI * 1.15) + 7 * nReef(along / 55 + 3, 1.7);
        var half = (11 + 11 * Math.pow(Math.sin(Math.PI * Math.min(1, t * 1.05 + 0.04)), 0.7) + 4 * nReef(along / 22 - 5, 4.1)) * smoothstep(-85, 15, along);   // tapers to a point at the inner end
        var ac = across - centre, ed = Math.abs(ac) - half + 3.5 * nReef(px / 7 + 2, pz / 7 - 3) + 1.5 * nReef(px / 2.5, pz / 2.5);
        var top = -(0.9 + 1.6 * t + 0.3 * nReef(px / 9, pz / 9) + 0.15 * nReef(px / 3, pz / 3));   // reef top deepens toward the tip
        // steep face on the up-wave side, gentler back toward the lagoon
        var face = ac > 0 ? 0.8 : 0.24;
        var hr = top - Math.max(0, ed) * face;
        // rounded tip; the inner end runs into the beach shelf
        var tipD = Math.max(0, along - reef.len);
        hr -= tipD * tipD / 140;
        var endFade = (1 - smoothstep(reef.len + 25, reef.len + 70, along)) * smoothstep(-90, -30, along);
        hr = hr * endFade + height[kk2] * (1 - endFade);
        // deep water hugging the face: the swell arrives unbroken
        var trench = ac > half ? smoothstep(half, half + 14, ac) * (1 - smoothstep(half + 40, half + 95, ac)) * endFade * smoothstep(0, 40, along) : 0;
        var deep = Math.min(height[kk2], -7 - 3 * t);
        var base = height[kk2] * (1 - trench) + deep * trench;
        if (hr > base && height[kk2] < 0.5) {
          height[kk2] = Math.min(hr, -0.6);
          coral[kk2] = Math.max(coral[kk2], 1 - smoothstep(-2, 4, ed) * endFade);
        } else if (height[kk2] < 0) height[kk2] = Math.min(height[kk2], base);
      }
    }
    return { height: height, coral: coral, tone: tone, peak: peak };
  }

  // Where the reef break sits: p0 is its inner end near the beach, dir runs
  // out along the ledge, side points to its up-wave face.
  function reefBreakFrame() {
    var a0 = WINDWARD + 0.5;                              // near the end of the open sector
    var p0 = [Math.cos(a0) * 395, Math.sin(a0) * 395];
    var sw = 35 * Math.PI / 180, crest = [-Math.sin(sw), Math.cos(sw)];   // crest lines of the swell
    // the ledge runs 45 degrees off the crests, seaward
    var ang = Math.atan2(crest[1], crest[0]) + 45 * Math.PI / 180;
    var dir = [Math.cos(ang), Math.sin(ang)];
    if (dir[0] * Math.cos(a0) + dir[1] * Math.sin(a0) < 0) { ang -= Math.PI / 2; dir = [Math.cos(ang), Math.sin(ang)]; }
    var side = [-dir[1], dir[0]];
    if (side[0] * -Math.cos(sw) + side[1] * -Math.sin(sw) < 0) side = [-side[0], -side[1]];   // face the incoming swell
    return { p0: p0, dir: dir, side: side, len: 230 };
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
    var hBreak = new Float32Array(n2);                     // height of the last breaker on this ray
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
        var hb = up >= 0 ? hBreak[up] : 0;
        if (b < 0 && Hc > 0.78 * d) { b = A[k]; hb = Hc; }   // break point
        // breaker limit 0.78 d; a bore keeps ~0.55 d plus the momentum that
        // carries it up the beach (scaled by the incoming swell)
        var cap = b >= 0 ? 0.55 * d + 0.16 * swellH * (1 - 0.8 * shadow[k]) : 0.78 * d;
        if (Hc > cap) { Hc = cap; r = Hc / Hu; }
        if (b >= 0 && Hc < 0.3 * d) b = -1;                  // reformed in deeper water
        ratio[k] = r; broke[k] = b; hBreak[k] = hb;
        // in the last metre of depth the bore surges toward its run-up level,
        // meeting the swash on the sand without a step
        if (b >= 0 && d < 1) Hc = Math.max(Hc, 1.3 * 0.45 * hb / 0.55 * (1 - d));
        H[k] = Hc * clamp01((24 - d) / 12);
        stage[k] = b < 0 ? Math.min(Hc / (0.78 * d), 0.999) : 1 + clamp01((A[k] - b) / (0.55 * SWELL_T));
      } else {
        // Swash: the bore runs up the sand to a run-up height R ~ 0.45 x the
        // breaker. On land H is sized so the bore crest (0.55 H) clears the
        // sand below R and falls short above it.
        var hb2 = up >= 0 ? hBreak[up] : 0, R = 0.45 * hb2, hg = h[k];
        hBreak[k] = hb2;
        H[k] = R > 0.02 && hg < R + 0.3 ? (hg + 1.3 * Math.max(R - hg, 0)) / 0.55 * smoothstep(R + 0.3, R, hg) : 0;
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
      N: N, SIZE: SIZE, DX: DX, HALF: HALF, OCEAN_FLOOR: OCEAN_FLOOR, SWELL_T: SWELL_T, GROUP: GROUP,
      height: terr.height, coral: terr.coral, tone: terr.tone, peak: terr.peak,
      sdf: f.sdf, land: f.land, fields: f,
      timings: { terrain: t1 - t0, fields: t2 - t1 },
      // helpers travel with the data so the renderer needs no global
      heightAt: heightAt, terrainTexture: terrainTexture, shoreTexture: shoreTexture, shoreKTexture: shoreKTexture, setSwell: setSwell,
      palmMesh: palmMesh, canopyMesh: canopyMesh, coralMeshes: coralMeshes, bilinear: bilinear,
      fishMesh: fishMesh, turtleMesh: turtleMesh, rayMesh: rayMesh,
    };
    setSwell(w, opts.swell == null ? 1.2 : opts.swell, opts.swellDir == null ? 35 : opts.swellDir);
    w.timings.shore = Date.now() - t2;
    var t3 = Date.now();
    w.plants = scatterPlants(w, opts.seed);
    w.timings.plants = Date.now() - t3;
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
  // Wave vector (rad/m) of the shore swell, pointing the way it travels, and
  // the phase of the wave-group envelope (sets of bigger waves every sixth
  // period) as cos/sin.
  var GROUP = 6;
  function shoreKTexture(w) {
    var out = new Float32Array(N * N * 4), kg = TAU / (SWELL_T * GROUP);
    for (var k = 0; k < N * N; k++) {
      var depth = Math.max(-w.height[k], 0.25), c = w.land[k] ? SWASH_C : phaseSpeed(depth), kk = TAU / (SWELL_T * c);
      out[4 * k] = w.dirX[k] * kk; out[4 * k + 1] = w.dirZ[k] * kk;
      out[4 * k + 2] = Math.cos(w.arrival[k] * kg); out[4 * k + 3] = Math.sin(w.arrival[k] * kg);
    }
    return out;
  }

  // --------------------------------------------------------------- plants
  // Where the trees grow: coconut palms crowd the beaches and motus, leaning
  // seaward; rainforest canopy covers the hills, thinning on steep rock.
  // Each palm: x, y, z, height, rotation, lean, phase, tint (8 floats);
  // each canopy tree: x, y, z, radius, rotation, squash, phase, tint.
  function scatterPlants(w, seed) {
    var rnd = mulberry32((seed || 7) + 1234), h = w.height, palms = [], trees = [];
    function slopeAt(x, z) {
      var e = 2;
      var gx = (heightAt(w, x + e, z) - heightAt(w, x - e, z)) / (2 * e), gz = (heightAt(w, x, z + e) - heightAt(w, x, z - e)) / (2 * e);
      return Math.sqrt(gx * gx + gz * gz);
    }
    var nGrove = makeNoise((seed || 7) + 77);
    for (var z = -HALF + 2; z < HALF - 2; z += 4.5) {
      for (var x = -HALF + 2; x < HALF - 2; x += 4.5) {
        var px = x + (rnd() - 0.5) * 4, pz = z + (rnd() - 0.5) * 4;
        var y = heightAt(w, px, pz);
        if (y < 0.9) continue;
        var inland = -bilinear(w.sdf, px, pz, 0), sl = slopeAt(px, pz);
        var grove = 0.5 + 0.5 * nGrove(px / 60, pz / 60);
        // palms: a belt behind the beach, scattered further up
        var pPalm = (inland > 5 && inland < 95 ? 0.5 * Math.exp(-Math.max(0, inland - 18) / 30) : 0) + (y < 30 ? 0.03 : 0);
        pPalm *= (sl < 0.35 ? 1 : 0) * (0.35 + 0.9 * grove);
        if (rnd() < pPalm) {
          // lean toward the sea, more so near the beach
          var gx = bilinear(w.sdf, px + 2, pz, 0) - bilinear(w.sdf, px - 2, pz, 0), gz = bilinear(w.sdf, px, pz + 2, 0) - bilinear(w.sdf, px, pz - 2, 0);
          var lean = Math.atan2(gz, gx), amt = (inland < 30 ? 0.6 : 0.25) + rnd() * 0.5;
          palms.push(px, y - 0.15, pz, 7 + rnd() * 7, lean, amt, rnd() * TAU, rnd());
          continue;
        }
        // rainforest canopy (one candidate in two on this finer grid)
        if (inland > 20 && y > 3.2 && sl < 0.85 && y < 218 && rnd() < 0.9 * (0.6 + 0.4 * grove) * (1 - smoothstep(0.6, 0.85, sl))) {
          trees.push(px, y - 0.6, pz, 4.5 + rnd() * 4 + Math.min(y / 50, 2), rnd() * TAU, 0.5 + rnd() * 0.25, rnd() * TAU, rnd());
        }
      }
    }
    return { palms: new Float32Array(palms), trees: new Float32Array(trees) };
  }

  // Coconut palm, height 1, trunk base at the origin, curving toward +x.
  // Vertices: position (3), normal (3), uv (2), kind (1: 0 trunk, 1 frond,
  // 2 coconut). Fronds are V-folded strips whose leaflets the fragment
  // shader cuts out (uv.x across -1..1, uv.y along 0..1).
  // far = true gives a ~130-triangle stand-in for distant palms.
  function palmMesh(seed, far) {
    var rnd = mulberry32((seed || 7) + 4321), V = [], I = [];
    function vert(p, n, u, v, kind) { V.push(p[0], p[1], p[2], n[0], n[1], n[2], u, v, kind); return V.length / 9 - 1; }
    function norm(a) { var l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }
    function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
    function add(a, b, k) { return [a[0] + b[0] * k, a[1] + b[1] * k, a[2] + b[2] * k]; }
    var SEG = far ? 4 : 10, SIDES = far ? 4 : 7, k, j;
    function spine(s) { return [0.16 * s * s, s, 0]; }
    for (k = 0; k <= SEG; k++) {
      var s = k / SEG, c = spine(s), t = norm([0.32 * s, 1, 0]), b = [0, 0, 1], u = cross(b, t);
      var r = 0.026 * (1 - 0.4 * s) * (1 + 0.7 * Math.exp(-s * 14));
      for (j = 0; j <= SIDES; j++) {
        var a = j / SIDES * TAU, off = add(add([0, 0, 0], u, Math.cos(a) * r), b, Math.sin(a) * r);
        vert(add(c, off, 1), norm(off), j / SIDES, s * 9, 0);
      }
    }
    for (k = 0; k < SEG; k++) for (j = 0; j < SIDES; j++) {
      var i0 = k * (SIDES + 1) + j, i1 = i0 + SIDES + 1;
      I.push(i0, i1, i0 + 1, i0 + 1, i1, i1 + 1);
    }
    var crown = spine(1), NF = far ? 8 : 13, STEPS = far ? 3 : 9;
    for (var f = 0; f < NF; f++) {
      var ang = f / NF * TAU + (rnd() - 0.5) * 0.4;
      var pitch = 0.55 - 0.85 * ((f * 5) % NF) / NF + (rnd() - 0.5) * 0.2;   // young fronds up, old ones drooping
      var dir = [Math.cos(ang) * Math.cos(pitch), Math.sin(pitch), Math.sin(ang) * Math.cos(pitch)];
      var len = 0.4 + rnd() * 0.12, base = V.length / 9;
      for (k = 0; k <= STEPS; k++) {
        var tt = k / STEPS;
        var q = add(add(crown, dir, len * tt), [0, -1, 0], 0.6 * len * tt * tt);
        var tg = norm(add(dir, [0, -1, 0], 1.2 * tt));
        var side = norm(cross(tg, [0, 1, 0])), up = norm(cross(side, tg));
        var wdt = 0.12 * len * Math.pow(Math.sin(Math.PI * Math.min(1, 0.08 + tt * 0.95)), 0.7);
        vert(add(add(q, side, wdt), up, -0.4 * wdt), norm(add(up, side, 0.4)), -1, tt, 1);
        vert(q, up, 0, tt, 1);
        vert(add(add(q, side, -wdt), up, -0.4 * wdt), norm(add(up, side, -0.4)), 1, tt, 1);
      }
      for (k = 0; k < STEPS; k++) {
        var a0 = base + k * 3, a1 = a0 + 3;
        I.push(a0, a1, a0 + 1, a0 + 1, a1, a1 + 1, a0 + 1, a1 + 1, a0 + 2, a0 + 2, a1 + 1, a1 + 2);
      }
    }
    // a few coconuts under the crown
    for (var cIdx = 0; cIdx < (far ? 0 : 4); cIdx++) {
      var ca = cIdx * 1.7 + 0.3, cc = add(crown, [Math.cos(ca) * 0.035, -0.04 - 0.015 * (cIdx % 2), Math.sin(ca) * 0.035], 1), cr = 0.016, cb = V.length / 9;
      var dirs = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
      dirs.forEach(function (d) { vert(add(cc, d, cr), d, 0, 0, 2); });
      [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]].forEach(function (t3) { I.push(cb + t3[0], cb + t3[1], cb + t3[2]); });
    }
    return { vertices: new Float32Array(V), indices: new Uint16Array(I) };
  }

  // Rainforest crown: a lumpy, squashed icosphere (radius 1).
  function canopyMesh(seed, far) {
    var n = makeNoise((seed || 7) + 555);
    var t = (1 + Math.sqrt(5)) / 2, P = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]];
    var F = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
    P = P.map(function (p) { var l = Math.hypot(p[0], p[1], p[2]); return [p[0] / l, p[1] / l, p[2] / l]; });
    for (var lvl = 0; lvl < (far ? 1 : 2); lvl++) {
      var mid = {}, F2 = [];
      var midpoint = function (a, b) {
        var key = a < b ? a + "_" + b : b + "_" + a;
        if (mid[key] != null) return mid[key];
        var m = [(P[a][0] + P[b][0]) / 2, (P[a][1] + P[b][1]) / 2, (P[a][2] + P[b][2]) / 2], l = Math.hypot(m[0], m[1], m[2]);
        P.push([m[0] / l, m[1] / l, m[2] / l]);
        return (mid[key] = P.length - 1);
      };
      F.forEach(function (f) { var a = midpoint(f[0], f[1]), b = midpoint(f[1], f[2]), c = midpoint(f[2], f[0]); F2.push([f[0], a, c], [f[1], b, a], [f[2], c, b], [a, b, c]); });
      F = F2;
    }
    var V = [], I = [];
    P.forEach(function (p) {
      var r = 1 + 0.22 * n(p[0] * 2.1 + 3, p[2] * 2.1 + p[1] * 1.3) + 0.08 * n(p[0] * 6, p[1] * 6 + p[2] * 3);
      V.push(p[0] * r, p[1] * r, p[2] * r, p[0], p[1], p[2], 0, p[1], 3);
    });
    F.forEach(function (f) { I.push(f[0], f[1], f[2]); });
    return { vertices: new Float32Array(V), indices: new Uint16Array(I) };
  }

  // ------------------------------------------------------------ reef life
  // Meshes share the plant vertex format: position, normal, uv, part.
  function builder() {
    var V = [], I = [];
    return {
      vert: function (p, n, u, v, k) { V.push(p[0], p[1], p[2], n[0], n[1], n[2], u, v, k || 0); return V.length / 9 - 1; },
      tri: function (a, b, c) { I.push(a, b, c); },
      quad: function (a, b, c, d) { I.push(a, b, c, a, c, d); },
      count: function () { return V.length / 9; },
      done: function () { return { vertices: new Float32Array(V), indices: new Uint16Array(I) }; },
    };
  }
  function v3norm(a) { var l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }
  function v3cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  // A tapered tube along points[] (radius per point), `sides` around.
  function tube(B, pts, radii, sides, part, v0, v1) {
    var base = B.count();
    for (var i = 0; i < pts.length; i++) {
      var t = v3norm(i < pts.length - 1 ? [pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1], pts[i + 1][2] - pts[i][2]] : [pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2]]);
      var u = v3norm(v3cross(t, Math.abs(t[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0])), b = v3cross(t, u);
      for (var j = 0; j <= sides; j++) {
        var a = j / sides * TAU, c = Math.cos(a), s = Math.sin(a);
        var n = [u[0] * c + b[0] * s, u[1] * c + b[1] * s, u[2] * c + b[2] * s];
        B.vert([pts[i][0] + n[0] * radii[i], pts[i][1] + n[1] * radii[i], pts[i][2] + n[2] * radii[i]], n, j / sides, v0 + (v1 - v0) * i / (pts.length - 1), part);
      }
    }
    for (i = 0; i < pts.length - 1; i++) for (j = 0; j < sides; j++) {
      var k0 = base + i * (sides + 1) + j, k1 = k0 + sides + 1;
      B.quad(k0, k1, k1 + 1, k0 + 1);
    }
  }
  // Coral colonies, about a metre across: brain, staghorn, table, sea fan,
  // soft tube coral, boulder. Shaders add the surface detail.
  function coralMeshes(seed) {
    var rnd = mulberry32((seed || 7) + 808), n = makeNoise((seed || 7) + 909), out = {}, B, i, j;
    // brain: an undercut dome
    B = builder();
    for (i = 0; i <= 10; i++) for (j = 0; j <= 20; j++) {
      var th = i / 10 * Math.PI * 0.56, ph = j / 20 * TAU;
      var p = [Math.sin(th) * Math.cos(ph), Math.cos(th) * 0.72 - 0.08, Math.sin(th) * Math.sin(ph)];
      B.vert(p, v3norm([p[0], p[1] / 0.72, p[2]]), j / 20, i / 10, 0);
    }
    for (i = 0; i < 10; i++) for (j = 0; j < 20; j++) { var a0 = i * 21 + j; B.quad(a0, a0 + 21, a0 + 22, a0 + 1); }
    out.brain = B.done();
    // staghorn: forking antlers
    B = builder();
    (function branch(start, dir, len, r, depth) {
      var pts = [], radii = [], end = [start[0] + dir[0] * len, start[1] + dir[1] * len, start[2] + dir[2] * len];
      for (var k = 0; k <= 2; k++) { var t = k / 2; pts.push([start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t, start[2] + (end[2] - start[2]) * t]); radii.push(r * (1 - 0.35 * t)); }
      tube(B, pts, radii, 4, depth, depth / 3, (depth + 1) / 3);
      if (depth >= 2) return;
      var kids = 3 + (rnd() < 0.5 ? 1 : 0);
      for (var c = 0; c < kids; c++) {
        var a = c / kids * TAU + rnd(), sp = 0.45 + rnd() * 0.3;
        var nd = v3norm([dir[0] + Math.cos(a) * sp, dir[1] + 0.25, dir[2] + Math.sin(a) * sp]);
        branch(end, nd, len * (0.65 + rnd() * 0.2), r * 0.62, depth + 1);
      }
    })([0, -0.05, 0], [0, 1, 0], 0.42, 0.07, 0);
    out.staghorn = B.done();
    // table: a scalloped plate on a stalk
    B = builder();
    tube(B, [[0, -0.05, 0], [0, 0.42, 0]], [0.1, 0.08], 6, 0, 0, 0.3);
    var NS = 28, ring = [];
    for (j = 0; j <= NS; j++) { var aa = j / NS * TAU; ring.push(1 + 0.07 * Math.sin(aa * 9) + 0.12 * n(Math.cos(aa) * 2, Math.sin(aa) * 2)); }
    // top and underside, joined by a rounded rim whose normals point outward
    var rims = {};
    [[1, 0.47], [-1, 0.42]].forEach(function (side) {
      var c = B.vert([0, side[1], 0], [0, side[0], 0], 0, 0.5, side[0] > 0 ? 1 : 2), ks = [];
      for (var q = 0; q <= NS; q++) { var aq = q / NS * TAU; ks.push(B.vert([Math.cos(aq) * ring[q], side[1] + 0.03 * ring[q], Math.sin(aq) * ring[q]], v3norm([Math.cos(aq) * 0.25, side[0], Math.sin(aq) * 0.25]), ring[q], 0.5 + q / NS * 0.5, side[0] > 0 ? 1 : 2)); }
      for (q = 0; q < NS; q++) { if (side[0] > 0) B.tri(c, ks[q + 1], ks[q]); else B.tri(c, ks[q], ks[q + 1]); }
      rims[side[0]] = ks;
    });
    var band = [];
    for (var q2 = 0; q2 <= NS; q2++) {
      var aq2 = q2 / NS * TAU, ro = [Math.cos(aq2), 0, Math.sin(aq2)], rr = ring[q2] + 0.035;
      band.push(B.vert([ro[0] * rr, 0.445 + 0.03 * ring[q2], ro[2] * rr], ro, ring[q2], 0.5 + q2 / NS * 0.5, 1));
    }
    for (q2 = 0; q2 < NS; q2++) { B.quad(rims[1][q2], rims[1][q2 + 1], band[q2 + 1], band[q2]); B.quad(band[q2], band[q2 + 1], rims[-1][q2 + 1], rims[-1][q2]); }
    out.table = B.done();
    // sea fan: a curved lattice blade (the shader cuts the holes)
    B = builder();
    for (i = 0; i <= 8; i++) for (j = 0; j <= 10; j++) {
      var u = j / 10 * 2 - 1, v = i / 8;
      var w = Math.pow(Math.sin(Math.PI * Math.min(1, v * 1.1 + 0.05)), 0.6);
      B.vert([u * 0.75 * w, v * (1.05 - 0.25 * u * u), 0.12 * Math.sin(u * 2.4) * v], v3norm([-0.2 * Math.cos(u * 2.4), 0, 1]), u, v, 3);
    }
    for (i = 0; i < 8; i++) for (j = 0; j < 10; j++) { var b0 = i * 11 + j; B.quad(b0, b0 + 1, b0 + 12, b0 + 11); }
    out.fan = B.done();
    // soft coral / tube sponges: a cluster of open tubes
    B = builder();
    for (var tIdx = 0; tIdx < 6; tIdx++) {
      var ta = tIdx / 6 * TAU + rnd() * 0.5, tr = tIdx === 0 ? 0 : 0.18 + rnd() * 0.12, hgt = 0.45 + rnd() * 0.55;
      var bx = Math.cos(ta) * tr, bz = Math.sin(ta) * tr, lean = 0.15 * tr;
      tube(B, [[bx, -0.05, bz], [bx + Math.cos(ta) * lean * 0.5, hgt * 0.5, bz + Math.sin(ta) * lean * 0.5], [bx + Math.cos(ta) * lean, hgt, bz + Math.sin(ta) * lean]],
        [0.09, 0.085, 0.1], 6, 4, 0, 1);
    }
    out.soft = B.done();
    // boulder (massive) coral: a lumpy low dome
    B = builder();
    for (i = 0; i <= 8; i++) for (j = 0; j <= 16; j++) {
      var th2 = i / 8 * Math.PI * 0.6, ph2 = j / 16 * TAU;
      var d = [Math.sin(th2) * Math.cos(ph2), Math.cos(th2), Math.sin(th2) * Math.sin(ph2)];
      var rr = 1 + 0.18 * n(d[0] * 2.3 + 1, d[2] * 2.3 + d[1]) + 0.07 * n(d[0] * 6, d[2] * 6 - d[1] * 3);
      B.vert([d[0] * rr, d[1] * rr * 0.55 - 0.1, d[2] * rr], d, j / 16, i / 8, 5);
    }
    for (i = 0; i < 8; i++) for (j = 0; j < 16; j++) { var c0 = i * 17 + j; B.quad(c0, c0 + 17, c0 + 18, c0 + 1); }
    out.boulder = B.done();
    return out;
  }

  // Reef fish, 1 m long along +x (head at +0.5), laterally flattened, with a
  // forked tail and a dorsal fin. part: 0 body, 1 fins. uv: (head..tail, around).
  function fishMesh() {
    var B = builder(), SEG = 12, AROUND = 10, i, j;
    function prof(t) {   // t 0 = snout .. 1 = tail root
      var s = Math.sin(Math.PI * Math.pow(t, 0.75));
      return [0.2 * s + 0.012, 0.075 * s + 0.006];          // half height, half width
    }
    for (i = 0; i <= SEG; i++) {
      var t = i / SEG, x = 0.5 - t * 0.95, r = prof(t);
      for (j = 0; j <= AROUND; j++) {
        var a = j / AROUND * TAU, cy = Math.cos(a), cz = Math.sin(a);
        B.vert([x, cy * r[0], cz * r[1]], v3norm([0.15 * (0.5 - t), cy / r[0], cz / r[1]]), t, j / AROUND, 0);
      }
    }
    for (i = 0; i < SEG; i++) for (j = 0; j < AROUND; j++) { var k0 = i * (AROUND + 1) + j; B.quad(k0, k0 + 1, k0 + AROUND + 2, k0 + AROUND + 1); }
    // forked tail (two-sided)
    var tb = B.vert([-0.45, 0, 0], [0, 0, 1], 1.0, 0.5, 1);
    var tu = B.vert([-0.72, 0.24, 0], [0, 0, 1], 1.3, 0.0, 1), tm = B.vert([-0.62, 0, 0], [0, 0, 1], 1.2, 0.5, 1), td = B.vert([-0.72, -0.24, 0], [0, 0, 1], 1.3, 1.0, 1);
    B.tri(tb, tu, tm); B.tri(tb, tm, td);
    // dorsal and anal fins
    var d0 = B.vert([0.15, 0.19, 0], [0, 0, 1], 0.35, 0, 1), d1 = B.vert([-0.25, 0.17, 0], [0, 0, 1], 0.75, 0, 1), d2 = B.vert([-0.1, 0.33, 0], [0, 0, 1], 0.6, 0, 1);
    B.tri(d0, d1, d2);
    var a0 = B.vert([-0.05, -0.17, 0], [0, 0, 1], 0.55, 1, 1), a1 = B.vert([-0.3, -0.13, 0], [0, 0, 1], 0.8, 1, 1), a2 = B.vert([-0.25, -0.26, 0], [0, 0, 1], 0.75, 1, 1);
    B.tri(a0, a2, a1);
    return B.done();
  }
  // Green sea turtle, ~1 m shell along +x. part: 0 shell, 1 belly/head,
  // 2/3 front flippers (left/right), 4/5 rear flippers.
  function turtleMesh() {
    var B = builder(), i, j, R = 10, C = 16;
    for (i = 0; i <= R; i++) for (j = 0; j <= C; j++) {          // carapace
      var th = i / R * Math.PI * 0.5, ph = j / C * TAU;
      var p = [Math.sin(th) * Math.cos(ph) * 0.5, Math.cos(th) * 0.2 + 0.02, Math.sin(th) * Math.sin(ph) * 0.4];
      B.vert(p, v3norm([p[0] / 0.25, p[1] / 0.04, p[2] / 0.16]), j / C, i / R, 0);
    }
    for (i = 0; i < R; i++) for (j = 0; j < C; j++) { var k0 = i * (C + 1) + j; B.quad(k0, k0 + C + 1, k0 + C + 2, k0 + 1); }
    var c = B.vert([0, 0.0, 0], [0, -1, 0], 0.5, 0.5, 1), rim = [];          // plastron
    for (j = 0; j <= C; j++) { var ph2 = j / C * TAU; rim.push(B.vert([Math.cos(ph2) * 0.5, 0.02, Math.sin(ph2) * 0.4], [0, -1, 0], j / C, 1, 1)); }
    for (j = 0; j < C; j++) B.tri(c, rim[j + 1], rim[j]);
    tube(B, [[0.42, 0.04, 0], [0.55, 0.06, 0], [0.66, 0.05, 0]], [0.07, 0.08, 0.05], 6, 1, 0, 1);   // head
    function flipper(root, tip, width, part) {
      var side = v3norm(v3cross([tip[0] - root[0], tip[1] - root[1], tip[2] - root[2]], [0, 1, 0]));
      var q = [], k;
      for (k = 0; k <= 3; k++) {
        var t = k / 3, w = width * Math.sin(Math.PI * Math.min(1, 0.15 + t * 0.85));
        var cc = [root[0] + (tip[0] - root[0]) * t, root[1] + (tip[1] - root[1]) * t, root[2] + (tip[2] - root[2]) * t];
        q.push(B.vert([cc[0] + side[0] * w, cc[1], cc[2] + side[2] * w], [0, 1, 0], t, 0, part), B.vert([cc[0] - side[0] * w, cc[1], cc[2] - side[2] * w], [0, 1, 0], t, 1, part));
      }
      for (k = 0; k < 3; k++) B.quad(q[2 * k], q[2 * k + 2], q[2 * k + 3], q[2 * k + 1]);
    }
    flipper([0.22, 0.03, -0.3], [0.05, 0.0, -0.95], 0.11, 2);
    flipper([0.22, 0.03, 0.3], [0.05, 0.0, 0.95], 0.11, 3);
    flipper([-0.32, 0.02, -0.22], [-0.52, 0.0, -0.42], 0.08, 4);
    flipper([-0.32, 0.02, 0.22], [-0.52, 0.0, 0.42], 0.08, 5);
    return B.done();
  }
  // Spotted eagle ray, 1 m long, wings spanning z -1..1 (they flap in the
  // vertex shader by |z|), whip tail. part: 0 body/wings, 1 tail.
  function rayMesh() {
    var B = builder(), NZ = 16, NX = 6, i, j;
    for (i = 0; i <= NZ; i++) for (j = 0; j <= NX; j++) {
      var zz = i / NZ * 2 - 1, az = Math.abs(zz);
      var lead = 0.35 - 0.25 * az * az, trail = -0.3 + 0.32 * Math.pow(az, 0.8);     // swept-back wings
      var x = lead + (trail - lead) * j / NX;
      var th = 0.07 * (1 - az * az) * Math.sin(Math.PI * j / NX);
      B.vert([x, th, zz], [0, 1, 0], (zz + 1) / 2, j / NX, 0);
    }
    for (i = 0; i < NZ; i++) for (j = 0; j < NX; j++) { var k0 = i * (NX + 1) + j; B.quad(k0, k0 + NX + 1, k0 + NX + 2, k0 + 1); }
    var h0 = B.vert([0.38, 0.02, 0], [0, 1, 0], 0.5, 0, 0), h1 = B.vert([0.5, 0.0, -0.07], [0, 1, 0], 0.5, 0, 0), h2 = B.vert([0.5, 0.0, 0.07], [0, 1, 0], 0.5, 0, 0);
    B.tri(h0, h2, h1);    // the snout
    tube(B, [[-0.28, 0.02, 0], [-0.8, 0.01, 0], [-1.6, 0.0, 0]], [0.025, 0.012, 0.004], 4, 1, 0, 1);
    return B.done();
  }

  return {
    N: N, SIZE: SIZE, DX: DX, HALF: HALF, OCEAN_FLOOR: OCEAN_FLOOR, SWELL_T: SWELL_T, SWASH_C: SWASH_C,
    WINDWARD: WINDWARD, GROUP: GROUP, reefBreakFrame: reefBreakFrame,
    build: build, setSwell: setSwell, heightAt: heightAt, bilinear: bilinear,
    scatterPlants: scatterPlants, palmMesh: palmMesh, canopyMesh: canopyMesh, coralMeshes: coralMeshes,
    fishMesh: fishMesh, turtleMesh: turtleMesh, rayMesh: rayMesh,
    terrainTexture: terrainTexture, shoreTexture: shoreTexture, shoreKTexture: shoreKTexture,
    phaseSpeed: phaseSpeed, shoaling: shoaling, fastSweep: fastSweep, makeNoise: makeNoise, mulberry32: mulberry32,
  };
});
