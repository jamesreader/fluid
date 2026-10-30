/*
 * ocean.js — real-time 3D ocean over a white-sand lagoon, in WebGL2.
 *
 * Waves: Tessendorf's FFT ocean. A fetch-limited JONSWAP wind sea with
 * finite-depth dispersion (w^2 = g k tanh(k h)) and the TMA shallow-water
 * correction is split across three wave cascades (233 m, 36 m and 12 m tiles,
 * 256^2 each), so the long swell and the centimetre ripples are both resolved
 * without visible tiling. Every frame the spectra are advanced on the GPU and
 * inverse-FFT'd (Stockham radix-2, one fragment pass per stage) into choppy
 * displacement, slopes, and the Jacobian that seeds whitecap foam.
 *
 * Light: the water is clear (absorption close to pure sea water: red is gone
 * after a few metres, blue survives tens), so most of what you see is the sand
 * floor through the refracting surface. Each surface pixel bends its view ray
 * with Snell's law and marches it down to the bathymetry heightfield. The
 * floor is lit by a caustic pattern computed from the same waves: a mesh of
 * sun rays is refracted through the surface onto the floor and each triangle
 * brightens by how much its area shrank (Evan Wallace's method).
 *
 * Coordinates: world units are metres, y up, sea level at y = 0. All GPU work
 * happens relative to a grid origin O that follows the camera in fixed steps;
 * the per-texture offsets mod(O, tile) are computed here in double precision so
 * float32 shaders never see large coordinates.
 *
 * Pure-JS pieces (spectrum, bathymetry, FFT index math) are exported for
 * ocean-test.js. Ocean.Renderer owns everything that needs a GL context.
 */
(function (global, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else global.Ocean = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var G = 9.81, TAU = Math.PI * 2;
  var FFT_N = 256;
  // Tile sizes (m). L1 = 3 * L2 exactly: the caustics are built from those two
  // cascades, so their sum repeats on the 36 m tile. L0 is incommensurate with
  // both, which breaks up repetition at large scale.
  var CASCADE_L = [233, 36, 12];
  // Cascade c keeps wavenumbers in [CUT[c], CUT[c+1]): each hands over at the
  // next tile's 4th harmonic, so no wave is counted twice and the hand-over
  // band is sampled by >= 10 texels per wavelength on the coarser side.
  var CASCADE_CUT = [0, TAU / 36 * 4, TAU / 12 * 4, Infinity];
  var CAUSTIC_L = CASCADE_L[1];

  var DEFAULTS = {
    wind: 6,           // m/s at 10 m
    windDir: 35,       // degrees, direction the waves travel toward
    fetch: 25,         // km of open water upwind
    alignment: 1,      // multiplies the directional spreading exponent
    chop: 1.0,         // horizontal displacement (Tessendorf lambda)
    foam: 0.6,
    surf: 1.6,         // height (m) of the swell that breaks on the reef and beaches
    swell: 0,          // optional extra long swell in the open-ocean FFT (m)
    surfSim: true,     // simulate the surf zone around the camera (shallow water + lips)
    lips: 0.25,        // how readily breaking fronts throw lips (0 = none)
    night: 0,          // 0 day .. 1 moonlit night (the sun sliders then place the moon)
    glow: 0.8,         // bioluminescent plankton: lights up where the water is stirred, at night
    depth: 4.5,        // water depth the FFT spectrum is built for
    clarity: 1,        // >1 clearer, <1 murkier
    caustics: 1,
    sunElev: 52,       // degrees above horizon
    sunAzim: 200,      // degrees, compass-ish around y
    clouds: 0.35,
    exposure: 0.22,
    bloom: true,
    renderScale: 1,
    meshDetail: 512,
    timeScale: 1,
  };

  // --------------------------------------------------------------- spectrum
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // 2n standard normals (Box-Muller), fixed per seed so changing the wind
  // rescales the same sea instead of reshuffling it.
  function gaussians(n, seed) {
    var r = mulberry32(seed), out = new Float32Array(2 * n);
    for (var i = 0; i < n; i++) {
      var u = Math.max(1e-12, r()), v = r(), m = Math.sqrt(-2 * Math.log(u));
      out[2 * i] = m * Math.cos(TAU * v);
      out[2 * i + 1] = m * Math.sin(TAU * v);
    }
    return out;
  }
  function lgamma(x) {   // Lanczos, g = 7
    var c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
      771.32342877765313, -176.61502916214059, 12.507343278686905,
      -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
    x -= 1;
    var a = c[0], t = x + 7.5;
    for (var i = 1; i < 9; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(TAU) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }
  // Normaliser of D(theta) = Q(s) |cos(theta/2)|^(2s) over the full circle.
  function spreadQ(s) {
    return Math.exp(2 * s * Math.LN2 + 2 * lgamma(s + 1) - Math.log(TAU) - lgamma(2 * s + 1));
  }

  // Sea-state constants for wind U (m/s), fetch F (m): JONSWAP peak frequency,
  // Phillips constant, and Mitsuyasu's peak spreading exponent.
  function seaState(p) {
    var U = Math.max(0.5, p.wind), F = Math.max(1000, p.fetch * 1000);
    var wp = 22 * Math.pow(G * G / (U * F), 1 / 3);
    return {
      U: U, F: F, wp: wp,
      alpha: 0.076 * Math.pow(U * U / (F * G), 0.22),
      sp: 11.5 * Math.pow(wp * U / G, -2.5),
      h: Math.max(0.5, p.depth),
    };
  }
  // JONSWAP S(w) times the TMA depth factor (Kitaigorodskii): long waves in
  // shallow water carry less energy than the deep-water spectrum says.
  function tma(w, h) {
    var wh = w * Math.sqrt(h / G);
    return wh <= 1 ? 0.5 * wh * wh : (wh < 2 ? 1 - 0.5 * (2 - wh) * (2 - wh) : 1);
  }
  function spectrumW(w, ss) {
    if (w <= 0) return 0;
    var sigma = w <= ss.wp ? 0.07 : 0.09;
    var r = Math.exp(-(w - ss.wp) * (w - ss.wp) / (2 * sigma * sigma * ss.wp * ss.wp));
    var S = ss.alpha * G * G / Math.pow(w, 5) * Math.exp(-1.25 * Math.pow(ss.wp / w, 4)) * Math.pow(3.3, r);
    return S * tma(w, ss.h);
  }
  function dispersion(k, h) {
    var kh = Math.min(k * h, 20), th = Math.tanh(kh), w = Math.sqrt(G * k * th);
    return { w: w, dwdk: G * (th + kh * (1 - th * th)) / (2 * w) };
  }
  // Ocean swell: a narrow JONSWAP peak (gamma 6) at 9.5 s, scaled to height
  // swellH, travelling with the wind sea and tightly aligned (s = 24).
  var SWELL_WP = TAU / 9.5, SWELL_S = 24, SWELL_Q = spreadQ(SWELL_S);
  function swellShape(w) {
    if (w <= 0) return 0;
    var sigma = w <= SWELL_WP ? 0.07 : 0.09;
    var r = Math.exp(-(w - SWELL_WP) * (w - SWELL_WP) / (2 * sigma * sigma * SWELL_WP * SWELL_WP));
    return Math.pow(w, -5) * Math.exp(-1.25 * Math.pow(SWELL_WP / w, 4)) * Math.pow(6, r);
  }
  var SWELL_NORM = (function () { var m = 0, dw = 0.0005; for (var w = dw / 2; w < 6; w += dw) m += swellShape(w) * dw; return 1 / m; })();
  function swellW(w, p) { return p.swell > 0 ? swellShape(w) * SWELL_NORM * (p.swell / 4) * (p.swell / 4) : 0; }
  // Variance density E(kx, kz) for one wavevector (m^4): S(w) D(theta) dw/dk / k,
  // wind sea plus swell.
  function spectrumK(kx, kz, ss, p) {
    var k = Math.sqrt(kx * kx + kz * kz);
    if (k <= 0) return 0;
    var d = dispersion(k, ss.h);
    var s = d.w <= ss.wp ? ss.sp * Math.pow(d.w / ss.wp, 5) : ss.sp * Math.pow(d.w / ss.wp, -2.5);
    s = Math.max(0.3, s * p.alignment);
    var dth = Math.atan2(kz, kx) - p.windDir * Math.PI / 180;
    var c = Math.abs(Math.cos(dth / 2));
    var E = spectrumW(d.w, ss) * spreadQ(s) * Math.pow(c, 2 * s);
    if (p.swell > 0) E += swellW(d.w, p) * tma(d.w, ss.h) * SWELL_Q * Math.pow(c, 2 * SWELL_S);
    return E * d.dwdk / k * Math.exp(-k * k * 1e-4);
  }

  // Initial amplitudes for cascade c, packed RGBA per texel as
  // (h0(k), conj(h0(-k))) so the GPU can advance h(k,t) with one fetch.
  // h0 = (xi_r + i xi_i) sqrt(E dk^2 / 4) makes Var(eta) = sum E dk^2.
  function buildSpectrum(c, p, xi) {
    var N = FFT_N, L = CASCADE_L[c], dk = TAU / L;
    var lo = CASCADE_CUT[c], hi = CASCADE_CUT[c + 1], ss = seaState(p);
    var amp = new Float32Array(N * N * 2), variance = 0;
    for (var m = 0; m < N; m++) {
      var kz = dk * (m < N / 2 ? m : m - N);
      for (var n = 0; n < N; n++) {
        var kx = dk * (n < N / 2 ? n : n - N), k = Math.sqrt(kx * kx + kz * kz);
        // Nyquist row/column has no Hermitian partner: leave it empty.
        if (k < lo || k >= hi || k === 0 || n === N / 2 || m === N / 2) continue;
        var E = spectrumK(kx, kz, ss, p) * dk * dk;
        var a = Math.sqrt(E / 4), i = m * N + n;
        amp[2 * i] = xi[2 * i] * a;
        amp[2 * i + 1] = xi[2 * i + 1] * a;
        variance += E;
      }
    }
    var out = new Float32Array(N * N * 4);
    for (m = 0; m < N; m++) {
      for (n = 0; n < N; n++) {
        var j = m * N + n, jm = ((N - m) % N) * N + ((N - n) % N);
        out[4 * j] = amp[2 * j];
        out[4 * j + 1] = amp[2 * j + 1];
        out[4 * j + 2] = amp[2 * jm];
        out[4 * j + 3] = -amp[2 * jm + 1];
      }
    }
    return { data: out, variance: variance };
  }
  // Expected surface variance (m^2) the cascades represent, and the 1D
  // integral of the same spectrum; ocean-test.js checks they agree.
  function spectrumVariance(p) {
    var ss = seaState(p), v = 0, N = FFT_N;
    for (var c = 0; c < 3; c++) {
      var dk = TAU / CASCADE_L[c], lo = CASCADE_CUT[c], hi = CASCADE_CUT[c + 1];
      for (var m = 0; m < N; m++) {
        for (var n = 0; n < N; n++) {
          if (n === N / 2 || m === N / 2) continue;
          var kx = dk * (n < N / 2 ? n : n - N), kz = dk * (m < N / 2 ? m : m - N);
          var k = Math.sqrt(kx * kx + kz * kz);
          if (k < lo || k >= hi || k === 0) continue;
          v += spectrumK(kx, kz, ss, p) * dk * dk;
        }
      }
    }
    return v;
  }
  function spectrumM0(p) {   // integral of S(w) over w (D integrates to 1)
    var ss = seaState(p), m0 = 0, dw = 0.0005;
    for (var w = dw / 2; w < 40; w += dw) m0 += (spectrumW(w, ss) + swellW(w, p) * tma(w, ss.h)) * dw;
    return m0;
  }

  // Stockham radix-2 inverse FFT, written the way the fragment shader gathers:
  // each output reads two inputs. Exported so the test can check the index
  // math against a direct DFT.
  function fftStockham(re, im) {
    var N = re.length, a = [Float64Array.from(re), Float64Array.from(im)];
    var b = [new Float64Array(N), new Float64Array(N)];
    for (var Ns = 1; Ns < N; Ns <<= 1) {
      for (var o = 0; o < N; o++) {
        var r = o & (2 * Ns - 1), k = r & (Ns - 1), j = ((o / (2 * Ns)) | 0) * Ns + k;
        var ang = Math.PI * k / Ns, wr = Math.cos(ang), wi = Math.sin(ang);
        if (r >= Ns) { wr = -wr; wi = -wi; }
        var br = a[0][j + N / 2], bi = a[1][j + N / 2];
        b[0][o] = a[0][j] + wr * br - wi * bi;
        b[1][o] = a[1][j] + wr * bi + wi * br;
      }
      var t = a; a = b; b = t;
    }
    return a;
  }

  // --------------------------------------------------------------- sky (CPU)
  // Single-scattering atmosphere constants, shared with the GLSL sky LUT:
  // Rayleigh (Bruneton's sea-level coefficients) and Mie, exponential
  // density with 8 km / 1.2 km scale heights.
  var EARTH_R = 6371e3, ATMOS_R = 6471e3;
  var K_RAY = [5.802e-6, 13.558e-6, 33.1e-6], K_MIE = 21e-6, H_RAY = 8e3, H_MIE = 1.2e3;
  var SUN_I = 22;              // sun irradiance above the atmosphere (scene units)
  // Fraction of sunlight that reaches sea level along sunDir, per channel.
  function sunTransmittance(sunDir) {
    var y = Math.max(sunDir[1], -0.05), r0 = EARTH_R + 2;
    var b = y * r0, tMax = -b + Math.sqrt(b * b - r0 * r0 + ATMOS_R * ATMOS_R);
    var n = 96, st = tMax / n, odR = 0, odM = 0;
    for (var i = 0; i < n; i++) {
      var t = (i + 0.5) * st, px = sunDir[0] * t, py = r0 + y * t, pz = sunDir[2] * t;
      var h = Math.sqrt(px * px + py * py + pz * pz) - EARTH_R;
      odR += Math.exp(-h / H_RAY) * st; odM += Math.exp(-h / H_MIE) * st;
    }
    return K_RAY.map(function (k) { return Math.exp(-(k * odR + K_MIE * 1.1 * odM)); });
  }

  // ===================================================================== GLSL
  var HEAD = "#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\nprecision highp sampler2DShadow;\n#define PI 3.14159265359\n#define TAU 6.28318530718\n";

  var VS_FULL = HEAD + [
    "out vec2 vUv;",
    "void main() {",
    "  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));",
    "  vUv = p;",
    "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);",
    "}"].join("\n");

  // h(k,t) = h0(k) e^{iwt} + conj(h0(-k)) e^{-iwt}, expanded into the eight
  // real fields we need, packed as four complex numbers A + iB (two real
  // fields per complex IFFT: their spectra are Hermitian).
  var FS_SPECTRUM = HEAD + [
    "uniform sampler2D uH0;",
    "uniform float uTime, uL, uDepth;",
    "uniform int uN;",
    "layout(location = 0) out vec4 o0;",
    "layout(location = 1) out vec4 o1;",
    "vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }",
    "vec2 mulI(vec2 a) { return vec2(-a.y, a.x); }",
    "void main() {",
    "  ivec2 id = ivec2(gl_FragCoord.xy);",
    "  float kx = TAU / uL * float(id.x < uN / 2 ? id.x : id.x - uN);",
    "  float kz = TAU / uL * float(id.y < uN / 2 ? id.y : id.y - uN);",
    "  float k = length(vec2(kx, kz));",
    "  if (k < 1e-6) { o0 = vec4(0.0); o1 = vec4(0.0); return; }",
    "  vec4 h0 = texelFetch(uH0, id, 0);",
    "  float w = sqrt(9.81 * k * tanh(min(k * uDepth, 20.0)));",
    "  float ph = mod(w * uTime, TAU);",
    "  vec2 e = vec2(cos(ph), sin(ph));",
    "  vec2 h = cmul(h0.xy, e) + cmul(h0.zw, vec2(e.x, -e.y));",
    "  vec2 ih = mulI(h);",
    "  float ik = 1.0 / k;",
    "  vec2 Dx = ih * kx * ik, Dz = ih * kz * ik;",       // choppy displacement
    "  vec2 Hx = ih * kx, Hz = ih * kz;",                 // height slopes
    "  vec2 Dxx = -h * kx * kx * ik, Dzz = -h * kz * kz * ik, Dxz = -h * kx * kz * ik;",
    "  o0 = vec4(Dx + mulI(h), Dz + mulI(Dxz));",
    "  o1 = vec4(Hx + mulI(Hz), Dxx + mulI(Dzz));",
    "}"].join("\n");

  // One Stockham radix-2 stage along x (uHoriz = 1) or y, on both MRT inputs.
  var FS_FFT = HEAD + [
    "uniform sampler2D uIn0, uIn1;",
    "uniform int uStage, uHoriz, uN;",
    "layout(location = 0) out vec4 o0;",
    "layout(location = 1) out vec4 o1;",
    "vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }",
    "void main() {",
    "  ivec2 p = ivec2(gl_FragCoord.xy);",
    "  int o = uHoriz == 1 ? p.x : p.y;",
    "  int Ns = 1 << uStage;",
    "  int r = o & (2 * Ns - 1);",
    "  int k = r & (Ns - 1);",
    "  int j = (o >> (uStage + 1)) * Ns + k;",
    "  ivec2 pa = p, pb = p;",
    "  if (uHoriz == 1) { pa.x = j; pb.x = j + uN / 2; } else { pa.y = j; pb.y = j + uN / 2; }",
    "  float ang = PI * float(k) / float(Ns);",
    "  vec2 w = vec2(cos(ang), sin(ang));",
    "  if (r >= Ns) w = -w;",
    "  vec4 a0 = texelFetch(uIn0, pa, 0), b0 = texelFetch(uIn0, pb, 0);",
    "  vec4 a1 = texelFetch(uIn1, pa, 0), b1 = texelFetch(uIn1, pb, 0);",
    "  o0 = vec4(a0.xy + cmul(w, b0.xy), a0.zw + cmul(w, b0.zw));",
    "  o1 = vec4(a1.xy + cmul(w, b1.xy), a1.zw + cmul(w, b1.zw));",
    "}"].join("\n");

  // Unpack the IFFT into displacement (+ foam) and derivative textures.
  // Foam accumulates where the surface folds (Jacobian < bias) and decays.
  var FS_ASSEMBLE = HEAD + [
    "uniform sampler2D uF0, uF1, uPrev;",
    "uniform float uChop, uDecay, uFoamBias, uFoamGain;",
    "layout(location = 0) out vec4 oDisp;",
    "layout(location = 1) out vec4 oDer;",
    "void main() {",
    "  ivec2 p = ivec2(gl_FragCoord.xy);",
    "  vec4 f0 = texelFetch(uF0, p, 0), f1 = texelFetch(uF1, p, 0);",
    "  float Dxx = uChop * f1.z, Dzz = uChop * f1.w, Dxz = uChop * f0.w;",
    "  float J = (1.0 + Dxx) * (1.0 + Dzz) - Dxz * Dxz;",
    "  float foam = texelFetch(uPrev, p, 0).w * uDecay + max(uFoamBias - J, 0.0) * uFoamGain;",
    "  oDisp = vec4(uChop * f0.x, f0.y, uChop * f0.z, min(foam, 2.0));",
    "  oDer = vec4(f1.x, f1.y, Dxx, Dzz);",
    "}"].join("\n");

  // ----------------------------------------------------------- shared chunks
  var C_COMMON = [
    "uniform vec3 uCam;",          // camera, relative to the grid origin O
    "uniform vec3 uSunDir;",       // unit vector toward the sun
    "uniform vec3 uSunCol;",       // sun irradiance at sea level
    "uniform float uTime;",
    "uniform vec3 uGlowCol;",      // plankton light per unit of stirring (0 by day)
    "uniform float uNight;",       // 0 day .. 1 moonlit night
    "float sat(float x) { return clamp(x, 0.0, 1.0); }",
  ].join("\n");

  var C_OCEAN = [
    "uniform sampler2D uDisp0, uDisp1, uDisp2, uDer0, uDer1, uDer2;",
    "uniform vec3 uCascL;",
    "uniform vec2 uOff0, uOff1, uOff2;",
    "vec4 oceanDisp(vec2 xz, vec3 lod) {",
    "  return textureLod(uDisp0, (xz + uOff0) / uCascL.x, lod.x)",
    "       + textureLod(uDisp1, (xz + uOff1) / uCascL.y, lod.y)",
    "       + textureLod(uDisp2, (xz + uOff2) / uCascL.z, lod.z);",
    "}",
    // Surface height above world point xz: invert the horizontal (choppy)
    // displacement by fixed-point iteration, then read the height there.
    // The wind sea dies out in the last couple of metres before the sand (long
    // cascades first); the lagoon keeps its full chop.
    "vec3 depthFade(float depth) { return vec3(sat(depth / 3.0), sat(depth / 1.6), sat((depth + 0.2) / 0.7)); }",
    "vec4 oceanDispF(vec2 xz, vec3 lod, vec3 f) {",
    "  return textureLod(uDisp0, (xz + uOff0) / uCascL.x, lod.x) * f.x",
    "       + textureLod(uDisp1, (xz + uOff1) / uCascL.y, lod.y) * f.y",
    "       + textureLod(uDisp2, (xz + uOff2) / uCascL.z, lod.z) * f.z;",
    "}",
    "float waterHeightF(vec2 xz, vec3 f) {",
    "  vec2 p = xz;",
    "  for (int i = 0; i < 4; i++) p = xz - oceanDispF(p, vec3(0.0), f).xz;",
    "  return oceanDispF(p, vec3(0.0), f).y;",
    "}",
  ].join("\n");

  var C_NOISE = [
    "uniform sampler2D uNoise;",   // 256^2 random texels, linear + repeat
    "vec4 vnoise4(vec2 p) {",       // smooth value noise, one fetch (iq's trick)
    "  vec2 i = floor(p), f = fract(p);",
    "  f = f * f * (3.0 - 2.0 * f);",
    "  return textureLod(uNoise, (i + f + 0.5) / 256.0, 0.0);",
    "}",
    "float vnoise(vec2 p) { return vnoise4(p).x; }",
  ].join("\n");

  // Sky radiance from the atmosphere LUT (upper hemisphere; v = sqrt(y) puts
  // most rows near the horizon, where the sky changes fastest). No sun disk.
  var C_ATMOS = [
    "uniform sampler2D uAtmo;",
    "vec3 atmosphere(vec3 d) {",
    "  vec2 uv = vec2((abs(d.x) + abs(d.z) > 1e-6 ? atan(d.z, d.x) : 0.0) / TAU + 0.5, sqrt(max(d.y, 0.0)));",
    "  return textureLod(uAtmo, uv, 0.0).rgb;",
    "}",
  ].join("\n");

  // Single-scattering (Nishita) sky, integrated once per sun position into a
  // small LUT. Quadratic step spacing keeps the near, dense, optically thick
  // part of horizontal rays well sampled (uniform steps turn the horizon yellow).
  var FS_ATMO = HEAD + [
    "uniform vec3 uSunDir;",
    "uniform float uSunI, uSkyBoost;",
    "const float RP = 6371e3, RA = 6471e3, KM = 21e-6, HR = 8e3, HM = 1.2e3, GM = 0.758;",
    "const vec3 KR = vec3(5.802e-6, 13.558e-6, 33.1e-6);",
    "float exitDist(vec3 r0, vec3 rd) { float b = dot(rd, r0), c = dot(r0, r0) - RA * RA; return -b + sqrt(max(b * b - c, 0.0)); }",
    "in vec2 vUv;",
    "out vec4 o;",
    "void main() {",
    "  float phi = (vUv.x - 0.5) * TAU, y = vUv.y * vUv.y, s = sqrt(max(1.0 - y * y, 0.0));",
    "  vec3 rd = vec3(cos(phi) * s, y, sin(phi) * s);",
    "  vec3 r0 = vec3(0.0, RP + 2.0, 0.0);",
    "  float tMax = exitDist(r0, rd);",
    "  float mu = dot(rd, uSunDir), gg = GM * GM;",
    "  float pR = 3.0 / (16.0 * PI) * (1.0 + mu * mu);",
    "  float pM = 3.0 / (8.0 * PI) * ((1.0 - gg) * (mu * mu + 1.0)) / (pow(1.0 + gg - 2.0 * mu * GM, 1.5) * (2.0 + gg));",
    "  float t = 0.0, oR = 0.0, oM = 0.0;",
    "  vec3 sR = vec3(0.0), sM = vec3(0.0);",
    "  for (int i = 0; i < 16; i++) {",
    "    float t1 = tMax * pow(float(i + 1) / 16.0, 2.0), st = t1 - t;",
    "    vec3 p = r0 + rd * (t + 0.5 * st);",
    "    float h = length(p) - RP;",
    "    float dR = exp(-h / HR) * st, dM = exp(-h / HM) * st;",
    "    oR += 0.5 * dR; oM += 0.5 * dM;",
    "    float L = exitDist(p, uSunDir), jR = 0.0, jM = 0.0, j0 = 0.0;",
    "    for (int j = 0; j < 6; j++) {",
    "      float j1 = L * pow(float(j + 1) / 6.0, 2.0);",
    "      float hh = length(p + uSunDir * (0.5 * (j0 + j1))) - RP;",
    "      jR += exp(-hh / HR) * (j1 - j0); jM += exp(-hh / HM) * (j1 - j0);",
    "      j0 = j1;",
    "    }",
    "    vec3 att = exp(-(KM * 1.1 * (oM + jM) + KR * (oR + jR)));",
    "    sR += dR * att; sM += dM * att;",
    "    oR += 0.5 * dR; oM += 0.5 * dM; t = t1;",
    "  }",
    "  o = vec4(uSkyBoost * uSunI * (pR * KR * sR + pM * KM * sM), 1.0);",
    "}"].join("\n");

  // A flat layer of fair-weather cumulus 1.6 km up.
  var C_CLOUDS = [
    "uniform vec2 uCloudOff;",
    "uniform float uCloudCover;",
    "float cloudFbm(vec2 p) {",
    "  float s = 0.0, a = 0.5;",
    "  for (int i = 0; i < 5; i++) { s += a * vnoise(p); p = mat2(1.6, 1.2, -1.2, 1.6) * p + vec2(17.3, 9.1); a *= 0.5; }",
    "  return s / 0.96875;",
    "}",
    "vec4 clouds(vec3 ro, vec3 rd) {",
    "  if (rd.y < 0.01 || uCloudCover <= 0.001) return vec4(0.0);",
    "  float t = (1600.0 - ro.y) / rd.y;",
    "  vec2 p = (ro.xz + rd.xz * t + uCloudOff) / 1100.0;",
    "  float th = mix(0.74, 0.36, uCloudCover);",
    "  float n = cloudFbm(p);",
    "  float a = smoothstep(th, th + 0.17, n);",
    "  if (a <= 0.0) return vec4(0.0);",
    "  float toward = cloudFbm(p + normalize(uSunDir.xz + 1e-4) * 0.22);",
    "  float lit = sat(1.0 - (toward - th) * 2.2);",
    "  float mu = dot(rd, uSunDir);",
    "  vec3 sky = atmosphere(vec3(0.0, 1.0, 0.0));",
    "  vec3 col = mix(sky * 1.1 + uSunCol * 0.035, uSunCol * 0.15 + sky * 0.6, lit);",
    "  col += uSunCol * 0.08 * pow(max(mu, 0.0), 8.0) * (1.0 - a * 0.6);",   // silver lining
    "  a *= smoothstep(0.015, 0.2, rd.y);",
    "  col = mix(col, atmosphere(rd), 1.0 - exp(-t / 12000.0));",
    "  return vec4(col * a, a);",
    "}",
  ].join("\n");

  var C_ENV = [
    "uniform sampler2D uEnv;",     // equirect sky (no sun disk), mipmapped
    // atan(0, 0) is undefined (NaN on some GPUs): straight up/down gets any longitude
    "vec2 dirToEquirect(vec3 d) { return vec2((abs(d.x) + abs(d.z) > 1e-6 ? atan(d.z, d.x) : 0.0) / TAU + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI); }",
    "vec3 envLod(vec3 d, float lod) { return textureLod(uEnv, dirToEquirect(d), lod).rgb; }",
    "vec3 skyIrradiance() { return PI * textureLod(uEnv, vec2(0.5, 0.25), 7.0).rgb; }",
  ].join("\n");

  // The island tile: height, coral mask, coast distance, sand tone (2 m texels).
  var C_WORLD = [
    "uniform sampler2D uWorld;",
    "uniform vec2 uWorldUV0;",     // uv of the grid origin O
    "uniform float uWorldInv;",    // 1 / tile size
    "vec2 worldUV(vec2 p) { return p * uWorldInv + uWorldUV0; }",
    "vec4 worldAt(vec2 p, float lod) { return textureLod(uWorld, worldUV(p), lod); }",
  ].join("\n");

  // Shore waves: a 9.5 s swell train whose crests are level sets of the
  // arrival time (phase phi, world.js). Its profile morphs with the breaking
  // stage s: a sharpening, front-steepened swell (s < 1), a plunging lip
  // thrown forward over the face (s ~ 1-1.4), then a whitewater bore with a
  // long sloping back that runs up the beach as swash (s -> 2).
  var C_SHORE = [
    "uniform sampler2D uShore, uShoreK;",
    "uniform vec2 uShoreWT, uGroupWT;",   // cos/sin of the swell and wave-group phases now
    "uniform float uSurf;",                // surf height scale (0 = off)
    // (forward lean, height) in units of H at phase th (crest at 0, front face < 0).
    // As the wave shoals it turns cnoidal: a narrow peaked crest over long
    // flat troughs, the front face narrower than the back, the crest leaning
    // shoreward. At the break the top throws forward over the face (the mesh
    // folds, making a real hollow), then the wave becomes a bore: a steep
    // turbulent front with a long sloping back.
    "vec2 surfProfile(float th, float s) {",
    "  th -= TAU * floor((th + PI) / TAU);",   // keep the phase in [-pi, pi)
    "  float steep = smoothstep(0.15, 1.0, s);",
    "  float wf = mix(1.1, 0.26, steep), wb = mix(1.1, 0.85, steep);",
    "  float u = th / (th < 0.0 ? wf : wb);",
    "  float pulse = exp(-u * u);",
    "  float c0 = (wf + wb) * 0.14105;",          // mean of the pulse over a period
    "  float y = mix(0.5 * cos(th), pulse - c0, sat(steep * 1.5));",
    "  float top = pulse * pulse;",
    "  float x = -0.22 * (1.0 - 0.6 * steep) * sin(th) + 0.3 * steep * top;",
    "  float plunge = smoothstep(0.85, 1.02, s) * (1.0 - smoothstep(1.2, 1.5, s));",
    "  float lip = top * top;",
    "  x += plunge * (0.6 * top + 0.5 * lip);",
    "  y -= plunge * 0.22 * lip * smoothstep(-0.05, 0.3, th);",
    "  float bf = th < 0.0 ? smoothstep(-0.32, 0.0, th) : pow(max(1.0 - th / PI, 0.0), 1.8);",
    "  vec2 bore = vec2(0.25 * exp(-th * th / 0.03), 0.75 * bf - 0.2);",
    "  return mix(vec2(x, y), bore, smoothstep(1.2, 1.55, s));",
    "}",
    "struct Surf { vec3 d; vec2 slope; vec4 jac; float th, s, H, foam; vec2 n; };",
    // Displacement, slopes (dY/dx, dY/dz), horizontal Jacobian (Dxx, Dzz, Dxz,
    // Dzx) and foam at undisplaced point p.
    "Surf shoreWave(vec2 p, float lod) {",
    "  Surf r; r.d = vec3(0.0); r.slope = vec2(0.0); r.jac = vec4(0.0); r.th = 0.0; r.s = 0.0; r.H = 0.0; r.foam = 0.0; r.n = vec2(1.0, 0.0);",
    "  vec4 a = textureLod(uShore, worldUV(p), lod);",
    "  float coh = length(a.xy);",         // mip-averaged phases lose coherence far away
    "  float H = a.z * uSurf * sat(coh * 1.3 - 0.2);",
    "  if (H < 0.004) return r;",
    "  vec4 b = textureLod(uShoreK, worldUV(p), lod);",
    "  H *= 0.72 + 0.28 * (uGroupWT.x * b.z + uGroupWT.y * b.w) / max(length(b.zw), 1e-3);",   // sets
    "  float th = atan(uShoreWT.y * a.x - uShoreWT.x * a.y, uShoreWT.x * a.x + uShoreWT.y * a.y);",
    "  vec2 K = b.xy, n = K / max(length(K), 1e-5);",
    "  float s = a.w;",
    "  vec2 f = surfProfile(th, s) * H;",
    "  vec2 df = (surfProfile(th + 0.012, s) - surfProfile(th - 0.012, s)) * (H / 0.024);",
    // theta = w t - phi, so d(theta)/dp = -K
    "  r.d = vec3(n.x * f.x, f.y, n.y * f.x);",
    "  r.slope = -df.y * K;",
    "  r.jac = -df.x * vec4(n.x * K.x, n.y * K.y, n.x * K.y, n.y * K.x);",
    "  r.th = th; r.s = s; r.H = H; r.n = n;",
    "  float crest = smoothstep(0.88, 1.05, s) * exp(-(th + 0.05) * (th + 0.05) / 0.08);",
    "  float bore = smoothstep(1.05, 1.3, s) * (th < 0.0 ? smoothstep(-0.32, -0.02, th) : exp(-th * 0.38));",
    "  r.foam = sat(max(crest, bore) + smoothstep(1.0, 1.7, s) * 0.15) * sat(H * 4.0);",
    "  return r;",
    "}",
  ].join("\n");

  // The surf-zone shallow-water simulation (see swe.js for the scheme): a
  // 205 m grid of 0.4 m cells that follows the camera. State texture: water
  // depth h, x-face velocity u, z-face velocity w, foam. Aux: lip launch
  // strength, flow direction, sand wetness.
  var C_SWE = [
    "uniform sampler2D uSwe;",
    "uniform vec2 uSweC;",          // O-relative centre of the grid
    "uniform float uSweSize, uSweOn;",
    "vec2 sweUV(vec2 p) { return (p - uSweC) / uSweSize + 0.5; }",
    "float sweMask(vec2 p) {",
    "  if (uSweOn < 0.5) return 0.0;",
    "  vec2 uv = sweUV(p), e = min(uv, 1.0 - uv);",
    "  return smoothstep(0.03, 0.09, min(e.x, e.y));",
    "}",
    // manual bilinear (float textures need not be filterable); t in texels, texel i at i
    "vec4 texBilin(sampler2D tex, vec2 t) {",
    "  ivec2 m = textureSize(tex, 0) - 1, i = ivec2(floor(t));",
    "  vec2 f = t - floor(t);",
    "  vec4 a = texelFetch(tex, clamp(i, ivec2(0), m), 0), b = texelFetch(tex, clamp(i + ivec2(1, 0), ivec2(0), m), 0);",
    "  vec4 c = texelFetch(tex, clamp(i + ivec2(0, 1), ivec2(0), m), 0), d = texelFetch(tex, clamp(i + ivec2(1, 1), ivec2(0), m), 0);",
    "  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);",
    "}",
    "vec4 sweAt(vec2 p) { return texBilin(uSwe, sweUV(p) * vec2(textureSize(uSwe, 0)) - 0.5); }",
    // the rendered surface: depth over the bed; a vanishing film tucks under the sand
    "float sweEta(float h, float bed) { return bed + h - 0.04 * (1.0 - smoothstep(0.0, 0.06, h)); }",
  ].join("\n");

  // Target state for the simulation: the analytic swell (height, and the
  // long-wave orbital velocity c eta / d along its direction of travel).
  var C_SWE_TARGET = [
    "vec4 sweTarget(vec2 pos, float bed) {",
    "  Surf s = shoreWave(pos, 0.0);",
    "  float eta = s.d.y, d = max(-bed, 0.0), h = max(eta - bed, 0.0);",
    "  float dd = max(d, 0.3), c = sqrt(9.81 * dd);",
    "  vec2 v = d > 0.3 ? s.n * c * eta / dd : vec2(0.0);",
    "  return vec4(h, v, 0.0);",
    "}",
  ].join("\n");

  // Pass 1: shift the grid with the camera (new cells start from the target),
  // then advect the face velocities and the foam semi-Lagrangian.
  var FS_SWE_ADV = HEAD + C_COMMON + "\n" + C_WORLD + "\n" + C_SHORE + "\n" + C_SWE_TARGET + "\n" + [
    "uniform sampler2D uSweIn;",
    "uniform ivec2 uShift;",
    "uniform vec2 uSweC;",
    "uniform float uCell, uDt;",
    "out vec4 o;",
    "vec4 F(ivec2 q) { ivec2 m = textureSize(uSweIn, 0) - 1; return texelFetch(uSweIn, clamp(q, ivec2(0), m), 0); }",
    "vec4 samp(vec2 t) {",
    "  ivec2 i = ivec2(floor(t)); vec2 f = t - floor(t);",
    "  return mix(mix(F(i), F(i + ivec2(1, 0)), f.x), mix(F(i + ivec2(0, 1)), F(i + ivec2(1, 1)), f.x), f.y);",
    "}",
    // velocity at a point in cell-corner space (cell i spans [i, i+1])
    "vec2 vel(vec2 X) { return vec2(samp(X - vec2(1.0, 0.5)).g, samp(X - vec2(0.5, 1.0)).b); }",
    "void main() {",
    "  ivec2 p = ivec2(gl_FragCoord.xy), n = textureSize(uSweIn, 0), q = p + uShift;",
    "  if (q.x < 0 || q.y < 0 || q.x >= n.x || q.y >= n.y) {",
    "    vec2 pos = uSweC + (vec2(p) + 0.5 - 0.5 * vec2(n)) * uCell;",
    "    o = sweTarget(pos, worldAt(pos, 0.0).x);",
    "    return;",
    "  }",
    "  float k = uDt / uCell;",
    "  vec2 Xu = vec2(q) + vec2(1.0, 0.5), Xw = vec2(q) + vec2(0.5, 1.0), Xc = vec2(q) + 0.5;",
    "  float u = samp(Xu - vel(Xu) * k - vec2(1.0, 0.5)).g;",
    "  float w = samp(Xw - vel(Xw) * k - vec2(0.5, 1.0)).b;",
    "  float foam = samp(Xc - vel(Xc) * k - 0.5).a;",
    "  o = vec4(F(q).r, u, w, foam);",
    "}",
  ].join("\n");

  // Pass 2: depth from upwind fluxes (conserves water, never negative).
  var FS_SWE_H = HEAD + [
    "uniform sampler2D uSweIn;",
    "uniform float uCell, uDt;",
    "out vec4 o;",
    "vec4 F(ivec2 q) { ivec2 m = textureSize(uSweIn, 0) - 1; return texelFetch(uSweIn, clamp(q, ivec2(0), m), 0); }",
    "void main() {",
    "  ivec2 p = ivec2(gl_FragCoord.xy);",
    "  vec4 c = F(p), l = F(p - ivec2(1, 0)), d = F(p - ivec2(0, 1)), r = F(p + ivec2(1, 0)), t = F(p + ivec2(0, 1));",
    "  float fxR = c.g * (c.g > 0.0 ? c.r : r.r), fxL = l.g * (l.g > 0.0 ? l.r : c.r);",
    "  float fzT = c.b * (c.b > 0.0 ? c.r : t.r), fzB = d.b * (d.b > 0.0 ? d.r : c.r);",
    "  o = vec4(max(0.0, c.r - uDt / uCell * (fxR - fxL + fzT - fzB)), c.gba);",
    "}",
  ].join("\n");

  // Pass 3: pressure gradient on wet faces, friction, speed limit; relax
  // toward the analytic swell in deep water and at the grid border; foam on
  // steep fronts; lip launch strength, flow direction and sand wetness.
  var FS_SWE_VEL = HEAD + C_COMMON + "\n" + C_WORLD + "\n" + C_SHORE + "\n" + C_SWE_TARGET + "\n" + [
    "uniform sampler2D uSweIn, uAuxIn;",
    "uniform ivec2 uAuxShift;",
    "uniform vec2 uSweC;",
    "uniform float uCell, uDt;",
    "layout(location = 0) out vec4 oS;",
    "layout(location = 1) out vec4 oA;",
    "const float DRY = 0.02, FRICTION = 0.003;",
    "vec4 F(ivec2 q) { ivec2 m = textureSize(uSweIn, 0) - 1; return texelFetch(uSweIn, clamp(q, ivec2(0), m), 0); }",
    "void main() {",
    "  ivec2 p = ivec2(gl_FragCoord.xy), n = textureSize(uSweIn, 0);",
    "  vec4 c = F(p), r = F(p + ivec2(1, 0)), t = F(p + ivec2(0, 1)), l = F(p - ivec2(1, 0)), d = F(p - ivec2(0, 1));",
    "  vec2 pos = uSweC + (vec2(p) + 0.5 - 0.5 * vec2(n)) * uCell;",
    "  float b0 = worldAt(pos, 0.0).x, bR = worldAt(pos + vec2(uCell, 0.0), 0.0).x, bT = worldAt(pos + vec2(0.0, uCell), 0.0).x;",
    "  float bL = worldAt(pos - vec2(uCell, 0.0), 0.0).x, bB = worldAt(pos - vec2(0.0, uCell), 0.0).x;",
    "  float e0 = b0 + c.r, eR = bR + r.r, eT = bT + t.r, eL = bL + l.r, eB = bB + d.r;",
    "  float vmax = 0.5 * uCell / uDt;",
    "  float hf = max(e0, eR) - max(b0, bR);",
    "  float u = hf > DRY ? c.g - 9.81 * uDt / uCell * (eR - e0) : 0.0;",
    "  u = clamp(u / (1.0 + uDt * FRICTION * abs(u) / max(hf, 0.05)), -vmax, vmax);",
    "  float hz = max(e0, eT) - max(b0, bT);",
    "  float w = hz > DRY ? c.b - 9.81 * uDt / uCell * (eT - e0) : 0.0;",
    "  w = clamp(w / (1.0 + uDt * FRICTION * abs(w) / max(hz, 0.05)), -vmax, vmax);",
    "  float h = c.r;",
    // the analytic swell drives deep water and the border band
    "  vec2 e = min(vec2(p), vec2(n) - 1.0 - vec2(p)) / float(n.x);",
    "  float border = 1.0 - smoothstep(0.0, 0.06, min(e.x, e.y));",
    "  float rr = 3.0 * smoothstep(2.4, 3.6, -b0) + 10.0 * border;",
    "  if (rr > 0.0) {",
    "    vec4 T = sweTarget(pos, b0);",
    "    float a = 1.0 - exp(-rr * uDt);",
    "    h += (T.r - h) * a; u += (T.g - u) * a; w += (T.b - w) * a;",
    "  }",
    // flow at the cell centre, the surface slope along it
    "  vec2 vc = 0.5 * vec2(u + l.g, w + d.b);",
    "  float sp = length(vc);",
    "  vec2 vn = sp > 1e-3 ? vc / sp : vec2(0.0);",
    "  vec2 ge = vec2(eR - eL, eT - eB) / (2.0 * uCell);",
    "  float steep = h > DRY ? -dot(ge, vn) : 0.0;",
    // foam: steep fronts churn white water, which drifts and fades
    "  float foam = c.a * exp(-uDt / 3.5) + uDt * 2.5 * smoothstep(0.7, 1.8, steep) * smoothstep(0.08, 0.3, h);",
    "  foam = min(foam, 3.0);",
    // aux: sand wetness (stays dark a few seconds after the swash recedes)
    "  vec4 ap = texelFetch(uAuxIn, clamp(p + uAuxShift, ivec2(0), n - 1), 0);",
    "  float wet = ap.a * exp(-uDt / 6.0);",
    "  if (h > DRY && b0 > -0.2) wet = 1.0;",
    // lip launch: a steep front standing well above still water, moving
    // shoreward, in the plunging band of the analytic surf, stronger where
    // the bed rises steeply under it (plunging rather than spilling)
    "  vec4 sh = textureLod(uShore, worldUV(pos), 0.0);",
    "  vec2 sn = normalize(textureLod(uShoreK, worldUV(pos), 0.0).xy + 1e-6);",
    "  float bedRise = dot(vec2(bR - bL, bT - bB) / (2.0 * uCell), vn);",
    "  float band = smoothstep(0.75, 0.92, sh.w) * (1.0 - smoothstep(1.3, 1.55, sh.w));",
    "  float lip = smoothstep(0.7, 1.5, steep) * smoothstep(0.3, 0.75, e0) * band * step(0.0, dot(vn, sn)) * (0.35 + 0.65 * smoothstep(0.03, 0.25, bedRise));",
    "  oS = vec4(h, u, w, foam);",
    "  oA = vec4(lip, vn, wet);",
    "}",
  ].join("\n");

  // Interactive ripples: a wave-equation height field on a 102 m grid that
  // follows the camera, added on top of the ocean (r height, g velocity,
  // b foam). Zero outside the grid.
  var C_RIPPLE = [
    "uniform sampler2D uRip;",
    "uniform vec2 uRipC;",          // O-relative centre of the grid
    "uniform float uRipSize;",      // metres across
    "vec2 ripUV(vec2 p) { return (p - uRipC) / uRipSize + 0.5; }",
    "float ripMask(vec2 uv) { vec2 e = min(uv, 1.0 - uv); return smoothstep(0.0, 0.06, min(e.x, e.y)); }",
    "vec4 ripple(vec2 p) { vec2 uv = ripUV(p); return textureLod(uRip, uv, 0.0) * ripMask(uv); }",
    "vec2 rippleSlope(vec2 p) {",
    "  float e = uRipSize / float(textureSize(uRip, 0).x);",
    "  return vec2(ripple(p + vec2(e, 0.0)).r - ripple(p - vec2(e, 0.0)).r, ripple(p + vec2(0.0, e)).r - ripple(p - vec2(0.0, e)).r) / (2.0 * e);",
    "}",
  ].join("\n");
  var FS_RIPPLE = HEAD + C_COMMON + "\n" + C_WORLD + "\n" + [
    "uniform sampler2D uRipPrev;",
    "uniform ivec2 uShift;",        // texels the grid moved since last step
    "uniform vec2 uRipC;",
    "uniform float uCell, uDt, uCmax;",
    "uniform vec4 uDrops[24];",     // x, z (O-relative), radius, velocity kick
    "uniform float uDropFoam[24];",
    "uniform int uNDrops;",
    "out vec4 o;",
    "vec4 at(ivec2 q) {",
    "  ivec2 n = textureSize(uRipPrev, 0);",
    "  if (q.x < 0 || q.y < 0 || q.x >= n.x || q.y >= n.y) return vec4(0.0);",
    "  return texelFetch(uRipPrev, q, 0);",
    "}",
    "void main() {",
    "  ivec2 p = ivec2(gl_FragCoord.xy), q = p + uShift, n = textureSize(uRipPrev, 0);",
    "  vec4 c = at(q);",
    "  float h = c.r, v = c.g, foam = c.b, glow = c.a;",
    "  vec2 pos = uRipC + (vec2(p) + 0.5 - 0.5 * vec2(n)) * uCell;",
    "  float depth = -worldAt(pos, 0.0).x;",
    "  if (depth < 0.06) { o = vec4(0.0, 0.0, foam * 0.97, glow * 0.9); return; }",   // the shore reflects
    "  float cc = min(uCmax, sqrt(9.81 * depth));",
    "  float lap = (at(q + ivec2(1, 0)).r + at(q - ivec2(1, 0)).r + at(q + ivec2(0, 1)).r + at(q - ivec2(0, 1)).r - 4.0 * h) / (uCell * uCell);",
    "  v += cc * cc * lap * uDt;",
    "  for (int i = 0; i < 24; i++) {",
    "    if (i >= uNDrops) break;",
    "    vec4 d = uDrops[i];",
    "    float w = exp(-dot(pos - d.xy, pos - d.xy) / (d.z * d.z));",
    "    v += d.w * w; foam += uDropFoam[i] * w; glow += (abs(d.w) * 0.3 + uDropFoam[i] * 1.5) * w;",
    "  }",
    // absorbing border, gentle loss everywhere
    "  vec2 e = min(vec2(p), vec2(n) - 1.0 - vec2(p)) / float(n.x);",
    "  float border = (1.0 - smoothstep(0.0, 0.07, min(e.x, e.y)));",
    "  v *= exp(-uDt * (0.35 + 9.0 * border));",
    "  h *= exp(-uDt * (0.04 + 9.0 * border));",
    "  h += v * uDt;",
    // steep, fast ripples whiten
    "  foam += max(abs(v) - 0.6, 0.0) * uDt * 3.0;",
    "  foam = min(foam * exp(-uDt * 0.4), 2.5);",
    // plankton flash when the water moves past them and fade within a second or two
    "  glow = min((glow + abs(v) * 3.0 * uDt) * exp(-uDt * 1.3), 3.0);",
    "  o = vec4(h, v, foam, glow);",
    "}",
  ].join("\n");

  // Sun shadow map of palms, trees and objects near the camera (ortho, follows
  // the camera); outside it everything counts as lit.
  var C_SHADOW = [
    "uniform sampler2DShadow uShadowMap;",
    "uniform mat4 uShadowMat;",    // O-relative world -> shadow map (0..1)
    "uniform float uShadowOn;",
    "float sunVis(vec3 p, float bias) {",
    "  if (uShadowOn < 0.5) return 1.0;",
    "  vec4 q = uShadowMat * vec4(p, 1.0);",
    "  vec2 e2 = abs(q.xy - 0.5) * 2.0;",
    "  float edge = smoothstep(0.85, 0.98, max(e2.x, e2.y));",
    "  if (edge >= 1.0 || q.z >= 1.0) return 1.0;",
    "  float d = q.z - bias, k = 1.0 / 2048.0;",
    "  float v = texture(uShadowMap, vec3(q.xy, d)) * 0.4",
    "          + (texture(uShadowMap, vec3(q.xy + vec2(k, 0.0), d)) + texture(uShadowMap, vec3(q.xy - vec2(k, 0.0), d))",
    "          +  texture(uShadowMap, vec3(q.xy + vec2(0.0, k), d)) + texture(uShadowMap, vec3(q.xy - vec2(0.0, k), d))) * 0.15;",
    "  return mix(v, 1.0, edge);",
    "}",
  ].join("\n");

  // Coral rock up close: lumps, knobs and crevices on reef ground, with finer
  // octaves switched on as the camera approaches (the heightfield's 2 m
  // texels would otherwise leave smooth domes).
  var C_CORALROCK = [
    "uniform vec2 uOffR;",
    "float coralLump(vec2 p, float coral, float dist) {",
    "  if (coral < 0.05) return 0.0;",
    "  float fade = 1.0 - smoothstep(30.0, 55.0, dist);",
    "  if (fade <= 0.0) return 0.0;",
    "  vec2 q = p + uOffR;",
    "  float l = (vnoise(q * 0.55) - 0.5) * 1.0 + (vnoise(q * 1.7 + 3.1) - 0.5) * 0.45;",
    "  l += (vnoise(q * 4.6 + 7.7) - 0.5) * 0.18 * (1.0 - smoothstep(6.0, 18.0, dist));",
    "  l += (vnoise(q * 11.0 + 1.3) - 0.5) * 0.07 * (1.0 - smoothstep(3.0, 9.0, dist));",
    "  return l * smoothstep(0.05, 0.5, coral) * fade;",
    "}",
  ].join("\n");

  // Ground shading shared by the terrain pass and the floor seen through the water.
  var C_WATER = [
    "uniform vec3 uSigT, uSigS;",   // extinction / scattering per metre
    "uniform vec3 uSunRefr;",       // sun direction after refraction into the water (pointing down)
    "uniform float uSunTrans;",     // Fresnel transmittance of the sun through flat water
    "uniform vec2 uOffD;",          // detail-noise offset
    "uniform float uMaxH;",
    "uniform sampler2D uCaustic;",
    "uniform float uCausL, uCausDepth, uCausStr;",
    "uniform vec2 uOffC, uCausShift;",
    "float groundH(vec2 p) { return worldAt(p, 0.0).x; }",
    // central differences one mip texel apart (2 m at lod 0)
    "vec3 groundNormal(vec2 p, float lod) {",
    "  float e = 2.0 * exp2(lod);",
    "  float hx = worldAt(p + vec2(e, 0.0), lod).x - worldAt(p - vec2(e, 0.0), lod).x;",
    "  float hz = worldAt(p + vec2(0.0, e), lod).x - worldAt(p - vec2(0.0, e), lod).x;",
    "  return normalize(vec3(-hx, 2.0 * e, -hz));",
    "}",
    // Refracted ray from a surface point down to the ground: secant iteration
    // on the gap between ray and terrain, which shrinks monotonically for the
    // steep rays refraction produces.
    "float groundGap(vec3 P, vec3 D, float t) { return P.y + D.y * t - groundH(P.xz + D.xz * t); }",
    "float traceGround(vec3 P, vec3 D) {",
    "  D.y = min(D.y, -0.05);",          // a folded lip can bend the ray level or upward
    "  float t0 = max((P.y - groundH(P.xz)) / -D.y, 0.0), f0 = groundGap(P, D, t0);",
    "  float t1 = t0 + f0 / -D.y, f1 = groundGap(P, D, t1);",
    "  for (int i = 0; i < 3; i++) {",
    "    float d = f1 - f0;",
    "    if (abs(d) < 1e-5) break;",
    "    float t2 = t1 - f1 * (t1 - t0) / d;",
    "    t0 = t1; f0 = f1; t1 = t2; f1 = groundGap(P, D, t1);",
    "  }",
    "  return clamp(t1, 0.0, 400.0);",
    "}",
    // Any ray, including grazing ones from under water: sphere-trace the
    // heightfield (vertical clearance is a safe step on gentle slopes).
    "float traceGroundMarch(vec3 P, vec3 D, float tMax) {",
    "  float t = 0.0;",
    "  for (int i = 0; i < 56; i++) {",
    "    vec3 q = P + D * t;",
    "    float h = q.y - groundH(q.xz);",
    "    if (h < 0.02) return t;",
    "    t += max(h * 0.8, 0.05);",
    "    if (t > tMax) break;",
    "  }",
    "  return -1.0;",
    "}",
    // Soft sun shadow from the island: march the heightfield toward the sun
    // with growing steps; the penumbra widens with distance like a real one.
    // Distant terrain is drawn from a smoothed mip (lod0), so the ray starts
    // on the detailed surface and never samples finer than what was drawn.
    "float terrainShadow(vec3 p, float lod0) {",
    "  if (uSunDir.y < 0.01) return 0.0;",
    "  p.y = max(p.y, worldAt(p.xz, lod0).x) + 0.3 + 0.6 * exp2(lod0);",
    "  float s = 1.0, t = 1.5;",
    "  for (int i = 0; i < 20; i++) {",
    "    vec3 q = p + uSunDir * t;",
    "    if (q.y > uMaxH) break;",
    "    float hq = worldAt(q.xz, clamp(max(log2(t / 12.0), lod0), 0.0, 6.0)).x;",
    "    s = min(s, sat((q.y - hq) / (0.05 * t + 0.4)));",
    "    t = t * 1.38 + 1.2;",
    "  }",
    "  return s;",
    "}",
    // Downwelling light at depth z and the radiance it scatters toward the eye.
    "vec3 downwelling(float z) {",
    "  vec3 sun = uSunCol * uSunTrans * exp(-uSigT * z / max(-uSunRefr.y, 0.2));",
    "  vec3 sky = skyIrradiance() * exp(-uSigT * z * 1.2);",
    "  return sun * max(-uSunRefr.y, 0.0) + sky * 0.5;",
    "}",
    // x2.4: light bounced off the white floor and multiple scattering roughly
    // double what single scattering of the downwelling light would give.
    "vec3 inscatter(float zMid) { return uSigS / uSigT * downwelling(zMid) * (2.4 * 1.6 / (4.0 * PI)); }",
    // Water between two points, d metres apart, around mean depth zMid.
    "vec3 throughWater(vec3 L, float d, float zMid) {",
    "  vec3 T = exp(-uSigT * d);",
    "  return L * T + inscatter(zMid) * (1.0 - T);",
    "}",
    // Sand: wave ripples (a 60 cm train and a 14 cm one, bent by noise).
    // Ripples fade with distance before they can alias.
    "vec3 sandNormal(vec2 w, vec3 n, float dist, float amp) {",
    "  float fade = (1.0 - smoothstep(6.0, 60.0, dist)) * amp;",
    "  if (fade <= 0.0) return n;",
    "  float warp = vnoise(w * 0.25) * 5.0 + vnoise(w * 1.0 + 7.0) * 1.2;",
    "  vec2 d1 = vec2(0.8, 0.6), d2 = vec2(-0.45, 0.89);",
    "  float k1 = TAU / 0.6, k2 = TAU / 0.14;",
    "  float p1 = dot(w, d1) * k1 + warp, p2 = dot(w, d2) * k2 + warp * 2.0;",
    // asymmetric ripple profile: gentle stoss side, steeper lee side
    "  float f2 = 1.0 - smoothstep(2.0, 14.0, dist);",
    "  vec2 g = d1 * (cos(p1) + 0.4 * cos(2.0 * p1)) * 0.32 + d2 * (cos(p2) + 0.3 * cos(2.0 * p2)) * 0.16 * f2;",
    "  g *= (0.45 + 0.55 * vnoise(w * 0.125 + 3.0)) * fade;",
    "  return normalize(n + vec3(-g.x, 0.0, -g.y));",
    "}",
    // White coral sand, faintly mottled, with sparse tufts of seagrass.
    "vec3 sandAlbedo(vec2 w, float tone) {",
    "  vec3 alb = vec3(0.93, 0.90, 0.82) * (0.9 + 0.1 * tone);",
    "  alb *= 0.93 + 0.07 * vnoise(w * 4.0) + 0.05 * (vnoise(w * 16.0) - 0.5);",
    "  float grass = (1.0 - smoothstep(-0.62, -0.52, tone)) * smoothstep(0.35, 0.75, vnoise(w * 2.0 + 11.0));",
    "  return mix(alb, vec3(0.16, 0.20, 0.09), grass * 0.85);",
    "}",
    // Ground albedo from the world data: beach sand (darker where wet), jungle
    // on the hills, basalt on steep faces and the summit, coral on the reef.
    // Returns wetness in .w-style via the out parameter.
    "vec3 groundAlbedo(vec2 w, vec4 wd, vec3 n, out float wet) {",
    "  float h = wd.x, slope = 1.0 - n.y;",
    "  vec3 sand = sandAlbedo(w, wd.w);",
    "  wet = h < 0.0 ? 1.0 : (1.0 - smoothstep(0.6, 2.2, h)) * (1.0 - smoothstep(4.0, 14.0, -wd.z));",
    "  if (h < 0.0) {",
    "    float cm = smoothstep(0.25, 0.75, wd.y * (0.75 + 0.6 * vnoise(w * 0.21)));",
    // coral rock: tan, mauve and ochre, with patches of green algae
    "    vec3 coral = mix(vec3(0.56, 0.47, 0.38), vec3(0.50, 0.38, 0.44), vnoise(w * 0.37));",
    "    coral = mix(coral, vec3(0.62, 0.50, 0.26), smoothstep(0.55, 0.85, vnoise(w * 0.8 + 5.0)));",
    "    coral *= 0.6 + 0.55 * vnoise(w * 1.6 + 3.0) * (0.7 + 0.3 * vnoise(w * 6.0));",
    "    coral = mix(coral, vec3(0.28, 0.40, 0.24), smoothstep(0.62, 0.9, vnoise(w * 0.9 + 9.0)) * 0.8);",
    "    return mix(sand, coral, cm);",
    "  }",
    "  vec3 land = mix(sand, sand * vec3(0.6, 0.6, 0.64), wet);",
    "  float scrub = smoothstep(2.3, 3.6, h + 1.2 * vnoise(w * 0.07)) * (1.0 - smoothstep(0.7, 0.9, slope));",
    // rainforest canopy: dark, mottled at the scale of crowns and of groves
    "  float crowns = vnoise(w * 0.35) * 0.6 + vnoise(w * 1.3) * 0.4;",
    "  vec3 jungle = mix(vec3(0.028, 0.06, 0.018), vec3(0.075, 0.12, 0.03), vnoise(w * 0.045));",
    "  jungle = mix(jungle, vec3(0.11, 0.13, 0.04), smoothstep(0.62, 0.9, vnoise(w * 0.11 + 5.0)) * 0.6);",   // paler groves
    "  jungle *= 0.6 + 0.8 * crowns;",
    "  jungle = mix(jungle, vec3(0.22, 0.24, 0.11), (1.0 - smoothstep(2.8, 5.5, h)) * 0.65);",  // beach scrub
    "  land = mix(land, jungle, scrub);",
    "  float rock = max(smoothstep(0.58, 0.8, slope + 0.12 * (vnoise(w * 0.05) - 0.5)), smoothstep(190.0, 225.0, h)) * smoothstep(0.8, 2.0, h);",
    "  vec3 basalt = mix(vec3(0.10, 0.095, 0.09), vec3(0.24, 0.22, 0.19), vnoise(w * 0.31)) * (0.8 + 0.4 * vnoise(w * 2.1));",
    "  basalt = mix(basalt, jungle * 1.3, smoothstep(0.55, 0.8, vnoise(w * 0.6)) * (1.0 - smoothstep(0.85, 0.95, slope)) * 0.7);",   // vines on the rock
    "  return mix(land, basalt, rock);",
    "}",
    "float causticAt(vec2 uv, float lod) { return textureLod(uCaustic, uv, lod).r; }",
    // Light on ground at p with normal n under z metres of water (0 = dry):
    // in air, the sun times the island's shadow plus sky; under water, the
    // refracted sun through the caustic pattern and attenuated sky. The two
    // blend over the first 30 cm so the waterline has no seam.
    "vec3 lightGround(vec3 p, vec3 n, vec3 alb, float z, float dist, float shadow) {",
    "  vec3 sky = skyIrradiance() * (0.55 + 0.45 * n.y);",
    "  vec3 air = alb * (uSunCol * max(dot(n, uSunDir), 0.0) * shadow + sky) / PI;",
    "  if (z <= 0.0) return air;",
    "  vec2 cuv = (p.xz + uOffC - uCausShift * z) / uCausL;",
    "  vec2 disp = uCausShift * z * 0.012 / uCausL;",
    "  float lod = log2(1.0 + dist * 0.05) + abs(z - uCausDepth) * 0.25;",
    "  vec3 c = vec3(causticAt(cuv - disp, lod), causticAt(cuv, lod), causticAt(cuv + disp, lod));",
    "  c = mix(vec3(1.0), c, uCausStr * smoothstep(0.1, 0.9, z) * (1.0 - smoothstep(18.0, 30.0, z)));",
    "  vec3 sunAtFloor = uSunCol * uSunTrans * exp(-uSigT * z / max(-uSunRefr.y, 0.2)) * shadow;",
    "  vec3 wet = alb * (sunAtFloor * max(dot(n, -uSunRefr), 0.0) * c + sky * exp(-uSigT * z * 1.2)) / PI;",
    "  return mix(air, wet, sat(z / 0.3));",
    "}",
    // Radiance leaving the ground at `hit`, seen from `dist` metres away
    // through the water (shadow comes from the surface point above it).
    "vec3 shadeFloor(vec3 hit, float dist, float shadow) {",
    "  float lod = clamp(log2(1.0 + dist / 40.0), 0.0, 4.0);",
    "  vec4 wd = worldAt(hit.xz, 0.0);",
    "  vec2 w = hit.xz + uOffD;",
    "  vec3 n = groundNormal(hit.xz, lod);",
    "  float wet;",
    "  vec3 alb = groundAlbedo(w, wd, n, wet);",
    "  n = sandNormal(w, n, dist, 1.0 - smoothstep(0.2, 0.6, wd.y));",
    // under water the sand reads like wet sand at the waterline, then brightens
    "  float z = max(-hit.y, 0.0);",
    "  alb *= mix(0.68, 1.0, smoothstep(0.0, 1.2, z));",
    "  return lightGround(hit, n, alb, z, dist, shadow * sunVis(hit, 0.002));",
    "}",
    // Distant ground fades into the horizon haze.
    "vec3 aerial(vec3 col, vec3 dir, float dist) {",
    "  float f = 1.0 - exp(-dist / 9000.0);",
    "  return mix(col, atmosphere(normalize(vec3(dir.x, max(dir.y, 0.02), dir.z))), f);",
    "}",
  ].join("\n");

  var C_FRESNEL = [
    // Unpolarised Fresnel reflectance, eta = n_incident / n_transmitted.
    "float fresnel(float cosI, float eta) {",
    "  float s2 = eta * eta * (1.0 - cosI * cosI);",
    "  if (s2 >= 1.0) return 1.0;",
    "  float cosT = sqrt(1.0 - s2);",
    "  float rs = (eta * cosI - cosT) / (eta * cosI + cosT);",
    "  float rp = (cosI - eta * cosT) / (cosI + eta * cosT);",
    "  return 0.5 * (rs * rs + rp * rp);",
    "}",
  ].join("\n");

  // Water height under the camera, written to a 1x1 texture each frame so
  // every later pass knows whether the eye is above or below the surface.
  var FS_PROBE = HEAD + C_COMMON + "\n" + C_OCEAN + "\n" + C_WORLD + "\n" + C_SHORE + "\n" + C_RIPPLE + "\n" + C_SWE + "\n" + [
    "uniform vec2 uProbePts[64];",   // O-relative; texel 0 is always the camera
    "out vec4 o;",
    // invert the horizontal displacement, then read the height there
    "float surfaceAt(vec2 xz) {",
    "  vec2 p = xz;",
    "  for (int i = 0; i < 4; i++) {",
    "    vec3 f = depthFade(-worldAt(p, 0.0).x);",
    "    p = xz - oceanDispF(p, vec3(0.0), f).xz - shoreWave(p, 0.0).d.xz * (1.0 - sweMask(p));",
    "  }",
    "  float sm = sweMask(p), bed = worldAt(p, 0.0).x;",
    "  float shore = mix(shoreWave(p, 0.0).d.y, bed + sweAt(p).r, sm);",
    "  return oceanDispF(p, vec3(0.0), depthFade(-bed)).y + shore + ripple(p).r;",
    "}",
    "void main() {",
    "  int i = int(gl_FragCoord.x);",
    "  vec2 xz = i == 0 ? uCam.xz : uProbePts[i];",
    // height, and the surf's current (cell-centre velocity) and how much of it is simulated here
    "  vec4 st = sweAt(xz), stL = sweAt(xz - vec2(0.4, 0.0)), stB = sweAt(xz - vec2(0.0, 0.4));",
    "  float sm = sweMask(xz);",
    "  o = vec4(surfaceAt(xz), 0.5 * vec2(st.g + stL.g, st.b + stB.b) * sm, sm);",
    "}",
  ].join("\n");

  var FS_ENV = HEAD + C_COMMON + "\n" + C_NOISE + "\n" + C_ATMOS + "\n" + C_CLOUDS + "\n" + [
    "in vec2 vUv;",
    "out vec4 o;",
    "void main() {",
    "  float phi = (vUv.x - 0.5) * TAU, th = vUv.y * PI;",
    "  vec3 d = vec3(cos(phi) * sin(th), cos(th), sin(phi) * sin(th));",
    "  vec3 col;",
    "  if (d.y >= 0.0) {",
    "    col = atmosphere(d);",
    "    vec4 c = clouds(uCam, d);",
    "    col = col * (1.0 - c.a) + c.rgb;",
    "  } else {",
    "    col = atmosphere(normalize(vec3(d.x, 0.0, d.z))) * 0.55;",
    "  }",
    "  o = vec4(col, 1.0);",
    "}",
  ].join("\n");

  // Caustics: refract a grid of sun rays through the two short cascades onto a
  // plane at uCausDepth; each triangle's brightness is its area ratio.
  var VS_CAUSTIC = HEAD + C_COMMON + "\n" + [
    "in vec2 aPos;",
    "uniform sampler2D uDisp1, uDisp2, uDer1, uDer2;",
    "uniform vec3 uCascL;",
    "uniform float uCausL, uMargin, uCausDepth, uLod1, uLod2;",
    "uniform vec2 uShift0;",
    "out vec2 vOld, vNew;",
    "void main() {",
    "  vec2 src = aPos * (uCausL + 2.0 * uMargin) - uMargin;",
    "  vec4 d = textureLod(uDisp1, src / uCascL.y, uLod1) + textureLod(uDisp2, src / uCascL.z, uLod2);",
    "  vec4 g = textureLod(uDer1, src / uCascL.y, uLod1) + textureLod(uDer2, src / uCascL.z, uLod2);",
    "  vec3 P = vec3(src.x + d.x, d.y, src.y + d.z);",
    "  vec3 n = normalize(vec3(-g.x / max(1.0 + g.z, 0.2), 1.0, -g.y / max(1.0 + g.w, 0.2)));",
    "  vec3 r = refract(-uSunDir, n, 1.0 / 1.333);",
    "  float t = (P.y + uCausDepth) / max(-r.y, 0.05);",
    "  vNew = P.xz + r.xz * t - uShift0;",
    "  vOld = P.xz - uSunDir.xz / max(uSunDir.y, 0.05) * P.y;",
    "  gl_Position = vec4(vNew / uCausL * 2.0 - 1.0, 0.0, 1.0);",
    "}",
  ].join("\n");
  var FS_CAUSTIC = HEAD + [
    "in vec2 vOld, vNew;",
    "out vec4 o;",
    "float area(vec2 a, vec2 b) { return abs(a.x * b.y - a.y * b.x); }",
    "void main() {",
    "  float r = area(dFdx(vOld), dFdy(vOld)) / max(area(dFdx(vNew), dFdy(vNew)), 1e-14);",
    "  o = vec4(min(r, 40.0), 0.0, 0.0, 1.0);",
    "}",
  ].join("\n");

  // Background: sky (camera above water) or the underwater view of the floor.
  var FS_BACK = HEAD + C_COMMON + "\n" + C_NOISE + "\n" + C_ATMOS + "\n" + C_CLOUDS + "\n" + C_ENV + "\n" + C_WORLD + "\n" + C_SHADOW + "\n" + C_WATER + "\n" + C_FRESNEL + "\n" + [
    "uniform sampler2D uProbe;",
    "uniform mat4 uInvViewProj;",
    "uniform vec3 uSunDisk;",       // sun disk radiance
    "float hash13(vec3 p) { p = fract(p * 0.1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y) * p.z); }",
    // A star in about one cell in 70 of a fine grid over the sky, a pixel or
    // so wide, twinkling; the Milky Way as a band of faint mottled light.
    "vec3 starField(vec3 d, float px) {",
    "  vec3 p = d * 260.0, c = floor(p);",
    "  float h = hash13(c);",
    "  vec3 s = vec3(0.0);",
    "  if (h > 0.986) {",
    "    vec3 sp = c + 0.5 + 0.5 * (vec3(hash13(c + 7.1), hash13(c + 3.7), hash13(c + 1.3)) - 0.5);",
    "    vec3 e = p - sp; e -= d * dot(e, d);",
    "    float b = 0.05 + 2.5 * pow(fract(h * 43.7), 5.0);",
    "    float tw = 0.65 + 0.35 * sin(uTime * (2.0 + 6.0 * fract(h * 13.1)) + h * 80.0);",
    "    vec3 tint = mix(vec3(1.0, 0.8, 0.62), vec3(0.72, 0.84, 1.0), fract(h * 7.3));",
    "    s = tint * b * tw * exp(-dot(e, e) / (px * px)) * 0.12;",
    "  }",
    "  vec3 mw = normalize(vec3(0.35, 0.5, -0.79));",
    "  vec2 q = d.xz / (1.0 + d.y) * 7.0;",
    "  float band = exp(-pow(dot(d, mw) / 0.2, 2.0)) * (0.35 + 0.65 * smoothstep(0.3, 0.8, vnoise(q) * 0.6 + vnoise(q * 2.7) * 0.4));",
    "  s += vec3(0.72, 0.78, 1.0) * band * 0.004;",
    "  return s * smoothstep(0.0, 0.2, d.y);",
    "}",
    "in vec2 vUv;",
    "out vec4 o;",
    "void main() {",
    "  vec4 q = uInvViewProj * vec4(vUv * 2.0 - 1.0, 1.0, 1.0);",
    "  vec3 rd = normalize(q.xyz / q.w - uCam);",
    "  float starPx = max(length(fwidth(rd)) * 260.0 * 0.6, 0.05);",   // a star stays about a pixel wide
    "  bool under = uCam.y < texelFetch(uProbe, ivec2(0), 0).x;",
    "  vec3 col;",
    "  if (!under) {",
    "    if (rd.y >= 0.0) {",
    "      col = atmosphere(rd);",
    "      float mu = dot(rd, uSunDir);",
    "      float disk = smoothstep(0.999956, 0.999976, mu);",
    "      if (uNight > 0.0) {",                       // the moon: grey maria on a pale disk
    "        vec3 t1 = normalize(cross(uSunDir, vec3(0.0, 1.0, 0.0)) + 1e-5), t2 = cross(t1, uSunDir);",
    "        vec2 mq = vec2(dot(rd, t1), dot(rd, t2)) / 0.0082;",
    "        float mare = smoothstep(0.45, 0.7, vnoise(mq * 1.7 + 3.0) * 0.65 + vnoise(mq * 4.3) * 0.35);",
    "        disk *= mix(1.0, (1.0 - 0.45 * mare) * (0.8 + 0.2 * sqrt(sat(1.0 - dot(mq, mq)))), uNight);",
    "        col += starField(rd, starPx) * smoothstep(0.55, 1.0, uNight);",
    "      }",
    "      col += uSunDisk * disk;",
    "      vec4 c = clouds(uCam, rd);",
    "      col = col * (1.0 - c.a) + c.rgb;",
    "    } else {",
    // past the far edge of the water mesh: distant, mirror-like sea
    "      vec3 r = vec3(rd.x, -rd.y, rd.z);",
    "      float F = fresnel(-rd.y, 1.0 / 1.333);",
    "      col = mix(inscatter(10.0), envLod(r, 2.0), F);",
    "    }",
    "  } else {",
    "    col = inscatter(0.5 * max(-uCam.y, 0.0) + 1.0);",
    "  }",
    "  if (any(isnan(col))) col = vec3(0.0);",
    "  o = vec4(min(col, vec3(3e4)), 0.0);",
    "}",
  ].join("\n");

  // Surface mesh: a camera-centred grid whose cells grow exponentially with
  // distance. Each vertex reads displacement at the mip level matching its
  // cell size, so waves too short for the mesh fade out instead of aliasing.
  var VS_WATER = HEAD + C_COMMON + "\n" + C_OCEAN + "\n" + C_WORLD + "\n" + C_SHORE + "\n" + C_RIPPLE + "\n" + C_SWE + "\n" + [
    "in vec2 aPos;",
    "in float aCell;",
    "uniform mat4 uViewProj;",
    "uniform float uN;",
    "out vec2 vXZ;",
    "out vec3 vPos;",
    "void main() {",
    "  vec3 lod = log2(max(aCell * uN / uCascL, vec3(1.0)));",
    "  float wl = clamp(log2(aCell / 2.0), 0.0, 8.0);",
    "  float depth = -worldAt(aPos, wl).x;",
    "  vec4 d = oceanDispF(aPos, lod, depthFade(depth));",
    "  vec3 sd = shoreWave(aPos, wl).d;",
    // inside the surf simulation its water replaces the analytic swell
    "  float sm = sweMask(aPos);",
    "  if (sm > 0.0) sd = mix(sd, vec3(0.0, sweEta(sweAt(aPos).r, worldAt(aPos, 0.0).x), 0.0), sm);",
    "  d.xyz += sd;",
    "  d.y += ripple(aPos).r * (1.0 - smoothstep(0.3, 0.8, aCell));",
    "  vec3 P = vec3(aPos.x + d.x, d.y, aPos.y + d.z);",
    "  vXZ = aPos;",
    "  vPos = P;",
    "  gl_Position = uViewProj * vec4(P, 1.0);",
    "}",
  ].join("\n");

  var FS_WATER = HEAD + C_COMMON + "\n" + C_OCEAN + "\n" + C_NOISE + "\n" + C_ATMOS + "\n" + C_ENV + "\n" + C_WORLD + "\n" + C_SHORE + "\n" + C_RIPPLE + "\n" + C_SWE + "\n" + C_SHADOW + "\n" + C_WATER + "\n" + C_FRESNEL + "\n" + [
    "uniform float uFoamStr;",
    "uniform sampler2D uSceneCopy;",   // opaque scene; alpha = distance of objects (0 = terrain/sky)
    "uniform vec2 uInvRes;",
    "in vec2 vXZ;",
    "in vec3 vPos;",
    "out vec4 o;",
    "float ggx(float NoH, float a) { float a2 = a * a; float d = NoH * NoH * (a2 - 1.0) + 1.0; return a2 / (PI * d * d); }",
    "float smithVis(float NoV, float NoL, float a) {",
    "  float a2 = a * a;",
    "  return 0.5 / (NoL * sqrt(NoV * NoV * (1.0 - a2) + a2) + NoV * sqrt(NoL * NoL * (1.0 - a2) + a2) + 1e-5);",
    "}",
    "void main() {",
    "  vec2 uv0 = (vXZ + uOff0) / uCascL.x, uv1 = (vXZ + uOff1) / uCascL.y, uv2 = (vXZ + uOff2) / uCascL.z;",
    "  vec4 wd = worldAt(vXZ, 0.0);",
    "  vec3 fd = depthFade(-wd.x);",
    "  vec4 g = texture(uDer0, uv0) * fd.x + texture(uDer1, uv1) * fd.y + texture(uDer2, uv2) * fd.z;",
    "  Surf sw = shoreWave(vXZ, 0.0);",
    // inside the surf simulation: its slopes, foam and (for the milky surf
    // zone) its fronts stand in for the analytic swell's
    "  float sm = sweMask(vXZ), sweFoam = 0.0;",
    "  if (sm > 0.0) {",
    "    float ec = 0.4, b0 = wd.x;",
    "    float eX = sweEta(sweAt(vXZ + vec2(ec, 0.0)).r, worldAt(vXZ + vec2(ec, 0.0), 0.0).x) - sweEta(sweAt(vXZ - vec2(ec, 0.0)).r, worldAt(vXZ - vec2(ec, 0.0), 0.0).x);",
    "    float eZ = sweEta(sweAt(vXZ + vec2(0.0, ec)).r, worldAt(vXZ + vec2(0.0, ec), 0.0).x) - sweEta(sweAt(vXZ - vec2(0.0, ec)).r, worldAt(vXZ - vec2(0.0, ec), 0.0).x);",
    "    vec4 st = sweAt(vXZ);",
    "    sweFoam = st.a;",
    "    sw.slope = mix(sw.slope, vec2(eX, eZ) / (2.0 * ec), sm);",
    "    sw.jac *= 1.0 - sm;",
    "    sw.foam = mix(sw.foam, sat(sweFoam * 0.8), sm);",
    "    float swH = max(st.r + b0, 0.0) * 1.5;",                       // crest height drives the milky glow
    "    sw.H = mix(sw.H, swH, sm);",
    "    sw.s = mix(sw.s, 0.5 + sat(sweFoam), sm);",
    "  }",
    // surface normal from both tangent vectors (handles the folded lip too)
    "  vec2 rs = rippleSlope(vXZ);",
    "  vec4 rp = ripple(vXZ);",
    "  float ripFoam = rp.b;",
    "  vec3 Tx = vec3(1.0 + g.z + sw.jac.x, g.x + sw.slope.x + rs.x, sw.jac.w);",
    "  vec3 Tz = vec3(sw.jac.z, g.y + sw.slope.y + rs.y, 1.0 + g.w + sw.jac.y);",
    "  vec3 N = normalize(cross(Tz, Tx));",
    "  if (N.y < 0.05) N = normalize(vec3(N.x, 0.05, N.z));",
    "  float foam = texture(uDisp0, uv0).w * fd.x + texture(uDisp1, uv1).w * fd.y + 0.5 * texture(uDisp2, uv2).w * fd.z;",
    "  vec3 P = vPos;",
    "  vec3 toEye = uCam - P;",
    "  float dist = length(toEye);",
    "  vec3 V = toEye / dist;",
    // Filtered normals lose slope variance with distance; give it back as roughness.
    "  float rough = clamp(0.045 + 0.05 * log2(1.0 + dist / 25.0), 0.045, 0.32);",
    "  float envLodR = clamp(log2(1.0 + dist / 60.0) * 1.3, 0.0, 5.0);",
    // seen from below, the surface shows its back face
    "  bool under = !gl_FrontFacing;",
    "  vec3 col;",
    "  if (!under) {",
    "    if (dot(N, V) < 0.02) N = normalize(N + V * (0.02 - dot(N, V)));",
    "    float NoV = dot(N, V);",
    "    float F = fresnel(NoV, 1.0 / 1.333);",
    "    vec3 R = reflect(-V, N); R.y = abs(R.y);",
    "    vec3 refl = envLod(R, envLodR);",
    // far away, a pixel covers many wavelets; bend the view ray less there
    "    vec3 Nr = normalize(mix(N, vec3(0.0, 1.0, 0.0), 0.6 * sat(dist / 70.0)));",
    "    vec3 T = refract(-V, Nr, 1.0 / 1.333);",
    "    float t = traceGround(P, T);",
    "    vec3 hit = P + T * t;",
    "    float shadow = terrainShadow(P, 0.0) * sunVis(P, 0.002);",
    // a raised crest is a thicker body of water to look through
    "    float body = max(P.y + 0.25 * sw.H, 0.0) * sat(1.2 - N.y) * 3.0 * sat(sw.H * 2.5);",
    "    vec3 refr = throughWater(shadeFloor(hit, dist + t, shadow), t + body, 0.5 * max(-hit.y, 0.0));",
    // the surf zone is milky with stirred-up sand: it catches sun and sky
    // light diffusely, which is what shows the shape of a breaking face
    "    float turbid = smoothstep(0.45, 1.15, sw.s) * sat(sw.H * 2.5) * 0.6;",
    "    vec3 milk = vec3(0.32, 0.6, 0.56) * (uSunCol * max(dot(N, uSunDir), 0.0) * shadow * 0.85 + skyIrradiance() * (0.4 + 0.3 * N.y)) / PI;",
    "    refr = mix(refr, milk, turbid);",
    // corals, fish, floating things: whatever was drawn behind this pixel,
    // looked at through the surface (offset by the normal like a refraction)
    "    vec4 so = texture(uSceneCopy, gl_FragCoord.xy * uInvRes + N.xz * (0.05 / (1.0 + dist * 0.04)));",
    "    if (so.a > dist + 0.03 && so.a - dist < t * 1.3 + 1.0) refr = mix(throughWater(so.rgb, so.a - dist, 0.5 * max(-hit.y, 0.0)), milk, turbid);",
    "    col = mix(refr, refl, F);",
    // sun glint (GGX)
    "    vec3 H = normalize(V + uSunDir);",
    "    float NoL = max(dot(N, uSunDir), 0.0), NoH = max(dot(N, H), 0.0);",
    "    float a = rough * rough;",
    "    col += uSunCol * shadow * (ggx(NoH, a) * smithVis(NoV, NoL, a) * fresnel(max(dot(V, H), 0.0), 1.0 / 1.333) * NoL);",
    // light through thin wave crests
    "    float sss = pow(sat(dot(V, -uSunDir) * 0.6 + 0.4), 3.0) * sat(P.y * 1.5) * sat(1.0 - N.y + 0.15);",
    "    col += inscatter(0.5) * sss * 1.4;",
    // sunlight passing through a raised crest lights it turquoise from inside
    "    float glow = sat(P.y * 0.9) * sat(1.15 - N.y) * sat(sw.H * 2.0);",
    "    col += vec3(0.06, 0.42, 0.40) * (uSunCol.g * 0.05) * glow * (0.6 + 0.8 * pow(sat(dot(V, -uSunDir) * 0.5 + 0.5), 2.0));",
    // whitecap foam, broken up by noise; surf whitewater is lacy where thin
    // and solid where the bore churns; a lacy edge rims the swash film
    "    vec2 fw = vXZ + uOffD - sw.n * (uTime * 0.6);",
    "    float lace = vnoise(fw * 0.9) * 0.55 + vnoise(fw * 3.1) * 0.45;",
    "    float film = P.y - groundH(P.xz);",
    "    float edge = smoothstep(0.0, 0.03, film) * (1.0 - smoothstep(0.04, 0.16, film)) * step(1.5, sw.s);",
    "    float surf = sat(sw.foam * 1.8 - 0.3 + (lace - 0.5) * 1.2) + edge * smoothstep(0.45, 0.65, lace) * 0.8;",
    "    surf = max(surf, sat(ripFoam * 1.4 - 0.35 + (lace - 0.5) * 2.2) * sat(ripFoam * 4.0));",   // wakes and splashes: lacy, solid only where churned
    "    float m = max(smoothstep(0.25, 0.95, foam * (0.5 + 0.9 * vnoise((vXZ + uOffD) * 2.0)) * uFoamStr), sat(surf));",
    "    vec3 foamCol = 0.85 * (uSunCol * max(dot(N, uSunDir), 0.0) * shadow + skyIrradiance()) / PI;",
    "    col = mix(col, foamCol, m * (1.0 - 0.45 * uNight));",
    // at night the plankton light up wherever the water is stirred: breaking
    // surf and whitewater, wakes, splashes, ripples spreading from anything moving
    "    if (uGlowCol.b > 0.0) {",
    "      vec2 gq = (vXZ + uOffD) * 5.0;",
    "      float spark = 0.35 + 0.9 * smoothstep(0.45, 0.95, vnoise(gq + vec2(uTime * 1.3, 0.0)) * 0.6 + vnoise(gq * 2.3 - vec2(0.0, uTime * 1.7)) * 0.4);",
    "      float stir = rp.a * 0.9 + surf * 1.4 + m * 0.5 + sat(ripFoam) * 0.8 + sat(sweFoam) * 0.6;",
    "      col += uGlowCol * stir * spark;",
    "    }",
    "  } else {",
    // seen from below: Snell's window to the sky, total internal reflection outside it
    "    vec3 Nw = -normalize(mix(N, vec3(0.0, 1.0, 0.0), 0.7 * sat(dist / 30.0)));",
    "    float cosI = max(dot(V, Nw), 1e-3);",
    "    float F = fresnel(cosI, 1.333);",
    "    vec3 R = reflect(-V, Nw);",
    "    float tr = traceGroundMarch(P, R, 120.0);",
    "    float zFog = 0.5 * max(-uCam.y, 0.0) + 1.0;",
    "    vec3 below = tr > 0.0 ? throughWater(shadeFloor(P + R * tr, dist + tr, 1.0), tr, zFog) : inscatter(zFog);",
    "    vec3 above = vec3(0.0);",
    "    if (F < 1.0) {",
    "      vec3 T = refract(-V, Nw, 1.333);",
    "      above = envLod(T, envLodR) + uSunCol * 12.0 * pow(max(dot(T, uSunDir), 0.0), 900.0);",
    "    }",
    "    col = mix(above, below, F);",
    "    col = mix(col, foam * vec3(0.7) * skyIrradiance() / PI, sat(foam * uFoamStr * 0.6));",
    "    if (uGlowCol.b > 0.0) col += uGlowCol * (rp.a * 0.9 + sat(sw.foam * 1.6) + sat(ripFoam) * 0.8 + sat(foam * 0.5)) * (0.5 + 0.7 * vnoise((vXZ + uOffD) * 5.0 + uTime));",
    "    col = throughWater(col, dist, zFog);",
    "  }",
    "  if (any(isnan(col))) col = vec3(0.0);",
    "  o = vec4(min(col, vec3(3e4)), 1.0);",   // stay inside half-float range
    "}",
  ].join("\n");

  // Terrain: the same camera-centred grid, lifted to the island heightfield at
  // the mip level matching each cell.
  var VS_TERRAIN = HEAD + C_COMMON + "\n" + C_NOISE + "\n" + C_WORLD + "\n" + C_CORALROCK + "\n" + [
    "in vec2 aPos;",
    "in float aCell;",
    "uniform mat4 uViewProj;",
    "out vec3 vPos;",
    "void main() {",
    "  vec4 wd = worldAt(aPos, clamp(log2(aCell / 2.0), 0.0, 8.0));",
    "  float h = wd.x;",
    "  h += coralLump(aPos, wd.y, distance(aPos, uCam.xz)) * (1.0 - smoothstep(-0.8, -0.2, h));",   // fades in with depth: no step at the edge
    "  vPos = vec3(aPos.x, h, aPos.y);",
    "  gl_Position = uViewProj * vec4(vPos, 1.0);",
    "}",
  ].join("\n");
  var FS_TERRAIN = HEAD + C_COMMON + "\n" + C_NOISE + "\n" + C_ATMOS + "\n" + C_ENV + "\n" + C_WORLD + "\n" + C_SHORE + "\n" + C_SHADOW + "\n" + C_WATER + "\n" + C_CORALROCK + "\n" + C_SWE + "\n" + [
    "uniform sampler2D uProbe, uSweAux;",
    "in vec3 vPos;",
    "out vec4 o;",
    "void main() {",
    "  vec3 P = vPos;",
    "  vec3 toEye = uCam - P;",
    "  float dist = length(toEye);",
    "  vec3 V = toEye / dist;",
    "  bool under = uCam.y < texelFetch(uProbe, ivec2(0), 0).x;",
    // seen from above, anything a metre under water is covered by the surface
    "  if (!under && P.y < -1.0) { o = vec4(0.0); return; }",   // alpha 0: not an object for the water to look through
    "  float lod = clamp(log2(1.0 + dist / 60.0), 0.0, 6.0);",
    "  vec4 wd = worldAt(P.xz, 0.0);",
    "  vec2 w = P.xz + uOffD;",
    "  vec3 n = groundNormal(P.xz, lod);",
    "  float lump = 0.0;",
    "  if (wd.x < -0.2 && wd.y > 0.05) {",          // coral rock relief, shaded from the same function
    "    float dr = distance(P.xz, uCam.xz), e = 0.12 + dr * 0.01;",
    "    float dw = (1.0 - smoothstep(-0.8, -0.2, wd.x));",
    "    lump = coralLump(P.xz, wd.y, dr) * dw;",
    "    float lx = (coralLump(P.xz + vec2(e, 0.0), wd.y, dr) - coralLump(P.xz - vec2(e, 0.0), wd.y, dr)) * dw;",
    "    float lz = (coralLump(P.xz + vec2(0.0, e), wd.y, dr) - coralLump(P.xz - vec2(0.0, e), wd.y, dr)) * dw;",
    "    n = normalize(n + vec3(-lx, 0.0, -lz) / (2.0 * e));",
    "  }",
    "  float wet;",
    "  vec3 alb = groundAlbedo(w, wd, n, wet);",
    "  alb *= 1.0 + clamp(lump * 1.2, -0.45, 0.2);",   // crevices darker, knobs lighter
    // sand the swash just ran over stays dark and glossy for a few seconds
    "  Surf sw = shoreWave(P.xz, 0.0);",
    "  float smT = sweMask(P.xz);",
    "  if (smT > 0.0 && wd.x > -0.3) {",           // the simulated swash wets the sand it reached
    "    float wet2 = texBilin(uSweAux, sweUV(P.xz) * vec2(textureSize(uSweAux, 0)) - 0.5).a * smT;",
    "    alb *= mix(1.0, 0.62, max(wet2 - wet, 0.0));",
    "    wet = max(wet, wet2);",
    "  } else if (wd.x > 0.0 && sw.H > 0.0) {",
    "    float zone = (1.0 - smoothstep(-0.04, 0.12, wd.x - 0.55 * sw.H));",
    "    float fresh = sw.th > 0.0 ? exp(-sw.th * 0.6) : exp(sw.th * 2.0);",
    "    float wet2 = zone * (0.6 + 0.4 * fresh);",
    "    alb *= mix(1.0, 0.62, max(wet2 - wet, 0.0));",
    "    wet = max(wet, wet2);",
    "  }",
    "  n = sandNormal(w, n, dist, (1.0 - smoothstep(1.5, 4.0, wd.x)) * 0.6);",
    "  float shadow = terrainShadow(P, lod) * sunVis(P + n * 0.05, 0.0015);",
    "  float z = max(-P.y, 0.0);",
    "  vec3 col = lightGround(P, n, alb, z, dist, shadow);",
    // wet sand shines in the sun
    "  vec3 H = normalize(V + uSunDir);",
    "  col += uSunCol * shadow * wet * 0.04 * pow(max(dot(n, H), 0.0), 120.0) * sat(P.y + 0.2);",
    "  if (under) col = throughWater(col, dist, 0.5 * (max(-uCam.y, 0.0) + z));",
    "  else col = aerial(col, -V, dist);",
    "  if (any(isnan(col))) col = vec3(0.0);",
    "  o = vec4(min(col, vec3(3e4)), 0.0);",
    "}",
  ].join("\n");

  // Plants: instanced palms (kind 0 trunk, 1 frond, 2 coconut) and canopy
  // trees (kind 3). Per instance: position (O-relative) and size; rotation,
  // lean (palms) or squash (trees), phase, tint; island-shadow visibility.
  var VS_PLANT = HEAD + C_COMMON + "\n" + [
    "in vec3 aPos;",
    "in vec3 aNrm;",
    "in vec2 aUV;",
    "in float aKind;",
    "in vec4 aInst0;",
    "in vec4 aInst1;",
    "in float aInst2;",
    "uniform mat4 uViewProj;",
    "uniform vec2 uWindXZ;",
    "uniform float uWindStr;",
    "out vec3 vPos, vN;",
    "out vec2 vUV;",
    "out float vKind, vTint, vVis, vH;",
    "void main() {",
    "  vec3 p = aPos, n = aNrm;",
    "  float sz = aInst0.w;",
    "  if (aKind < 2.5) {",
    "    p.x += (aInst1.y - 1.0) * 0.16 * p.y * p.y;",   // lean: shear the trunk's curve
    "    if (aKind > 0.5 && aKind < 1.5) p.y += sin(uTime * 3.1 + aInst1.z * 5.0 + aUV.y * 3.0) * 0.012 * aUV.y * uWindStr;",
    "  } else {",
    "    p.y = p.y * aInst1.y + 0.55;",
    "    n = normalize(vec3(n.x, n.y / aInst1.y, n.z));",
    "  }",
    "  float c = cos(aInst1.x), sn = sin(aInst1.x);",
    "  p = vec3(c * p.x - sn * p.z, p.y, sn * p.x + c * p.z);",
    "  n = vec3(c * n.x - sn * n.z, n.y, sn * n.x + c * n.z);",
    "  float hy = p.y;",
    "  p *= sz;",
    "  p.xz += uWindXZ * (0.6 + 0.4 * sin(uTime * 1.2 + aInst1.z)) * uWindStr * 0.035 * hy * hy * sz;",
    "  vPos = aInst0.xyz + p; vN = n; vUV = aUV; vKind = aKind; vTint = aInst1.w; vVis = aInst2; vH = hy;",
    "  gl_Position = uViewProj * vec4(vPos, 1.0);",
    "}",
  ].join("\n");
  // Leaflets: diagonal slats off the rachis, ragged at the tips; they merge
  // into a solid blade once smaller than a pixel so far fronds don't sparkle.
  var C_LEAF = [
    "bool leafGap(vec2 uv) {",
    "  float x = abs(uv.x), y = uv.y, g = y * 46.0 + x * 5.0;",
    "  float keep = mix(0.58, 1.01, sat(fwidth(g) * 1.2 - 0.2));",
    "  float edge = 1.0 - 0.2 * fract(g * 0.5) - 0.3 * smoothstep(0.7, 1.0, y);",
    "  return x > 0.08 && (fract(g) > keep || x > edge);",
    "}",
  ].join("\n");
  var FS_PLANT = HEAD + C_COMMON + "\n" + C_NOISE + "\n" + C_ATMOS + "\n" + C_ENV + "\n" + C_WORLD + "\n" + C_SHADOW + "\n" + C_LEAF + "\n" + [
    "uniform vec2 uOffD;",
    "in vec3 vPos, vN;",
    "in vec2 vUV;",
    "in float vKind, vTint, vVis, vH;",
    "out vec4 o;",
    "vec3 aerialP(vec3 col, vec3 dir, float dist) {",
    "  float f = 1.0 - exp(-dist / 9000.0);",
    "  return mix(col, atmosphere(normalize(vec3(dir.x, max(dir.y, 0.02), dir.z))), f);",
    "}",
    "void main() {",
    "  bool frond = vKind > 0.5 && vKind < 1.5, gap = leafGap(vUV);",   // derivatives in uniform flow
    "  if (frond && gap) discard;",
    "  vec3 n = normalize(vN);",
    "  if (frond && dot(n, uCam - vPos) < 0.0) n = -n;",   // fronds are two-sided; solids keep their outward normals
    "  vec3 P = vPos, toEye = uCam - P;",
    "  float dist = length(toEye);",
    "  vec3 V = toEye / dist;",
    "  vec2 w = P.xz + uOffD;",
    "  vec3 alb; float trans = 0.0, ao = 1.0;",
    "  if (vKind < 0.5) {",                         // trunk: grey-brown with leaf-scar rings
    "    alb = vec3(0.42, 0.38, 0.31) * (0.8 + 0.2 * sin(vUV.y * TAU)) * (0.85 + 0.3 * vTint);",
    "  } else if (frond) {",
    "    alb = mix(vec3(0.07, 0.17, 0.035), vec3(0.16, 0.24, 0.05), vTint);",
    "    alb = mix(alb, vec3(0.34, 0.31, 0.11), smoothstep(0.55, 1.0, vUV.y) * (0.3 + 0.5 * vTint));",
    "    alb *= 0.85 + 0.3 * abs(vUV.x);",
    "    trans = 0.55;",
    "  } else if (vKind < 2.5) {",
    "    alb = vec3(0.24, 0.25, 0.07);",
    "  } else {",                                   // canopy
    "    alb = mix(vec3(0.035, 0.085, 0.025), vec3(0.10, 0.16, 0.045), vTint);",
    "    if (vTint > 0.996) alb = vec3(0.30, 0.11, 0.05);",   // the odd flame tree
    "    float leaves = vnoise(w * 0.9 + vPos.y * 0.7) * 0.6 + vnoise(w * 3.7) * 0.4;",
    "    alb *= 0.7 + 0.6 * leaves;",
    "    n = normalize(n + 0.35 * vec3(leaves - 0.5, 0.0, vnoise(w * 1.3) - 0.5));",
    "    ao = mix(0.45, 1.0, sat(vH * 0.8 + 0.2));",
    "    trans = 0.3;",
    "  }",
    "  float vis = vVis * sunVis(P + n * 0.08, 0.002);",
    "  vec3 sky = PI * textureLod(uEnv, vec2(0.5, 0.25), 7.0).rgb;",
    "  float ndl = dot(n, uSunDir);",
    // sunlight bounced up from the ground (white sand near the beach)
    "  float ground = mix(0.12, 0.55, 1.0 - smoothstep(2.0, 8.0, P.y));",
    "  vec3 bounce = uSunCol * max(uSunDir.y, 0.0) * ground * (0.5 - 0.5 * n.y);",
    "  vec3 col = alb * (uSunCol * (max(ndl, 0.0) + trans * max(-ndl, 0.0) * 0.7) * vis + sky * (0.45 + 0.35 * n.y) * ao + bounce * ao) / PI;",
    // sunlight through the leaves when you look toward the sun
    "  col += alb * uSunCol * trans * 0.25 * pow(sat(dot(-V, uSunDir)), 4.0) * vis / PI * 3.0;",
    "  col = aerialP(col, -V, dist);",
    "  if (any(isnan(col))) col = vec3(0.0);",
    "  o = vec4(min(col, vec3(3e4)), 0.0);",
    "}",
  ].join("\n");
  var FS_PLANT_SHADOW = HEAD + C_COMMON + "\n" + C_LEAF + "\n" + [
    "in vec3 vPos, vN;",
    "in vec2 vUV;",
    "in float vKind, vTint, vVis, vH;",
    "out vec4 o;",
    "void main() {",
    "  bool gap = leafGap(vUV);",
    "  if (vKind > 0.5 && vKind < 1.5 && gap) discard;",
    "  o = vec4(1.0);",
    "}",
  ].join("\n");

  // Reef life and other objects: instanced meshes with a per-instance
  // position + scale, rotation quaternion, and four parameters (seed, ...).
  var VS_THING = HEAD + C_COMMON + "\n" + [
    "in vec3 aPos;",
    "in vec3 aNrm;",
    "in vec2 aUV;",
    "in float aKind;",
    "in vec4 aInst0;",
    "in vec4 aInst1;",
    "in vec4 aInst2;",
    "uniform mat4 uViewProj;",
    "uniform float uObjKind, uGrowR;",
    "out vec3 vPos, vN, vLocal;",
    "out vec2 vUV;",
    "out float vPart;",
    "out vec4 vParams;",
    "vec3 qrot(vec4 q, vec3 v) { return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v); }",
    "void main() {",
    "  vec3 p = aPos, n = aNrm;",
    "  float k = uObjKind, seed = aInst2.x;",
    "  if (k > 2.5 && k < 4.5) p.xz += vec2(0.06, 0.03) * sin(uTime * 0.9 + seed * 6.0) * p.y * p.y;",   // surge sways fans and soft corals
    "  if (k > 9.5 && k < 10.5) {",                    // fish: a wave travels down the body, growing to the tail
    "    float along = sat((0.45 - p.x) / 1.15);",
    "    p.z += sin(uTime * aInst2.z + aInst2.w - p.x * 7.0) * 0.1 * along * along;",
    "  } else if (k > 10.5 && k < 11.5) {",           // turtle: front flippers beat, rear ones steer
    "    float ph = uTime * 1.3 + aInst2.w;",
    "    if (aKind > 1.5 && aKind < 3.5) { float r = max(abs(p.z) - 0.28, 0.0); p.y += sin(ph) * 0.32 * r; p.x += cos(ph) * 0.18 * r; }",
    "    else if (aKind > 3.5) { float r = max(length(p.xz - vec2(-0.32, 0.0)) - 0.2, 0.0); p.y += sin(ph * 0.7 + 1.0) * 0.25 * r; }",
    "  } else if (k > 11.5 && k < 12.5) {",           // eagle ray: wing beats ripple out to the tips
    "    float az = abs(p.z);",
    "    if (aKind < 0.5) p.y += sin(uTime * 1.1 + aInst2.w - az * 1.4) * 0.28 * pow(az, 1.6);",
    "    else p.z += sin(uTime * 2.0 + p.x * 3.0) * 0.06 * (-p.x);",
    "  }",
    "  float s = aInst0.w;",
    "  if (uGrowR > 0.0) s *= 1.0 - smoothstep(uGrowR * 0.72, uGrowR, distance(aInst0.xz, uCam.xz));",   // grow in as you approach
    "  vPos = aInst0.xyz + qrot(aInst1, p * s);",
    "  vN = qrot(aInst1, n); vLocal = p; vUV = aUV; vPart = aKind; vParams = aInst2;",
    "  gl_Position = uViewProj * vec4(vPos, 1.0);",
    "}",
  ].join("\n");
  var FS_THING = HEAD + C_COMMON + "\n" + C_NOISE + "\n" + C_ATMOS + "\n" + C_ENV + "\n" + C_WORLD + "\n" + C_SHADOW + "\n" + C_WATER + "\n" + [
    "uniform sampler2D uProbe;",
    "uniform float uObjKind;",
    "in vec3 vPos, vN, vLocal;",
    "in vec2 vUV;",
    "in float vPart;",
    "in vec4 vParams;",
    "out vec4 o;",
    "vec3 pick3(float t, vec3 a, vec3 b, vec3 c) { return t < 0.33 ? a : t < 0.66 ? b : c; }",
    "void main() {",
    "  int k = int(uObjKind + 0.5);",
    "  float seed = vParams.x;",
    "  vec3 n = normalize(vN);",
    "  vec3 P = vPos, toEye = uCam - P;",
    "  if ((k == 3 || k == 4) && dot(n, toEye) < 0.0) n = -n;",   // fans and open tubes are seen from both sides
    "  float dist = length(toEye);",
    "  vec3 lp = vLocal * 3.0 + seed * 17.0;",
    "  vec3 alb;",
    "  if (k == 0) {",                                 // brain coral: meandering ridges
    "    alb = pick3(seed, vec3(0.50, 0.42, 0.20), vec3(0.34, 0.44, 0.24), vec3(0.52, 0.36, 0.30));",
    "    float m = sin((vnoise(lp.xz * 1.3) * 9.0 + vnoise(lp.xz * 2.7 + lp.y) * 4.0) * 3.1);",
    "    alb *= 0.7 + 0.35 * m;",
    "    n = normalize(n + 0.25 * vec3(cos(m * 3.0), 0.0, sin(m * 3.0)) * (1.0 - smoothstep(4.0, 15.0, dist)));",
    "  } else if (k == 1) {",                          // staghorn: pale growing tips
    "    alb = pick3(seed, vec3(0.48, 0.34, 0.22), vec3(0.38, 0.28, 0.36), vec3(0.42, 0.40, 0.24));",
    "    alb = mix(alb, vec3(0.80, 0.74, 0.62), smoothstep(0.75, 1.0, vUV.y));",
    "    alb *= 0.85 + 0.3 * vnoise(lp.xy * 8.0);",
    "  } else if (k == 2) {",                          // table coral: polyp-speckled plate
    "    alb = pick3(seed, vec3(0.42, 0.40, 0.24), vec3(0.32, 0.38, 0.30), vec3(0.50, 0.40, 0.28));",
    "    alb *= 0.75 + 0.4 * vnoise(lp.xz * 9.0) * (vPart > 1.5 ? 0.5 : 1.0);",
    "    if (vPart > 1.5) { alb *= 0.85; n = normalize(n + vec3(0.0, 0.9, 0.0)); }",   // underside: lit by light bounced off the sand
    "  } else if (k == 3) {",                          // sea fan: an open lattice
    "    float u = vUV.x * 10.0, v = vUV.y * 13.0, fwu = fwidth(u);",
    "    float gx = abs(fract(u + v * 0.35 + 0.15 * sin(v * 2.0)) - 0.5), gy = abs(fract(v + u * 0.2) - 0.5);",
    "    if (min(gx, gy) > 0.14 && abs(vUV.x) > 0.04 && fwu < 0.45) discard;",
    "    alb = pick3(seed, vec3(0.50, 0.10, 0.36), vec3(0.62, 0.48, 0.10), vec3(0.55, 0.16, 0.14));",
    "  } else if (k == 4) {",                          // soft coral / tube sponges
    "    alb = pick3(seed, vec3(0.78, 0.36, 0.10), vec3(0.80, 0.66, 0.16), vec3(0.66, 0.30, 0.46));",
    "    alb *= 0.8 + 0.35 * vnoise(vec2(vUV.x * 12.0, vUV.y * 20.0) + seed * 9.0);",
    "    if (vUV.y > 0.97) alb *= 0.35;",                // dark open mouths
    "  } else if (k == 5) {",                          // boulder coral: pitted dome
    "    alb = pick3(seed, vec3(0.52, 0.46, 0.26), vec3(0.42, 0.42, 0.36), vec3(0.46, 0.38, 0.30));",
    "    alb *= 0.7 + 0.45 * vnoise(lp.xz * 14.0 + lp.y * 7.0);",
    "  } else if (k == 10) {",                         // reef fish by species
    "    int sp = int(vParams.y + 0.5);",
    "    float t = vUV.x, top = cos(vUV.y * TAU), belly = (1.0 - smoothstep(-0.6, 0.2, top));",
    "    if (sp == 0) alb = vec3(0.95, 0.76, 0.06);",
    "    else if (sp == 1) alb = mix(vec3(0.05, 0.32, 0.68), vec3(0.45, 0.72, 0.85), belly);",
    "    else if (sp == 2) {",
    "      alb = mix(vec3(0.78, 0.72, 0.22), vec3(0.92, 0.92, 0.80), belly);",
    "      if (t > 0.14 && t < 0.86 && fract(t * 5.2) > 0.62 && top > -0.55) alb *= 0.12;",   // black bars
    "    } else if (sp == 3) {",
    "      alb = mix(vec3(0.10, 0.55, 0.50), vec3(0.40, 0.78, 0.66), belly);",
    "      alb = mix(alb, vec3(0.88, 0.46, 0.56), smoothstep(0.55, 0.75, vnoise(vec2(t * 6.0, vUV.y * 8.0) + seed * 20.0)));",
    "    } else if (sp == 4) {",
    "      alb = mix(vec3(0.92, 0.92, 0.88), vec3(0.96, 0.80, 0.10), smoothstep(0.45, 0.7, t));",
    "      if (t > 0.09 && t < 0.17) alb *= 0.12;",       // eye band
    "    } else alb = mix(vec3(0.22, 0.32, 0.42), vec3(0.85, 0.88, 0.90), belly);",
    "    if (vPart < 0.5 && t < 0.1 && abs(top) < 0.45 && t > 0.05) alb = vec3(0.03);",   // eye
    "    if (vPart > 0.5) { alb *= 0.85; if (dot(n, toEye) < 0.0) n = -n; }",
    "  } else if (k == 11) {",                         // green turtle
    "    if (vPart < 0.5) {",
    "      float sc = vnoise(vLocal.xz * 9.0 + seed * 30.0);",
    "      alb = mix(vec3(0.26, 0.20, 0.10), vec3(0.50, 0.40, 0.20), smoothstep(0.3, 0.7, sc));",
    "      alb *= 0.75 + 0.35 * smoothstep(0.15, 0.4, abs(fract(sc * 3.0) - 0.5));",   // dark seams between scutes
    "    } else {",
    "      alb = mix(vec3(0.40, 0.40, 0.28), vec3(0.70, 0.64, 0.44), (1.0 - smoothstep(-0.8, -0.2, n.y)));",
    "      alb *= 0.75 + 0.4 * vnoise(vLocal.xz * 30.0);",
    "      if (dot(n, toEye) < 0.0) n = -n;",
    "    }",
    "  } else if (k == 12) {",                         // spotted eagle ray: dark back, white spots, white belly
    "    bool below = dot(n, toEye) < 0.0;",
    "    if (below) { n = -n; alb = vec3(0.86, 0.86, 0.82); }",
    "    else {",
    "      alb = vec3(0.09, 0.11, 0.15);",
    "      if (vnoise(vUV * vec2(26.0, 9.0) + seed * 40.0) > 0.78) alb = vec3(0.86, 0.86, 0.80);",
    "    }",
    "    if (vPart > 0.5) alb = vec3(0.06);",
    "  } else if (k == 20) {",                         // beach ball: six panels, white caps
    "    float sl = floor(fract(atan(vLocal.z, vLocal.x) / TAU + seed) * 6.0);",
    "    alb = sl < 1.0 ? vec3(0.85, 0.08, 0.06) : sl < 2.0 ? vec3(0.92) : sl < 3.0 ? vec3(0.06, 0.25, 0.80) : sl < 4.0 ? vec3(0.95, 0.80, 0.05) : sl < 5.0 ? vec3(0.92) : vec3(0.10, 0.65, 0.22);",
    "    if (abs(vLocal.y) > 0.88) alb = vec3(0.92);",
    "  } else if (k == 21) {",                         // coconut husk
    "    alb = vec3(0.36, 0.24, 0.12) * (0.7 + 0.5 * vnoise(vec2(atan(vLocal.z, vLocal.x) * 12.0, vLocal.y * 40.0)));",
    "  } else if (k == 22) {",                         // basalt rock
    "    alb = vec3(0.16, 0.15, 0.14) * (0.7 + 0.6 * vnoise(vLocal.xz * 9.0 + vLocal.y * 5.0));",
    "  } else if (k == 23) {",                         // the boat
    "    int part = int(vPart + 0.5);",
    "    if (part == 0) {",
    "      alb = vec3(0.88, 0.88, 0.86);",
    "      if (vLocal.y < 0.12 && vLocal.y > -0.03) alb = vec3(0.04, 0.12, 0.30);",   // boot stripe
    "      else if (vLocal.y <= -0.03) alb = vec3(0.12, 0.26, 0.40);",                // bottom paint
    "    } else if (part == 2) alb = vec3(0.50, 0.33, 0.18) * (0.8 + 0.25 * step(0.5, fract(vLocal.z * 9.0)));",   // teak
    "    else if (part == 3) alb = vec3(0.90, 0.90, 0.88);",
    "    else if (part == 4) alb = vec3(0.05, 0.05, 0.06);",
    "    else alb = vec3(0.04, 0.05, 0.06);",
    "  } else alb = vec3(0.5);",
    "  float z = max(-P.y, 0.0);",
    "  float shadow = terrainShadow(P, 0.0) * sunVis(P + n * 0.05, 0.002);",
    "  vec3 col = lightGround(P, n, alb, z, dist, shadow);",
    "  bool under = uCam.y < texelFetch(uProbe, ivec2(0), 0).x;",
    // at night, swimmers wear a halo of the plankton they disturb
    "  if (k >= 10 && k <= 12 && uGlowCol.b > 0.0) col += uGlowCol * (0.2 + 1.5 * pow(1.0 - abs(dot(n, toEye)) / dist, 3.0)) * (0.5 + 0.7 * vnoise(vLocal.xz * 9.0 + uTime * 3.0));",
    "  if (under) col = throughWater(col, dist, 0.5 * (max(-uCam.y, 0.0) + z));",
    "  if (any(isnan(col))) col = vec3(0.0);",
    // alpha carries the distance so the water can look through its surface at us
    "  o = vec4(min(col, vec3(3e4)), under ? 0.0 : dist);",
    "}",
  ].join("\n");
  var FS_THING_SHADOW = HEAD + C_COMMON + "\n" + [
    "uniform float uObjKind;",
    "in vec3 vPos, vN, vLocal;",
    "in vec2 vUV;",
    "in float vPart;",
    "in vec4 vParams;",
    "out vec4 o;",
    "void main() {",
    "  if (int(uObjKind + 0.5) == 3) {",
    "    float u = vUV.x * 10.0, v = vUV.y * 13.0;",
    "    float gx = abs(fract(u + v * 0.35 + 0.15 * sin(v * 2.0)) - 0.5), gy = abs(fract(v + u * 0.2) - 0.5);",
    "    if (min(gx, gy) > 0.14 && abs(vUV.x) > 0.04) discard;",
    "  }",
    "  o = vec4(1.0);",
    "}",
  ].join("\n");

  // Spray: camera-facing soft sprites, stretched along their velocity.
  var VS_SPRAY = HEAD + C_COMMON + "\n" + [
    "in vec4 aInst0;",       // x, y, z, size
    "in vec4 aInst1;",       // vx, vy, vz, alpha
    "uniform mat4 uViewProj;",
    "uniform vec3 uCamRight, uCamUp;",
    "out vec2 vQ;",
    "out float vA;",
    "void main() {",
    "  int c = gl_VertexID % 6;",
    "  vec2 q = vec2(c == 1 || c == 2 || c == 4 ? 1.0 : -1.0, c >= 2 && c <= 4 ? 1.0 : -1.0);",
    "  vec3 v = aInst1.xyz, P = aInst0.xyz;",
    "  vec3 vs = v - dot(v, normalize(uCam - P)) * normalize(uCam - P);",   // screen-plane velocity
    "  float vl = length(vs);",
    "  vec3 along = vl > 1e-3 ? vs / vl : uCamUp, across = normalize(cross(along, normalize(uCam - P)));",
    "  float stretch = 1.0 + min(vl * 0.08, 2.5);",
    "  P += across * q.x * aInst0.w + along * q.y * aInst0.w * stretch;",
    "  vQ = q; vA = aInst1.w * smoothstep(0.4, 1.6, distance(uCam, P));",   // drops at the lens would fill the view
    "  gl_Position = uViewProj * vec4(P, 1.0);",
    "}",
  ].join("\n");
  var FS_SPRAY = HEAD + C_COMMON + "\n" + C_ENV + "\n" + [
    "in vec2 vQ;",
    "in float vA;",
    "out vec4 o;",
    "void main() {",
    "  float d = length(vQ);",
    "  float a = (1.0 - smoothstep(0.25, 1.0, d)) * vA * 0.7;",
    "  if (a < 0.004) discard;",
    "  vec3 sky = PI * textureLod(uEnv, vec2(0.5, 0.25), 7.0).rgb;",
    "  vec3 col = (uSunCol * 0.95 + sky) / PI * 0.9 + uGlowCol * 2.0;",
    "  o = vec4(col * a, a);",
    "}",
  ].join("\n");

  // Lip particles: 256^2 slots, state in two float textures,
  //   A = (position O-relative, life left), B = (velocity, age; age < 0: spray).
  // Dead slots look for a breaking front (aux.r from the surf simulation) by
  // rejection sampling and launch from its crest at the wave's speed; the
  // lip flies ballistically over the face and, landing, shatters into spray.
  var C_PCG = [
    "uint pcg(uint v) { uint st = v * 747796405u + 2891336453u; uint wd = ((st >> ((st >> 28u) + 4u)) ^ st) * 277803737u; return (wd >> 22u) ^ wd; }",
    "float rnd(inout uint s) { s = pcg(s); return float(s) * (1.0 / 4294967296.0); }",
  ].join("\n");
  var FS_LIP_UPDATE = HEAD + C_COMMON + "\n" + C_WORLD + "\n" + C_SWE + "\n" + C_PCG + "\n" + [
    "uniform sampler2D uPA, uPB, uSweAux;",
    "uniform float uDt, uFrame, uLipRate;",
    "uniform vec2 uOShift;",            // the grid origin moved: keep particles put in the world
    "layout(location = 0) out vec4 oA;",
    "layout(location = 1) out vec4 oB;",
    "float waterLevel(vec2 p) {",
    "  if (sweMask(p) <= 0.0) return 0.0;",
    "  float bed = worldAt(p, 0.0).x, h = sweAt(p).r;",
    "  return h > 0.02 ? bed + h : bed;",
    "}",
    "void main() {",
    "  ivec2 id = ivec2(gl_FragCoord.xy);",
    "  vec4 A = texelFetch(uPA, id, 0), B = texelFetch(uPB, id, 0);",
    "  uint seed = pcg(uint(id.x) + uint(id.y) * 256u + uint(uFrame) * 65536u);",
    "  if (A.w > 0.0) {",
    "    vec3 v = B.xyz; v.y -= 9.81 * uDt; v *= exp(-0.12 * uDt);",
    "    vec3 P = A.xyz + vec3(uOShift.x, 0.0, uOShift.y) + v * uDt;",
    "    float life = A.w - uDt;",
    "    bool lip = B.w >= 0.0;",
    "    float age = lip ? B.w + uDt : B.w - uDt;",
    "    float wl = waterLevel(P.xz);",
    "    if (P.y < wl - (lip ? 0.12 : 0.0) && v.y < 0.0) {",
    "      if (lip) {",                         // the lip lands and shatters into spray
    "        float r1 = rnd(seed), r2 = rnd(seed), r3 = rnd(seed);",
    "        v = vec3(v.x * 0.45 + (r1 - 0.5) * 2.4, abs(v.y) * 0.35 + 0.6 + r2 * 2.4, v.z * 0.45 + (r3 - 0.5) * 2.4);",
    "        P.y = wl + 0.05; life = 0.35 + 0.8 * rnd(seed); age = -1e-3;",
    "      } else life = 0.0;",
    "    }",
    "    oA = vec4(P, life); oB = vec4(v, age);",
    "    return;",
    "  }",
    "  oA = vec4(0.0); oB = vec4(0.0);",
    "  if (uSweOn < 0.5 || uLipRate <= 0.0) return;",
    "  ivec2 n = textureSize(uSweAux, 0);",
    "  float cell = uSweSize / float(n.x);",
    "  for (int i = 0; i < 6; i++) {",
    "    vec2 cu = vec2(rnd(seed), rnd(seed));",
    "    ivec2 c = ivec2(cu * vec2(n));",
    "    vec4 ax = texelFetch(uSweAux, c, 0);",
    // every dead slot tries a few cells, so each cell gets ~1.5 tries a
    // frame: accept in proportion to how hard it breaks and to the frame time
    "    if (ax.r * uLipRate * 120.0 * uDt <= rnd(seed)) continue;",
    "    vec2 dir = normalize(ax.gb + 1e-5);",
    "    vec2 p0 = uSweC + (vec2(c) + vec2(rnd(seed), rnd(seed)) - 0.5 * vec2(n)) * cell;",
    // launch from the top of the face: walk back up the slope to the crest
    "    vec2 best = p0; float top = -1e9, depth = 0.3;",
    "    for (int k = 0; k < 5; k++) {",
    "      vec2 q = p0 - dir * (float(k) * 0.35);",
    "      float hq = sweAt(q).r, e = worldAt(q, 0.0).x + hq;",
    "      if (e > top) { top = e; best = q; depth = hq; }",
    "    }",
    // a plunging crest outruns its wave (that's why it overturns): thrown
    // ahead of the face, the lip falls past its toe and leaves the tube
    "    float speed = sqrt(9.81 * max(depth, 0.3)) * (1.15 + 0.3 * rnd(seed));",
    "    oA = vec4(best.x, top + 0.02, best.y, 2.2);",
    "    oB = vec4(dir.x * speed, 0.3 + 0.9 * rnd(seed), dir.y * speed, 0.0);",
    "    return;",
    "  }",
    "}",
  ].join("\n");
  // Particles drawn as points straight from the state textures
  // (uKind 0: lip, 1: spray).
  var VS_LIP = HEAD + [
    "uniform sampler2D uPA, uPB;",
    "uniform mat4 uViewProj;",
    "uniform float uPointScale, uRadius, uKind, uMaxPoint;",
    "out float vR, vZ, vAge;",
    "void main() {",
    "  ivec2 n = textureSize(uPA, 0), t = ivec2(gl_VertexID % n.x, gl_VertexID / n.x);",
    "  vec4 A = texelFetch(uPA, t, 0), B = texelFetch(uPB, t, 0);",
    "  bool spray = B.w < 0.0;",
    "  if (A.w <= 0.0 || (uKind < 0.5) == spray) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; vR = vZ = vAge = 0.0; return; }",
    "  vec4 c = uViewProj * vec4(A.xyz, 1.0);",
    "  float r = uKind < 0.5 ? uRadius : 0.04 + 0.07 * fract(float(gl_VertexID) * 0.618034);",
    "  gl_Position = c;",
    "  gl_PointSize = clamp(2.0 * r * uPointScale / max(c.w, 0.05), 1.0, uMaxPoint);",
    "  vR = r; vZ = c.w; vAge = uKind < 0.5 ? B.w : sat01(A.w * 2.0) * smoothstep(0.4, 1.6, c.w);",
    "}",
  ].join("\n").replace("sat01(A.w * 2.0)", "clamp(A.w * 2.0, 0.0, 1.0)");
  // eye distance of the sphere's front surface (depth-tested by its centre)
  var FS_LIP_DEPTH = HEAD + [
    "in float vR, vZ, vAge;",
    "out vec4 o;",
    "void main() {",
    "  vec2 q = gl_PointCoord * 2.0 - 1.0;",
    "  float d2 = dot(q, q);",
    "  if (d2 > 1.0) discard;",
    "  o = vec4(vZ - vR * sqrt(1.0 - d2), 0.0, 0.0, 1.0);",
    "}",
  ].join("\n");
  // additive: water thickness along the view ray, and how aerated it is
  var FS_LIP_THICK = HEAD + [
    "in float vR, vZ, vAge;",
    "out vec4 o;",
    "void main() {",
    "  vec2 q = gl_PointCoord * 2.0 - 1.0;",
    "  float d2 = dot(q, q);",
    "  if (d2 > 1.0) discard;",
    "  float th = vR * sqrt(1.0 - d2);",
    "  o = vec4(th, th * smoothstep(0.35, 1.1, vAge), 0.0, 0.0);",
    "}",
  ].join("\n");
  // depth-aware blur that keeps silhouettes sharp
  var FS_LIP_BLUR = HEAD + [
    "uniform sampler2D uSrc;",
    "uniform vec2 uStep;",
    "in vec2 vUv;",
    "out vec4 o;",
    "void main() {",
    "  float z0 = texture(uSrc, vUv).r;",
    "  if (z0 > 9000.0) { o = vec4(z0); return; }",
    "  float sum = 0.0, wsum = 0.0, falloff = 1.0 / (0.12 + 0.01 * z0);",
    "  for (int k = -7; k <= 7; k++) {",
    "    float z = texture(uSrc, vUv + uStep * float(k)).r;",
    "    if (z > 9000.0) continue;",
    "    float dz = (z - z0) * falloff;",
    "    float wgt = exp(-float(k * k) / 18.0 - dz * dz);",
    "    sum += z * wgt; wsum += wgt;",
    "  }",
    "  o = vec4(sum / wsum, 0.0, 0.0, 1.0);",
    "}",
  ].join("\n");
  // The lip as a surface: normals from the smoothed depth, reflection,
  // refraction of what's behind, the body colour of the water it's made of,
  // the sun shining through thin water, and foam as it aerates.
  var FS_LIP_COMPOSITE = HEAD + C_COMMON + "\n" + C_NOISE + "\n" + C_ATMOS + "\n" + C_ENV + "\n" + C_WORLD + "\n" + C_SHADOW + "\n" + C_WATER + "\n" + C_FRESNEL + "\n" + [
    "uniform sampler2D uLipZ, uLipT, uSceneCopy, uSceneDepth;",
    "uniform vec2 uTanHalf;",
    "uniform vec3 uCamRight, uCamUp, uCamFwd;",
    "uniform float uNear, uFar;",
    "in vec2 vUv;",
    "out vec4 o;",
    "float lipZ(vec2 uv) { return texture(uLipZ, uv).r; }",
    "vec3 viewPos(vec2 uv, float z) { return vec3((uv * 2.0 - 1.0) * uTanHalf * z, z); }",
    "void main() {",
    "  vec3 base = texture(uSceneCopy, vUv).rgb;",
    "  float z = lipZ(vUv);",
    "  if (z > 9000.0) { o = vec4(base, 1.0); return; }",
    "  float d = texture(uSceneDepth, vUv).r;",
    "  float sz = d >= 1.0 ? 1e9 : 2.0 * uNear * uFar / (uFar + uNear - (d * 2.0 - 1.0) * (uFar - uNear));",
    "  if (z > sz + 0.05) { o = vec4(base, 1.0); return; }",
    "  vec2 tx = 1.0 / vec2(textureSize(uLipZ, 0));",
    "  vec3 P = viewPos(vUv, z);",
    "  float zr = lipZ(vUv + vec2(tx.x, 0.0)), zl = lipZ(vUv - vec2(tx.x, 0.0)), zu = lipZ(vUv + vec2(0.0, tx.y)), zd = lipZ(vUv - vec2(0.0, tx.y));",
    "  vec3 ddx = (abs(zr - z) < abs(z - zl) && zr < 9000.0) || zl > 9000.0 ? viewPos(vUv + vec2(tx.x, 0.0), zr < 9000.0 ? zr : z) - P : P - viewPos(vUv - vec2(tx.x, 0.0), zl);",
    "  vec3 ddy = (abs(zu - z) < abs(z - zd) && zu < 9000.0) || zd > 9000.0 ? viewPos(vUv + vec2(0.0, tx.y), zu < 9000.0 ? zu : z) - P : P - viewPos(vUv - vec2(0.0, tx.y), zd);",
    "  vec3 nv = normalize(cross(ddx, ddy));",
    "  if (nv.z > 0.0) nv = -nv;",                            // face the camera
    "  vec3 N = normalize(uCamRight * nv.x + uCamUp * nv.y + uCamFwd * nv.z);",
    "  vec3 V = -normalize(uCamRight * P.x + uCamUp * P.y + uCamFwd * P.z);",
    "  vec4 th = texture(uLipT, vUv);",
    "  float thick = th.r, aer = sat(th.g / max(th.r, 1e-3));",
    "  float NoV = max(dot(N, V), 0.02);",
    "  float F = fresnel(NoV, 1.0 / 1.333);",
    "  vec3 refl = envLod(reflect(-V, N), 1.0);",
    "  vec3 behind = texture(uSceneCopy, vUv + nv.xy * 0.035).rgb;",
    "  vec3 Tr = exp(-uSigT * thick * 2.5);",
    "  vec3 body = inscatter(1.0) * 1.6;",
    "  vec3 col = mix(behind * Tr + body * (1.0 - Tr), refl, F);",
    // sunlight through the thin curl: the glowing barrel
    "  float back = pow(sat(dot(-V, uSunDir) * 0.5 + 0.5), 3.0);",
    "  col += vec3(0.05, 0.50, 0.44) * uSunCol * 0.06 * back * exp(-thick * 1.2);",
    "  vec3 Hh = normalize(V + uSunDir);",
    "  col += uSunCol * 0.6 * pow(max(dot(N, Hh), 0.0), 180.0) * F;",
    "  vec3 foam = 0.9 * (uSunCol * max(dot(N, uSunDir), 0.0) * 0.8 + skyIrradiance()) / PI;",
    "  col = mix(col, foam, sat(aer * 1.3));",
    "  col += uGlowCol * (0.25 + 2.2 * aer) * (0.6 + 0.6 * vnoise(vUv * 90.0 + uTime * 2.0));",
    "  float a = smoothstep(0.0, 0.06, thick);",
    "  col = mix(base, col, a);",
    "  if (any(isnan(col))) col = base;",
    "  o = vec4(min(col, vec3(3e4)), 1.0);",
    "}",
  ].join("\n");
  var FS_SPRAY_POINT = HEAD + C_COMMON + "\n" + C_ENV + "\n" + [
    "in float vR, vZ, vAge;",
    "out vec4 o;",
    "void main() {",
    "  float d = length(gl_PointCoord * 2.0 - 1.0);",
    "  float a = (1.0 - smoothstep(0.2, 1.0, d)) * vAge * 0.75;",
    "  if (a < 0.004) discard;",
    "  vec3 sky = PI * textureLod(uEnv, vec2(0.5, 0.25), 7.0).rgb;",
    "  o = vec4(((uSunCol * 0.95 + sky) / PI * 0.9 + uGlowCol * 2.5) * a, a);",
    "}",
  ].join("\n");

  // ------------------------------------------------------------- post / bloom
  var FS_DOWN = HEAD + [
    "uniform sampler2D uSrc;",
    "uniform vec2 uTexel;",
    "uniform float uThreshold;",     // > 0 only on the first (bright-pass) level
    "in vec2 vUv;",
    "out vec4 o;",
    "void main() {",
    "  vec3 c = 0.25 * (texture(uSrc, vUv + uTexel * vec2(-1.0, -1.0)).rgb + texture(uSrc, vUv + uTexel * vec2(1.0, -1.0)).rgb",
    "                 + texture(uSrc, vUv + uTexel * vec2(-1.0, 1.0)).rgb + texture(uSrc, vUv + uTexel * vec2(1.0, 1.0)).rgb);",
    "  if (uThreshold > 0.0) {",
    "    c = clamp(c, 0.0, 3e4);",
    "    float l = max(c.r, max(c.g, c.b));",
    "    c *= max(l - uThreshold, 0.0) / max(l, 1e-4);",
    "    c = min(c, vec3(60.0));",
    "  }",
    "  o = vec4(c, 1.0);",
    "}",
  ].join("\n");
  var FS_UP = HEAD + [
    "uniform sampler2D uSrc;",
    "uniform vec2 uTexel;",
    "in vec2 vUv;",
    "out vec4 o;",
    "void main() {",
    "  vec3 c = vec3(0.0);",      // 3x3 tent
    "  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {",
    "    float w = (2.0 - abs(float(i))) * (2.0 - abs(float(j))) / 16.0;",
    "    c += w * texture(uSrc, vUv + uTexel * vec2(float(i), float(j))).rgb;",
    "  }",
    "  o = vec4(c, 1.0);",
    "}",
  ].join("\n");
  var FS_COMPOSITE = HEAD + [
    "uniform sampler2D uScene, uBloom;",
    "uniform float uExposure, uBloomStr, uNight;",
    "uniform vec3 uWhite;",
    "in vec2 vUv;",
    "out vec4 o;",
    "vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }",
    "void main() {",
    "  vec3 c = texture(uScene, vUv).rgb + texture(uBloom, vUv).rgb * uBloomStr;",
    "  c *= uExposure * uWhite;",
    // moonlight: rods see little colour and lean blue; bright things keep theirs
    "  float L = dot(c, vec3(0.15, 0.7, 0.15));",
    "  c = mix(c, L * vec3(0.6, 0.8, 1.25), uNight * 0.45 * (1.0 - smoothstep(0.15, 1.2, L)));",
    "  c = aces(c);",
    "  c = pow(c, vec3(1.0 / 2.2));",
    "  float n = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);",
    "  o = vec4(c + (n - 0.5) / 255.0, 1.0);",
    "}",
  ].join("\n");

  // ============================================================ GL plumbing
  function compile(gl, type, src, label) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      var log = gl.getShaderInfoLog(s), lines = src.split("\n");
      var m = /0:(\d+)/.exec(log || ""), at = m ? +m[1] : 0;
      var ctx = at ? "\n" + lines.slice(Math.max(0, at - 3), at + 2).map(function (l, i) { return (at - 2 + i) + ": " + l; }).join("\n") : "";
      throw new Error(label + " shader: " + log + ctx);
    }
    return s;
  }
  var SAMPLER_TYPES = {};
  function Program(gl, vs, fs, label) {
    var p = gl.createProgram();
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs, label + " vertex"));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs, label + " fragment"));
    // fixed locations so one grid VAO serves the water and the terrain
    gl.bindAttribLocation(p, 0, "aPos");
    gl.bindAttribLocation(p, 1, "aCell");
    ["aNrm", "aUV", "aKind", "aInst0", "aInst1", "aInst2"].forEach(function (n, i) { gl.bindAttribLocation(p, 2 + i, n); });
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(label + " link: " + gl.getProgramInfoLog(p));
    SAMPLER_TYPES[gl.SAMPLER_2D] = 1; SAMPLER_TYPES[gl.SAMPLER_2D_SHADOW] = 1;
    this.gl = gl; this.p = p; this.u = {};
    gl.useProgram(p);
    var n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS), unit = 0;
    for (var i = 0; i < n; i++) {
      var info = gl.getActiveUniform(p, i), name = info.name.replace(/\[0\]$/, "");
      var u = { loc: gl.getUniformLocation(p, info.name), type: info.type, unit: -1 };
      if (SAMPLER_TYPES[info.type]) { u.unit = unit++; gl.uniform1i(u.loc, u.unit); }
      this.u[name] = u;
    }
  }
  Program.prototype.use = function () { this.gl.useProgram(this.p); return this; };
  // Set uniforms from an object; names the compiler optimised away are skipped.
  Program.prototype.set = function (vals) {
    var gl = this.gl;
    for (var name in vals) {
      var u = this.u[name], v = vals[name];
      if (!u) continue;
      switch (u.type) {
        case gl.FLOAT: if (typeof v === "number") gl.uniform1f(u.loc, v); else gl.uniform1fv(u.loc, v); break;
        case gl.FLOAT_VEC2: gl.uniform2fv(u.loc, v); break;
        case gl.FLOAT_VEC3: gl.uniform3fv(u.loc, v); break;
        case gl.FLOAT_VEC4: gl.uniform4fv(u.loc, v); break;
        case gl.INT: case gl.BOOL: gl.uniform1i(u.loc, v); break;
        case gl.INT_VEC2: gl.uniform2iv(u.loc, v); break;
        case gl.FLOAT_MAT4: gl.uniformMatrix4fv(u.loc, false, v); break;
        default:
          if (u.unit >= 0) { gl.activeTexture(gl.TEXTURE0 + u.unit); gl.bindTexture(gl.TEXTURE_2D, v); }
      }
    }
    return this;
  };

  // ------------------------------------------------------------------ math
  function perspective(fovy, aspect, near, far) {
    var f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
  }
  function ortho(l, r, b, t, n, f) {
    return new Float32Array([2 / (r - l), 0, 0, 0, 0, 2 / (t - b), 0, 0, 0, 0, -2 / (f - n), 0,
      -(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n), 1]);
  }
  // Frustum planes (a, b, c, d; inside when a x + b y + c z + d > 0) from a view-projection matrix.
  function frustumPlanes(m) {
    var P = [], rows = [[m[0], m[4], m[8], m[12]], [m[1], m[5], m[9], m[13]], [m[2], m[6], m[10], m[14]], [m[3], m[7], m[11], m[15]]];
    [[3, 0, 1], [3, 0, -1], [3, 1, 1], [3, 1, -1], [3, 2, 1], [3, 2, -1]].forEach(function (c) {
      var a = rows[c[0]], b = rows[c[1]], pl = [a[0] + c[2] * b[0], a[1] + c[2] * b[1], a[2] + c[2] * b[2], a[3] + c[2] * b[3]];
      var l = Math.hypot(pl[0], pl[1], pl[2]);
      P.push([pl[0] / l, pl[1] / l, pl[2] / l, pl[3] / l]);
    });
    return P;
  }
  function lookDir(eye, fwd, up) {   // view matrix from eye + forward (up: y unless given, e.g. rolling on a swell)
    var f = norm3(fwd), s = norm3(cross3(f, up || [0, 1, 0])), u = cross3(s, f);
    return new Float32Array([s[0], u[0], -f[0], 0, s[1], u[1], -f[1], 0, s[2], u[2], -f[2], 0,
      -dot3(s, eye), -dot3(u, eye), dot3(f, eye), 1]);
  }
  function mul4(a, b) {
    var o = new Float32Array(16);
    for (var c = 0; c < 4; c++) for (var r = 0; r < 4; r++) {
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
    return o;
  }
  function inv4(m) {
    var a = m, o = new Float32Array(16);
    var b00 = a[0] * a[5] - a[1] * a[4], b01 = a[0] * a[6] - a[2] * a[4], b02 = a[0] * a[7] - a[3] * a[4];
    var b03 = a[1] * a[6] - a[2] * a[5], b04 = a[1] * a[7] - a[3] * a[5], b05 = a[2] * a[7] - a[3] * a[6];
    var b06 = a[8] * a[13] - a[9] * a[12], b07 = a[8] * a[14] - a[10] * a[12], b08 = a[8] * a[15] - a[11] * a[12];
    var b09 = a[9] * a[14] - a[10] * a[13], b10 = a[9] * a[15] - a[11] * a[13], b11 = a[10] * a[15] - a[11] * a[14];
    var det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    det = 1 / det;
    o[0] = (a[5] * b11 - a[6] * b10 + a[7] * b09) * det; o[1] = (a[2] * b10 - a[1] * b11 - a[3] * b09) * det;
    o[2] = (a[13] * b05 - a[14] * b04 + a[15] * b03) * det; o[3] = (a[10] * b04 - a[9] * b05 - a[11] * b03) * det;
    o[4] = (a[6] * b08 - a[4] * b11 - a[7] * b07) * det; o[5] = (a[0] * b11 - a[2] * b08 + a[3] * b07) * det;
    o[6] = (a[14] * b02 - a[12] * b05 - a[15] * b01) * det; o[7] = (a[8] * b05 - a[10] * b02 + a[11] * b01) * det;
    o[8] = (a[4] * b10 - a[5] * b08 + a[7] * b06) * det; o[9] = (a[1] * b08 - a[0] * b10 - a[3] * b06) * det;
    o[10] = (a[12] * b04 - a[13] * b02 + a[15] * b00) * det; o[11] = (a[9] * b02 - a[8] * b04 - a[11] * b00) * det;
    o[12] = (a[5] * b07 - a[4] * b09 - a[6] * b06) * det; o[13] = (a[0] * b09 - a[1] * b07 + a[2] * b06) * det;
    o[14] = (a[13] * b01 - a[12] * b03 - a[14] * b00) * det; o[15] = (a[8] * b03 - a[9] * b01 + a[10] * b00) * det;
    return o;
  }
  function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function cross3(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  function norm3(a) { var l = Math.sqrt(dot3(a, a)) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }
  function refract3(I, N, eta) {
    var c = -dot3(N, I), k = 1 - eta * eta * (1 - c * c);
    if (k < 0) return [0, 0, 0];
    var t = eta * c - Math.sqrt(k);
    return [eta * I[0] + t * N[0], eta * I[1] + t * N[1], eta * I[2] + t * N[2]];
  }
  function fresnelJS(cosI, eta) {
    var s2 = eta * eta * (1 - cosI * cosI);
    if (s2 >= 1) return 1;
    var cosT = Math.sqrt(1 - s2), rs = (eta * cosI - cosT) / (eta * cosI + cosT), rp = (cosI - eta * cosT) / (cosI + eta * cosT);
    return 0.5 * (rs * rs + rp * rp);
  }
  function mod(a, m) { return ((a % m) + m) % m; }

  // ========================================================== the renderer
  // world: an island from World.build() (world.js).
  // opts.life: a Life.Ecosystem (life.js) to draw the reef's animals.
  function Renderer(canvas, params, world, opts) {
    var gl = canvas.getContext("webgl2", { antialias: false, alpha: false, depth: false, powerPreference: "high-performance", preserveDrawingBuffer: false });
    if (!gl) throw new Error("This browser has no WebGL2, which the ocean needs.");
    var cbf = gl.getExtension("EXT_color_buffer_float");
    var cbh = cbf ? null : gl.getExtension("EXT_color_buffer_half_float");
    if (!cbf && !cbh) throw new Error("This GPU can't render to floating-point textures (EXT_color_buffer_float), which the ocean needs.");
    gl.getExtension("OES_texture_float_linear");
    this.aniso = gl.getExtension("EXT_texture_filter_anisotropic");
    this.gl = gl;
    this.canvas = canvas;
    this.fftFormat = cbf ? gl.RGBA32F : gl.RGBA16F;
    this.p = Object.assign({}, DEFAULTS, params || {});
    this.time = 0;
    this.cur = 0;               // foam ping-pong index
    this.frame = 0;
    this.cloudOff = [0, 0];
    this.vao = gl.createVertexArray();

    var mk = function (vs, fs, label) { return new Program(gl, vs, fs, label); };
    this.progSpectrum = mk(VS_FULL, FS_SPECTRUM, "spectrum");
    this.progFFT = mk(VS_FULL, FS_FFT, "fft");
    this.progAssemble = mk(VS_FULL, FS_ASSEMBLE, "assemble");
    this.progProbe = mk(VS_FULL, FS_PROBE, "probe");
    this.progAtmo = mk(VS_FULL, FS_ATMO, "atmosphere");
    this.progEnv = mk(VS_FULL, FS_ENV, "sky");
    this.progCaustic = mk(VS_CAUSTIC, FS_CAUSTIC, "caustics");
    this.progBack = mk(VS_FULL, FS_BACK, "background");
    this.progWater = mk(VS_WATER, FS_WATER, "water");
    this.progTerrain = mk(VS_TERRAIN, FS_TERRAIN, "terrain");
    this.progPlant = mk(VS_PLANT, FS_PLANT, "plants");
    this.progPlantShadow = mk(VS_PLANT, FS_PLANT_SHADOW, "plant shadows");
    this.progThing = mk(VS_THING, FS_THING, "reef life");
    this.progThingShadow = mk(VS_THING, FS_THING_SHADOW, "reef life shadows");
    this.progRipple = mk(VS_FULL, FS_RIPPLE, "ripples");
    this.progSweAdv = mk(VS_FULL, FS_SWE_ADV, "surf advect");
    this.progSweH = mk(VS_FULL, FS_SWE_H, "surf depth");
    this.progSweVel = mk(VS_FULL, FS_SWE_VEL, "surf velocity");
    this.progLipUpdate = mk(VS_FULL, FS_LIP_UPDATE, "lip update");
    this.progLipDepth = mk(VS_LIP, FS_LIP_DEPTH, "lip depth");
    this.progLipThick = mk(VS_LIP, FS_LIP_THICK, "lip thickness");
    this.progLipBlur = mk(VS_FULL, FS_LIP_BLUR, "lip smoothing");
    this.progLipComp = mk(VS_FULL, FS_LIP_COMPOSITE, "lip surface");
    this.progSprayPt = mk(VS_LIP, FS_SPRAY_POINT, "surf spray");
    this.progSpray = mk(VS_SPRAY, FS_SPRAY, "spray");
    this.progDown = mk(VS_FULL, FS_DOWN, "bloom down");
    this.progUp = mk(VS_FULL, FS_UP, "bloom up");
    this.progComposite = mk(VS_FULL, FS_COMPOSITE, "composite");

    var N = FFT_N, self = this;
    // FFT work buffers (shared by cascades) and per-cascade outputs.
    this.fftA = this._mrt([this._tex(N, N, this.fftFormat, gl.NEAREST, gl.CLAMP_TO_EDGE), this._tex(N, N, this.fftFormat, gl.NEAREST, gl.CLAMP_TO_EDGE)]);
    this.fftB = this._mrt([this._tex(N, N, this.fftFormat, gl.NEAREST, gl.CLAMP_TO_EDGE), this._tex(N, N, this.fftFormat, gl.NEAREST, gl.CLAMP_TO_EDGE)]);
    this.casc = CASCADE_L.map(function (L, c) {
      var der = self._tex(N, N, gl.RGBA16F, gl.LINEAR_MIPMAP_LINEAR, gl.REPEAT, true);
      var disp = [self._tex(N, N, gl.RGBA16F, gl.LINEAR_MIPMAP_LINEAR, gl.REPEAT, true), self._tex(N, N, gl.RGBA16F, gl.LINEAR_MIPMAP_LINEAR, gl.REPEAT, true)];
      return {
        L: L, der: der, disp: disp,
        fbo: [self._mrt([disp[0], der]), self._mrt([disp[1], der])],
        h0: self._tex(N, N, gl.RGBA32F, gl.NEAREST, gl.CLAMP_TO_EDGE),
        xi: gaussians(N * N, 1013 + 7919 * c),
      };
    });
    // 64 water-height probes (texel 0 = camera), read back a frame late
    // through a pixel-pack buffer so the CPU never waits on the GPU
    this.probe = this._tex(64, 1, this.fftFormat, gl.NEAREST, gl.CLAMP_TO_EDGE);
    this.probePBO = gl.createBuffer();
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.probePBO);
    gl.bufferData(gl.PIXEL_PACK_BUFFER, 64 * 16, gl.STREAM_READ);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    this.probeFence = null; this.probeWait = 0;
    this.probeRead = new Float32Array(64 * 4);
    this.probeHeights = null;
    this.probePts = new Float32Array(128);
    // interactive ripples
    var floatLinear = !!gl.getExtension("OES_texture_float_linear");
    var ripFmt = cbf && floatLinear ? gl.RGBA32F : gl.RGBA16F;
    this.RIP_N = 512; this.RIP_CELL = 0.2;
    this.rip = [0, 1].map(function () { return self._tex(self.RIP_N, self.RIP_N, ripFmt, gl.LINEAR, gl.CLAMP_TO_EDGE); });
    this.ripFbo = this.rip.map(function (t) { return self._mrt([t]); });
    this.ripCur = 0; this.ripC = null; this.drops = [];
    this.rip.forEach(function (t, i) { self._target(self.ripFbo[i], self.RIP_N, self.RIP_N); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); });
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    // surf-zone shallow-water simulation (ping-pong state + aux)
    this.SWE_N = 512; this.SWE_CELL = 0.4;
    var sweFmt = cbf ? gl.RGBA32F : gl.RGBA16F;
    this.sweS = [0, 1].map(function () { return self._tex(self.SWE_N, self.SWE_N, sweFmt, gl.NEAREST, gl.CLAMP_TO_EDGE); });
    this.sweA = [0, 1].map(function () { return self._tex(self.SWE_N, self.SWE_N, gl.RGBA16F, gl.NEAREST, gl.CLAMP_TO_EDGE); });
    this.sweSF = this.sweS.map(function (t) { return self._mrt([t]); });
    this.sweVF = [[self._mrt([this.sweS[0], this.sweA[0]]), self._mrt([this.sweS[0], this.sweA[1]])], [self._mrt([this.sweS[1], this.sweA[0]]), self._mrt([this.sweS[1], this.sweA[1]])]];
    this.sweCur = 0; this.sweAux = 0; this.sweCtr = null;
    // lip particles: 256^2 slots
    this.LIP_N = 256;
    var lipFmt = cbf ? gl.RGBA32F : gl.RGBA16F;
    this.lipA = [0, 1].map(function () { return self._tex(self.LIP_N, self.LIP_N, lipFmt, gl.NEAREST, gl.CLAMP_TO_EDGE); });
    this.lipB = [0, 1].map(function () { return self._tex(self.LIP_N, self.LIP_N, lipFmt, gl.NEAREST, gl.CLAMP_TO_EDGE); });
    this.lipF = [0, 1].map(function (i) { return self._mrt([self.lipA[i], self.lipB[i]]); });
    this.lipF.forEach(function (f) { self._target(f, self.LIP_N, self.LIP_N); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); });
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.lipCur = 0; this.lipFrame = 0; this.lipO = null;
    this.emptyVao = gl.createVertexArray();
    this.maxPoint = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1];
    // spray particles
    this.spray = { vao: gl.createVertexArray(), buf: gl.createBuffer(), n: 0, data: null };
    gl.bindVertexArray(this.spray.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.spray.buf);
    gl.bufferData(gl.ARRAY_BUFFER, 4000 * 32, gl.DYNAMIC_DRAW);
    [5, 6].forEach(function (loc, i) { gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 32, i * 16); gl.vertexAttribDivisor(loc, 1); });
    gl.bindVertexArray(null);
    this.dynamic = {};
    this.probeFbo = this._mrt([this.probe]);
    this.atmo = this._tex(256, 128, gl.RGBA16F, gl.LINEAR, gl.REPEAT);
    gl.bindTexture(gl.TEXTURE_2D, this.atmo);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.atmoFbo = this._mrt([this.atmo]);
    this.atmoKey = "";
    this.env = this._tex(512, 256, gl.RGBA16F, gl.LINEAR_MIPMAP_LINEAR, gl.REPEAT, true);
    this.envFbo = this._mrt([this.env]);
    this.caustic = this._tex(1024, 1024, gl.R16F, gl.LINEAR_MIPMAP_LINEAR, gl.REPEAT, true);
    this.causticFbo = this._mrt([this.caustic]);

    // Static textures: random noise and the island.
    var nz = new Uint8Array(256 * 256 * 4), rnd = mulberry32(99);
    this.noiseData = nz;
    for (var i = 0; i < nz.length; i++) nz[i] = (rnd() * 256) | 0;
    this.noise = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.noise);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 256, 0, gl.RGBA, gl.UNSIGNED_BYTE, nz);
    this._params(gl.LINEAR, gl.REPEAT);
    this.world = world;
    var WN = world.N;
    this.worldTex = this._tex(WN, WN, gl.RGBA16F, gl.LINEAR_MIPMAP_LINEAR, gl.CLAMP_TO_EDGE, true);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, WN, WN, gl.RGBA, gl.FLOAT, world.terrainTexture(world));
    gl.generateMipmap(gl.TEXTURE_2D);
    var maxH = 0;
    for (i = 0; i < world.height.length; i++) if (world.height[i] > maxH) maxH = world.height[i];
    this.maxH = maxH + 2;
    this.shoreTex = this._tex(WN, WN, gl.RGBA16F, gl.LINEAR_MIPMAP_LINEAR, gl.CLAMP_TO_EDGE, true);
    this.shoreKTex = this._tex(WN, WN, gl.RGBA16F, gl.LINEAR_MIPMAP_LINEAR, gl.CLAMP_TO_EDGE, true);
    this._uploadShore();

    this._buildCausticMesh(384);
    this._buildWaterMesh(this.p.meshDetail);
    this._buildPlants();
    this._buildReef();
    // animals (life.js), if the page provided an ecosystem
    this.life = (opts && opts.life) || null;
    this.fauna = [this._thingKind(world.fishMesh(), 10, 600), this._thingKind(world.turtleMesh(), 11, 8), this._thingKind(world.rayMesh(), 12, 8)];
    this.shadowMap = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.shadowMap);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.DEPTH_COMPONENT24, 2048, 2048);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_FUNC, gl.LEQUAL);
    this.shadowFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this.shadowMap, 0);
    gl.drawBuffers([gl.NONE]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.spectrumDirty = true;
    this.width = 0; this.height = 0;
  }

  // Shore-wave fields for the world's current swell (see world.js).
  Renderer.prototype._uploadShore = function () {
    var gl = this.gl, w = this.world, N = w.N;
    gl.bindTexture(gl.TEXTURE_2D, this.shoreTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, N, N, gl.RGBA, gl.FLOAT, w.shoreTexture(w));
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, this.shoreKTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, N, N, gl.RGBA, gl.FLOAT, w.shoreKTexture(w));
    gl.generateMipmap(gl.TEXTURE_2D);
  };
  // Re-run the shore model for new swell settings (about half a second of CPU).
  Renderer.prototype.rebuildShore = function () {
    var w = this.world;
    if (w.swell === this.p.surf && w.swellDir === this.p.windDir) return;
    w.setSwell(w, this.p.surf, this.p.windDir);
    this._uploadShore();
  };

  Renderer.prototype._params = function (filter, wrap) {
    var gl = this.gl;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter === gl.NEAREST ? gl.NEAREST : gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
  };
  Renderer.prototype._tex = function (w, h, ifmt, filter, wrap, mips) {
    var gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    var levels = mips ? Math.floor(Math.log2(Math.max(w, h))) + 1 : 1;
    gl.texStorage2D(gl.TEXTURE_2D, levels, ifmt, w, h);
    this._params(filter, wrap);
    if (mips && this.aniso) gl.texParameterf(gl.TEXTURE_2D, this.aniso.TEXTURE_MAX_ANISOTROPY_EXT, 8);
    t.w = w; t.h = h;
    return t;
  };
  Renderer.prototype._mrt = function (texs, depth) {
    var gl = this.gl, f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    var bufs = texs.map(function (t, i) {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0);
      return gl.COLOR_ATTACHMENT0 + i;
    });
    if (depth && depth instanceof WebGLTexture) gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depth, 0);
    else if (depth) gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
    gl.drawBuffers(bufs);
    var st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (st !== gl.FRAMEBUFFER_COMPLETE) throw new Error("framebuffer incomplete (0x" + st.toString(16) + ")");
    f.w = texs[0].w; f.h = texs[0].h;
    f.tex0 = texs[0]; f.tex1 = texs[1];
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return f;
  };
  // Instanced plant meshes: static vertex data plus a per-frame instance
  // buffer for each draw (camera view and shadow map).
  Renderer.prototype._buildPlants = function () {
    var gl = this.gl, w = this.world, self = this;
    // Each kind gets three draws: near and far LOD for the camera, and the
    // near mesh again for the shadow map.
    function kind(meshNear, meshFar, data, stride) {
      var k = { data: data, n: data.length / stride, sunKey: "" };
      k.vis = new Float32Array(k.n).fill(1);
      k.inst = new Float32Array(k.n * 9);
      k.instFar = new Float32Array(k.n * 9);
      ["main", "far", "shadow"].forEach(function (pass) {
        var mesh = pass === "far" ? meshFar : meshNear;
        var vao = gl.createVertexArray(), vb = gl.createBuffer(), ib = gl.createBuffer(), ibuf = gl.createBuffer();
        gl.bindVertexArray(vao);
        gl.bindBuffer(gl.ARRAY_BUFFER, vb);
        gl.bufferData(gl.ARRAY_BUFFER, mesh.vertices, gl.STATIC_DRAW);
        [[0, 3, 0], [2, 3, 3], [3, 2, 6], [4, 1, 8]].forEach(function (a) {
          gl.enableVertexAttribArray(a[0]);
          gl.vertexAttribPointer(a[0], a[1], gl.FLOAT, false, 36, a[2] * 4);
        });
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.indices, gl.STATIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, ibuf);
        gl.bufferData(gl.ARRAY_BUFFER, k.inst.byteLength, gl.DYNAMIC_DRAW);
        [[5, 4, 0], [6, 4, 4], [7, 1, 8]].forEach(function (a) {
          gl.enableVertexAttribArray(a[0]);
          gl.vertexAttribPointer(a[0], a[1], gl.FLOAT, false, 36, a[2] * 4);
          gl.vertexAttribDivisor(a[0], 1);
        });
        gl.bindVertexArray(null);
        k[pass] = { vao: vao, buf: ibuf, n: 0, count: mesh.indices.length };
      });
      return k;
    }
    this.palms = kind(w.palmMesh(7, false), w.palmMesh(7, true), w.plants.palms, 8);
    this.trees = kind(w.canopyMesh(7, false), w.canopyMesh(7, true), w.plants.trees, 8);
    this.palms.radius = 1.0; this.trees.radius = 1.3;
    void self;
  };
  // Island shadow on each plant (heightfield march toward the sun), redone
  // when the sun moves; per pixel this would cost far more.
  Renderer.prototype._plantSun = function (k, sun, key) {
    if (k.sunKey === key) return;
    k.sunKey = key;
    var w = this.world, d = k.data;
    for (var i = 0; i < k.n; i++) {
      var x = d[i * 8], y = d[i * 8 + 1] + d[i * 8 + 3] * 0.7, z = d[i * 8 + 2], s = 1, t = 1.5;
      if (sun[1] < 0.01) { k.vis[i] = 0; continue; }
      for (var st = 0; st < 20; st++) {
        var qy = y + sun[1] * t;
        if (qy > this.maxH) break;
        var hq = w.heightAt(w, x + sun[0] * t, z + sun[2] * t);
        s = Math.min(s, Math.max(0, Math.min(1, (qy - hq) / (0.05 * t + 0.4))));
        t = t * 1.38 + 1.2;
      }
      k.vis[i] = s;
    }
  };
  // Fill instance buffers with the plants that pass the test; with farFrom
  // set, those beyond 300 m of that point go to the far-LOD draw.
  Renderer.prototype._fillPlants = function (k, pass, O, test, farFrom) {
    var d = k.data, gl = this.gl, n = 0, nf = 0, out = k.inst, outF = k.instFar;
    for (var i = 0; i < k.n; i++) {
      var o = i * 8, x = d[o] - O[0], y = d[o + 1], z = d[o + 2] - O[1], r = d[o + 3] * k.radius;
      if (!test(x, y + r * 0.5, z, r)) continue;
      var far = farFrom && (x - farFrom[0]) * (x - farFrom[0]) + (z - farFrom[2]) * (z - farFrom[2]) > 300 * 300;
      var buf = far ? outF : out, q = (far ? nf++ : n++) * 9;
      buf[q] = x; buf[q + 1] = y; buf[q + 2] = z; buf[q + 3] = d[o + 3];
      buf[q + 4] = d[o + 4]; buf[q + 5] = d[o + 5]; buf[q + 6] = d[o + 6]; buf[q + 7] = d[o + 7]; buf[q + 8] = k.vis[i];
    }
    var P = k[pass];
    P.n = n;
    // (a zero count would mean "to the end of the array" in WebGL2)
    if (n) { gl.bindBuffer(gl.ARRAY_BUFFER, P.buf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, out, 0, n * 9); }
    if (farFrom) {
      k.far.n = nf;
      if (nf) { gl.bindBuffer(gl.ARRAY_BUFFER, k.far.buf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, outF, 0, nf * 9); }
    }
  };

  // ---------------------------------------------------------- interaction
  // Ripple drops for the next frame: [x, z, radius, velocity kick, foam].
  Renderer.prototype.addDrops = function (list) { for (var i = 0; i < list.length && this.drops.length < 24; i++) this.drops.push(list[i]); };
  // Up to 63 world (x, z) points whose water height to measure (slot 0 is
  // the camera); results come back from probeResults() a frame later.
  Renderer.prototype.setProbePoints = function (pts) {
    this.probeWorld = pts;
  };
  Renderer.prototype.probeResults = function () { return this.probeHeights; };
  // Surf currents (x, z per probe) measured with the heights; zero outside the simulation.
  Renderer.prototype.probeCurrentResults = function () { return this.probeCurrents; };
  Renderer.prototype.addDynamicKind = function (name, mesh, code, max) { this.dynamic[name] = this._thingKind(mesh, code, max); return this.dynamic[name]; };
  // fill(inst, O) writes 12-float instances relative to O and returns the count
  Renderer.prototype.setDynamic = function (name, fill) { this.dynamic[name].fill = fill; };
  Renderer.prototype.setSpray = function (fill) { this.spray.fill = fill; };
  Renderer.prototype._readProbes = function () {
    var gl = this.gl;
    if (!this.probeFence) return;
    // A fence only signals once the browser has returned to its event loop;
    // after a few frames without one (a busy page, or no event loop at all)
    // read anyway and accept the stall.
    var st = gl.clientWaitSync(this.probeFence, 0, 0);
    if (st === gl.TIMEOUT_EXPIRED && ++this.probeWait < 4) return;
    this.probeWait = 0;
    gl.deleteSync(this.probeFence); this.probeFence = null;
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.probePBO);
    gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, this.probeRead);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    var h = this.probeHeights || (this.probeHeights = new Float32Array(64));
    var cu = this.probeCurrents || (this.probeCurrents = new Float32Array(192));   // (x, z, weight) per probe
    for (var i = 0; i < 64; i++) { h[i] = this.probeRead[i * 4]; for (var c = 0; c < 3; c++) cu[i * 3 + c] = this.probeRead[i * 4 + 1 + c]; }
  };
  // One or two wave-equation steps; the grid recentres on `centre` (world).
  Renderer.prototype._stepRipples = function (dt, centre, O, common) {
    var gl = this.gl, N = this.RIP_N, cell = this.RIP_CELL;
    var c = [Math.round(centre[0] / cell), Math.round(centre[2] / cell)];
    var shift = [0, 0];
    if (this.ripC) shift = [c[0] - this.ripC[0], c[1] - this.ripC[1]];
    if (Math.abs(shift[0]) > N / 2 || Math.abs(shift[1]) > N / 2) shift = [N, N];   // jumped: start clean
    this.ripC = c;
    var ripC = [c[0] * cell - O[0], c[1] * cell - O[1]];
    var drops = new Float32Array(96), foams = new Float32Array(24), nd = Math.min(this.drops.length, 24);
    for (var i = 0; i < nd; i++) {
      var d = this.drops[i];
      drops[i * 4] = d[0] - O[0]; drops[i * 4 + 1] = d[1] - O[1]; drops[i * 4 + 2] = d[2]; drops[i * 4 + 3] = d[3]; foams[i] = d[4] || 0;
    }
    this.drops = [];
    var steps = 2, h = Math.min(dt, 0.05) / steps;
    for (var s = 0; s < steps; s++) {
      var src = this.ripCur, dst = 1 - src;
      this._target(this.ripFbo[dst], N, N);
      this.progRipple.use().set(common).set({
        uRipPrev: this.rip[src], uShift: s === 0 ? shift : [0, 0], uRipC: ripC, uCell: cell, uDt: h, uCmax: 2.2,
        uDrops: drops, uDropFoam: foams, uNDrops: s === 0 ? nd : 0,
      });
      this._fullscreen();
      this.ripCur = dst;
    }
    common.uRip = this.rip[this.ripCur]; common.uRipC = ripC; common.uRipSize = N * cell;
  };

  // Surf-zone shallow water: recentre on the camera, then substeps of
  // advect / depth / velocity (CFL-limited).
  Renderer.prototype._stepSurf = function (dt, centre, O, common) {
    var gl = this.gl, N = this.SWE_N, cell = this.SWE_CELL;
    var c = [Math.round(centre[0] / cell), Math.round(centre[2] / cell)];
    var shift = [0, 0], first = !this.sweCtr;
    if (this.sweCtr) shift = [c[0] - this.sweCtr[0], c[1] - this.sweCtr[1]];
    if (first || Math.abs(shift[0]) > N / 2 || Math.abs(shift[1]) > N / 2) shift = [N, N];   // start from the swell
    this.sweCtr = c;
    var ctr = [c[0] * cell - O[0], c[1] * cell - O[1]];
    common.uSweC = ctr; common.uSweSize = N * cell; common.uSweOn = this.p.surfSim ? 1 : 0;
    if (!this.p.surfSim) { common.uSwe = this.sweS[this.sweCur]; common.uSweAux = this.sweA[this.sweAux]; return; }
    // gravity waves cross a cell in ~0.024 s in 30 m of water: never step
    // longer than half that (on a slow frame the surf runs slow instead)
    var steps = Math.max(1, Math.min(6, Math.ceil(dt / 0.012))), h = Math.min(dt / steps, 0.012);
    if (dt <= 0) { steps = shift[0] || shift[1] ? 1 : 0; h = 1e-4; }
    for (var s = 0; s < steps; s++) {
      var a = this.sweCur, b = 1 - a, au = this.sweAux, ab = 1 - au;
      var u = { uCell: cell, uDt: h, uSweC: ctr };
      this._target(this.sweSF[b], N, N);
      this.progSweAdv.use().set(common).set(u).set({ uSweIn: this.sweS[a], uShift: s === 0 ? shift : [0, 0] });
      this._fullscreen();
      this._target(this.sweSF[a], N, N);
      this.progSweH.use().set(u).set({ uSweIn: this.sweS[b] });
      this._fullscreen();
      this._target(this.sweVF[b][ab], N, N);
      this.progSweVel.use().set(common).set(u).set({ uSweIn: this.sweS[a], uAuxIn: this.sweA[au], uAuxShift: s === 0 ? shift : [0, 0] });
      this._fullscreen();
      this.sweCur = b; this.sweAux = ab;
    }
    common.uSwe = this.sweS[this.sweCur]; common.uSweAux = this.sweA[this.sweAux];
  };

  // Advance the lip particles (after the surf step, before drawing).
  Renderer.prototype._stepLips = function (dt, O, common) {
    var gl = this.gl, N = this.LIP_N, a = this.lipCur, b = 1 - a;
    var shift = this.lipO ? [this.lipO[0] - O[0], this.lipO[1] - O[1]] : [0, 0];
    this.lipO = O.slice();
    this._target(this.lipF[b], N, N);
    this.progLipUpdate.use().set(common).set({
      uPA: this.lipA[a], uPB: this.lipB[a], uDt: Math.min(dt, 0.05), uFrame: (this.lipFrame++) % 65536,
      uLipRate: this.p.surfSim && dt > 0 ? this.p.lips : 0, uOShift: shift,
    });
    this._fullscreen();
    this.lipCur = b;
  };
  // Draw the lips as a liquid surface over the scene, then the surf spray.
  Renderer.prototype._drawLips = function (common, vp, w, h, camera) {
    var gl = this.gl, NP = this.LIP_N * this.LIP_N, self = this;
    var fw = norm3(camera.fwd), rt = norm3(cross3(fw, camera.up || [0, 1, 0])), upc = cross3(rt, fw);
    var f = 1 / Math.tan(camera.fov / 2);
    var pts = { uPA: this.lipA[this.lipCur], uPB: this.lipB[this.lipCur], uViewProj: vp, uRadius: 0.28, uMaxPoint: this.maxPoint };
    gl.bindVertexArray(this.emptyVao);
    // 1. eye depth of the nearest lip surface
    this._target(this.lipZF, this.lipW, this.lipH);
    gl.clearColor(1e4, 0, 0, 0); gl.clearDepth(1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS);
    this.progLipDepth.use().set(pts).set({ uKind: 0, uPointScale: 0.5 * this.lipH * f });
    gl.drawArrays(gl.POINTS, 0, NP);
    gl.disable(gl.DEPTH_TEST);
    // 2. thickness and aeration, additive
    this._target(this.lipTF, this.lipW, this.lipH);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    this.progLipThick.use().set(pts).set({ uKind: 0, uPointScale: 0.5 * this.lipH * f });
    gl.drawArrays(gl.POINTS, 0, NP);
    gl.disable(gl.BLEND);
    // 3. smooth the depth (separable, edge-preserving)
    this._target(this.lipZ2F, this.lipW, this.lipH);
    this.progLipBlur.use().set({ uSrc: this.lipZ, uStep: [1 / this.lipW, 0] });
    this._fullscreen();
    this._target(this.lipZBF, this.lipW, this.lipH);
    this.progLipBlur.use().set({ uSrc: this.lipZ2, uStep: [0, 1 / this.lipH] });
    this._fullscreen();
    // 4. composite over a copy of the scene
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.sceneFbo);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.copyFbo);
    gl.blitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    this._target(this.sceneColorFbo, w, h);
    this.progLipComp.use().set(common).set({
      uLipZ: this.lipZ, uLipT: this.lipT, uSceneCopy: this.sceneCopy, uSceneDepth: this.sceneDepth,
      uTanHalf: [Math.tan(camera.fov / 2) * w / h, Math.tan(camera.fov / 2)], uCamRight: rt, uCamUp: upc, uCamFwd: fw,
      uNear: camera.near || 0.25, uFar: 30000,
    });
    this._fullscreen();
    // 5. spray from the shattered lips
    this._target(this.sceneFbo, w, h);
    gl.enable(gl.DEPTH_TEST); gl.depthMask(false);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(this.emptyVao);
    this.progSprayPt.use().set(common).set(pts).set({ uKind: 1, uPointScale: 0.5 * h * f });
    gl.drawArrays(gl.POINTS, 0, NP);
    gl.disable(gl.BLEND); gl.depthMask(true); gl.disable(gl.DEPTH_TEST);
    void self;
  };

  // An instanced mesh with a 12-float instance layout (see VS_THING).
  Renderer.prototype._thingKind = function (mesh, code, maxInst) {
    var gl = this.gl, vao = gl.createVertexArray(), vb = gl.createBuffer(), ib = gl.createBuffer(), buf = gl.createBuffer();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vb);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.vertices, gl.STATIC_DRAW);
    [[0, 3, 0], [2, 3, 3], [3, 2, 6], [4, 1, 8]].forEach(function (a) {
      gl.enableVertexAttribArray(a[0]);
      gl.vertexAttribPointer(a[0], a[1], gl.FLOAT, false, 36, a[2] * 4);
    });
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.indices, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, maxInst * 48, gl.DYNAMIC_DRAW);
    [5, 6, 7].forEach(function (loc, i) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 48, i * 16);
      gl.vertexAttribDivisor(loc, 1);
    });
    gl.bindVertexArray(null);
    return { vao: vao, buf: buf, count: mesh.indices.length, code: code, inst: new Float32Array(maxInst * 12), n: 0, max: maxInst };
  };
  Renderer.prototype._drawThings = function (prog, kinds, extra) {
    var gl = this.gl;
    kinds.forEach(function (k) {
      if (!k.n) return;
      prog.set({ uObjKind: k.code }).set(extra || {});
      gl.bindVertexArray(k.vao);
      gl.drawElementsInstanced(gl.TRIANGLES, k.count, gl.UNSIGNED_SHORT, 0, k.n);
    });
  };

  // Coral colonies in world-anchored cells around the camera, densest where
  // the reef mask says coral; they grow in at the edge of the range, so the
  // reef gains detail as you swim toward it instead of popping.
  var REEF_TYPES = ["brain", "staghorn", "table", "fan", "soft", "boulder"];
  Renderer.prototype._buildReef = function () {
    var meshes = this.world.coralMeshes(7), self = this;
    this.reef = REEF_TYPES.map(function (name, i) { return self._thingKind(meshes[name], i, 2500); });
    this.reefKey = null;
  };
  function cellHash(i, j, salt) {
    var h = Math.imul(i, 374761393) ^ Math.imul(j, 668265263) ^ Math.imul(salt, 2246822519);
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
  }
  // The shaders' value noise and coral-rock relief, on the CPU, so colonies
  // and fish sit on the same lumpy surface the GPU draws.
  Renderer.prototype.vnoise = function (x, y) {
    var nz = this.noiseData, ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
    // LINEAR filtering of texel centres at (i + f + 0.5) / 256 blends texels i and i+1
    var i0 = ((ix % 256) + 256) % 256, j0 = ((iy % 256) + 256) % 256, i1 = (i0 + 1) % 256, j1 = (j0 + 1) % 256;
    var a = nz[(j0 * 256 + i0) * 4], b = nz[(j0 * 256 + i1) * 4], c = nz[(j1 * 256 + i0) * 4], d = nz[(j1 * 256 + i1) * 4];
    return ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy) / 255;
  };
  Renderer.prototype.coralLump = function (x, z, coral, h, dist) {
    if (coral < 0.05 || dist > 55) return 0;
    var P = 256 / 0.55 * 11, qx = mod(x, P), qz = mod(z, P), vn = this.vnoise.bind(this);
    function ss(a, b, v) { var t = Math.max(0, Math.min(1, (v - a) / (b - a))); return t * t * (3 - 2 * t); }
    var l = (vn(qx * 0.55, qz * 0.55) - 0.5) * 1.0 + (vn(qx * 1.7 + 3.1, qz * 1.7 + 3.1) - 0.5) * 0.45;
    l += (vn(qx * 4.6 + 7.7, qz * 4.6 + 7.7) - 0.5) * 0.18 * (1 - ss(6, 18, dist));
    l += (vn(qx * 11 + 1.3, qz * 11 + 1.3) - 0.5) * 0.07 * (1 - ss(3, 9, dist));
    return l * ss(0.05, 0.5, coral) * (1 - ss(30, 55, dist)) * ss(-0.2, -0.8, h);
  };
  // Ground height including the coral-rock relief near camera position cam.
  Renderer.prototype.reefGround = function (x, z, cam) {
    var w = this.world, h = w.heightAt(w, x, z);
    if (h > -0.2) return h;
    var d = cam ? Math.hypot(x - cam[0], z - cam[2]) : 0;
    return h + this.coralLump(x, z, w.bilinear(w.coral, x, z, 0), h, d);
  };
  Renderer.prototype._updateReef = function (camW, O, R) {
    var w = this.world, CELL = 1.25;
    var key = Math.round(camW[0] / 1.5) + "," + Math.round(camW[2] / 1.5) + "," + R + "," + O[0].toFixed(2) + "," + O[1].toFixed(2);
    if (key === this.reefKey) return;
    this.reefKey = key;
    var kinds = this.reef, gl = this.gl;
    kinds.forEach(function (k) { k.n = 0; });
    var ci0 = Math.floor((camW[0] - R) / CELL), ci1 = Math.ceil((camW[0] + R) / CELL);
    var cj0 = Math.floor((camW[2] - R) / CELL), cj1 = Math.ceil((camW[2] + R) / CELL);
    for (var cj = cj0; cj <= cj1; cj++) {
      for (var ci = ci0; ci <= ci1; ci++) {
        var r1 = cellHash(ci, cj, 1), r2 = cellHash(ci, cj, 2), r3 = cellHash(ci, cj, 3);
        var x = (ci + r1) * CELL, z = (cj + r2) * CELL;
        if ((x - camW[0]) * (x - camW[0]) + (z - camW[2]) * (z - camW[2]) > R * R) continue;
        var g0 = w.heightAt(w, x, z);
        if (g0 > -0.5 || g0 < -28) continue;
        var cm = w.bilinear(w.coral, x, z, 0);
        // sit on the lumpy rock the GPU draws (at its full-detail relief)
        var g = g0 + this.coralLump(x, z, cm, g0, Math.hypot(x - camW[0], z - camW[2]));
        if (r3 > (cm > 0.15 ? 0.2 + 0.65 * cm : 0.01)) continue;
        var r4 = cellHash(ci, cj, 4), r5 = cellHash(ci, cj, 5), r6 = cellHash(ci, cj, 6), deep = g < -3;
        // shallow reef: brain, staghorn, table, boulder; deeper: fans and soft corals join
        var t = deep ? (r4 < 0.15 ? 0 : r4 < 0.3 ? 1 : r4 < 0.4 ? 2 : r4 < 0.6 ? 3 : r4 < 0.8 ? 4 : 5)
                     : (r4 < 0.25 ? 0 : r4 < 0.5 ? 1 : r4 < 0.72 ? 2 : r4 < 0.8 ? 4 : 5);
        var k = kinds[t];
        if (k.n >= k.max) continue;
        var size = [[0.4, 0.8], [0.5, 0.7], [0.5, 0.75], [0.6, 0.8], [0.45, 0.45], [0.6, 1.2]][t];
        var sc = size[0] + size[1] * r5, yaw = r6 * TAU, q = k.n * 12;
        k.inst[q] = x - O[0]; k.inst[q + 1] = g - 0.04; k.inst[q + 2] = z - O[1]; k.inst[q + 3] = sc;
        k.inst[q + 4] = 0; k.inst[q + 5] = Math.sin(yaw / 2); k.inst[q + 6] = 0; k.inst[q + 7] = Math.cos(yaw / 2);
        k.inst[q + 8] = cellHash(ci, cj, 7); k.inst[q + 9] = t; k.inst[q + 10] = 0; k.inst[q + 11] = 0;
        k.n++;
      }
    }
    kinds.forEach(function (k) {
      if (k.n) { gl.bindBuffer(gl.ARRAY_BUFFER, k.buf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, k.inst, 0, k.n * 12); }
    });
  };

  Renderer.prototype._grid = function (n, attribs) {
    var gl = this.gl, vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    var idx = new Uint32Array((n - 1) * (n - 1) * 6), k = 0;
    for (var j = 0; j < n - 1; j++) {
      for (var i = 0; i < n - 1; i++) {
        var a = j * n + i, b = a + 1, c = a + n, d = c + 1;
        // flip the diagonal per quadrant so the grid is symmetric about its centre
        if ((i < n / 2) === (j < n / 2)) { idx[k++] = a; idx[k++] = c; idx[k++] = d; idx[k++] = a; idx[k++] = d; idx[k++] = b; }
        else { idx[k++] = a; idx[k++] = c; idx[k++] = b; idx[k++] = b; idx[k++] = c; idx[k++] = d; }
      }
    }
    var ib = gl.createBuffer(), bufs = [ib];
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    attribs.forEach(function (at) {
      var vb = gl.createBuffer();
      bufs.push(vb);
      gl.bindBuffer(gl.ARRAY_BUFFER, vb);
      gl.bufferData(gl.ARRAY_BUFFER, at.data, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(at.loc);
      gl.vertexAttribPointer(at.loc, at.size, gl.FLOAT, false, 0, 0);
    });
    gl.bindVertexArray(null);
    return { vao: vao, count: idx.length, buffers: bufs };
  };
  Renderer.prototype._buildCausticMesh = function (n) {
    var pos = new Float32Array(n * n * 2);
    for (var j = 0; j < n; j++) for (var i = 0; i < n; i++) { pos[2 * (j * n + i)] = i / (n - 1); pos[2 * (j * n + i) + 1] = j / (n - 1); }
    var loc = this.gl.getAttribLocation(this.progCaustic.p, "aPos");
    this.causticMesh = this._grid(n, [{ loc: loc, size: 2, data: pos }]);
    this.causticMesh.cell = (CAUSTIC_L + 2 * 2.5) / (n - 1);
  };
  // Camera-centred grid out to R metres; x = sign(u) R (e^{a|u|} - 1)/(e^a - 1)
  // with `a` solved so the centre cells are cell0 wide.
  Renderer.prototype._buildWaterMesh = function (n) {
    if (this.waterMesh) {
      var gl0 = this.gl;
      gl0.deleteVertexArray(this.waterMesh.vao);
      this.waterMesh.buffers.forEach(function (b) { gl0.deleteBuffer(b); });
    }
    var R = 12000, cell0 = 0.08 * 384 / n, du = 2 / (n - 1);
    var lo = 0.1, hi = 30;
    for (var it = 0; it < 80; it++) {
      var a = 0.5 * (lo + hi);
      if (R * a / (Math.exp(a) - 1) * du > cell0) lo = a; else hi = a;
    }
    var A = 0.5 * (lo + hi), E = Math.exp(A) - 1;
    function warp(u) { return Math.sign(u) * R * (Math.exp(A * Math.abs(u)) - 1) / E; }
    function cell(u) { return R * A * Math.exp(A * Math.abs(u)) / E * du; }
    var pos = new Float32Array(n * n * 2), cs = new Float32Array(n * n);
    for (var j = 0; j < n; j++) {
      var v = -1 + j * du;
      for (var i = 0; i < n; i++) {
        var u = -1 + i * du, k = j * n + i;
        pos[2 * k] = warp(u); pos[2 * k + 1] = warp(v);
        cs[k] = Math.max(cell(u), cell(v));
      }
    }
    var gl = this.gl, p = this.progWater.p;
    this.waterMesh = this._grid(n, [{ loc: gl.getAttribLocation(p, "aPos"), size: 2, data: pos }, { loc: gl.getAttribLocation(p, "aCell"), size: 1, data: cs }]);
    this.waterMesh.cell0 = cell(0);
    this.waterMesh.n = n;
  };

  Renderer.prototype.setParams = function (q) {
    var wave = ["wind", "windDir", "fetch", "alignment", "depth", "swell"];
    for (var k in q) {
      if (!(k in DEFAULTS) || this.p[k] === q[k]) continue;
      this.p[k] = q[k];
      if (wave.indexOf(k) >= 0) this.spectrumDirty = true;
      if (k === "meshDetail") this._buildWaterMesh(q[k]);
    }
  };
  // Ground height (m above sea level) at world (x, z); the sea floor is negative.
  Renderer.prototype.groundHeightAt = function (x, z) { return this.world.heightAt(this.world, x, z); };
  Renderer.prototype.significantWaveHeight = function () { return this.hs || 0; };

  Renderer.prototype._uploadSpectrum = function () {
    var gl = this.gl, self = this, v = 0;
    this.casc.forEach(function (c, i) {
      var s = buildSpectrum(i, self.p, c.xi);
      v += s.variance;
      gl.bindTexture(gl.TEXTURE_2D, c.h0);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, FFT_N, FFT_N, gl.RGBA, gl.FLOAT, s.data);
    });
    this.hs = 4 * Math.sqrt(v);
    this.spectrumDirty = false;
  };

  Renderer.prototype._resizeTargets = function (w, h) {
    var gl = this.gl;
    if (this.sceneTex) {
      gl.deleteTexture(this.sceneTex); gl.deleteTexture(this.sceneDepth); gl.deleteFramebuffer(this.sceneFbo); gl.deleteFramebuffer(this.sceneColorFbo);
      this.bloom.forEach(function (b) { gl.deleteTexture(b.tex); gl.deleteFramebuffer(b.fbo); });
    }
    this.sceneTex = this._tex(w, h, gl.RGBA16F, gl.LINEAR, gl.CLAMP_TO_EDGE);
    // scene depth as a texture: the lip surface tests itself against it
    this.sceneDepth = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.sceneDepth);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.DEPTH_COMPONENT24, w, h);
    this._params(gl.NEAREST, gl.CLAMP_TO_EDGE);
    this.sceneFbo = this._mrt([this.sceneTex], this.sceneDepth);
    this.sceneColorFbo = this._mrt([this.sceneTex]);
    // half-resolution lip surface targets
    if (this.lipZ) { [this.lipZ, this.lipT, this.lipZ2].forEach(function (t) { gl.deleteTexture(t); }); gl.deleteRenderbuffer(this.lipDepthRB); }
    var hw = Math.max(1, w >> 1), hh = Math.max(1, h >> 1);
    this.lipZ = this._tex(hw, hh, gl.RGBA16F, gl.LINEAR, gl.CLAMP_TO_EDGE);
    this.lipZ2 = this._tex(hw, hh, gl.RGBA16F, gl.LINEAR, gl.CLAMP_TO_EDGE);
    this.lipT = this._tex(hw, hh, gl.RGBA16F, gl.LINEAR, gl.CLAMP_TO_EDGE);
    this.lipDepthRB = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, this.lipDepthRB);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, hw, hh);
    this.lipZF = this._mrt([this.lipZ], this.lipDepthRB);
    this.lipZ2F = this._mrt([this.lipZ2]);
    this.lipZBF = this._mrt([this.lipZ]);
    this.lipTF = this._mrt([this.lipT]);
    this.lipW = hw; this.lipH = hh;
    if (this.sceneCopy) { gl.deleteTexture(this.sceneCopy); gl.deleteFramebuffer(this.copyFbo); }
    this.sceneCopy = this._tex(w, h, gl.RGBA16F, gl.LINEAR, gl.CLAMP_TO_EDGE);
    this.copyFbo = this._mrt([this.sceneCopy]);
    this.bloom = [];
    var bw = w, bh = h;
    for (var i = 0; i < 6; i++) {
      bw = Math.max(1, bw >> 1); bh = Math.max(1, bh >> 1);
      var t = this._tex(bw, bh, gl.RGBA16F, gl.LINEAR, gl.CLAMP_TO_EDGE);
      this.bloom.push({ tex: t, fbo: this._mrt([t]), w: bw, h: bh });
    }
    this.width = w; this.height = h;
  };

  Renderer.prototype._target = function (fbo, w, h) {
    var gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, w, h);
  };
  Renderer.prototype._fullscreen = function () {
    this.gl.bindVertexArray(this.vao);
    this.gl.drawArrays(this.gl.TRIANGLES, 0, 3);
  };

  // Advance the waves to this.time.
  Renderer.prototype._simulate = function (dt) {
    var gl = this.gl, N = FFT_N, self = this, p = this.p;
    if (this.spectrumDirty) this._uploadSpectrum();
    gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST);
    var prev = this.cur; this.cur ^= 1;
    // Foam half-life ~1.4 s; injection scales with the foam slider.
    var decay = Math.pow(0.5, dt / 1.4);
    this.casc.forEach(function (c) {
      self._target(self.fftA, N, N);
      self.progSpectrum.use().set({ uH0: c.h0, uTime: self.time, uL: c.L, uDepth: Math.max(0.5, p.depth), uN: N });
      self._fullscreen();
      var src = self.fftA, dst = self.fftB;
      for (var horiz = 1; horiz >= 0; horiz--) {
        for (var st = 0; st < 8; st++) {
          self._target(dst, N, N);
          self.progFFT.use().set({ uIn0: src.tex0, uIn1: src.tex1, uStage: st, uHoriz: horiz, uN: N });
          self._fullscreen();
          var t = src; src = dst; dst = t;
        }
      }
      self._target(c.fbo[self.cur], N, N);
      self.progAssemble.use().set({
        uF0: src.tex0, uF1: src.tex1, uPrev: c.disp[prev], uChop: p.chop,
        uDecay: decay, uFoamBias: 0.55 + 0.35 * p.foam, uFoamGain: dt * 6.0,
      });
      self._fullscreen();
    });
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.casc.forEach(function (c) {
      gl.bindTexture(gl.TEXTURE_2D, c.disp[self.cur]); gl.generateMipmap(gl.TEXTURE_2D);
      gl.bindTexture(gl.TEXTURE_2D, c.der); gl.generateMipmap(gl.TEXTURE_2D);
    });
  };

  // camera: { pos: [x, y, z] (world), fwd: [x, y, z], fov (radians) }
  Renderer.prototype.render = function (dt, camera) {
    var gl = this.gl, p = this.p;
    var w = Math.max(1, Math.round(this.canvas.width * p.renderScale)), h = Math.max(1, Math.round(this.canvas.height * p.renderScale));
    if (w !== this.width || h !== this.height) this._resizeTargets(w, h);
    this.time += dt * p.timeScale;
    this.frame++;
    this._simulate(Math.max(dt * p.timeScale, 1e-4));

    // Grid origin O: camera xz snapped to the finest mesh cell.
    var cell0 = this.waterMesh.cell0;
    var O = [Math.round(camera.pos[0] / cell0) * cell0, Math.round(camera.pos[2] / cell0) * cell0];
    var cam = [camera.pos[0] - O[0], camera.pos[1], camera.pos[2] - O[1]];
    // near plane at 25 cm keeps the depth buffer fine enough to separate
    // water from beach a kilometre away
    // (a camera floating at the waterline asks for a closer one: camera.near)
    var near = camera.near || 0.25;
    var proj = perspective(camera.fov, w / h, near, 30000);
    var view = lookDir(cam, camera.fwd, camera.up);
    var vp = mul4(proj, view);

    // Sun, sky and water optics.
    var el = p.sunElev * Math.PI / 180, az = p.sunAzim * Math.PI / 180;
    var sun = [Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az)];
    var trans = sunTransmittance(sun);
    // At night the moon takes the sun's place, a few hundred times dimmer
    // (really ~400000; the eye adapts the rest, see exposure below).
    var night = Math.max(0, Math.min(1, p.night)), lightK = Math.pow(1 / 300, night);
    // moonlight is shot day-for-night: a cool blue the eye reads as moonlit
    var moonTint = [0.62, 0.8, 1.25].map(function (v) { return 1 + (v - 1) * night; });
    var sunCol = trans.map(function (t, i) { return t * SUN_I * lightK * moonTint[i]; });
    // disk radiance = irradiance / solid angle, capped to keep bloom sane
    var sunDisk = trans.map(function (t) { return Math.min(t * SUN_I / 6.8e-5, 4000) * lightK; });
    var gn = Math.max(0, Math.min(1, (night - 0.3) / 0.6)), glowK = p.glow * gn * gn * (3 - 2 * gn) * 0.3;
    this.exposureNow = p.exposure * Math.pow(45, night);
    var sunRefr = refract3([-sun[0], -sun[1], -sun[2]], [0, 1, 0], 1 / 1.333);
    var sunTrans = 1 - fresnelJS(Math.max(sun[1], 0.01), 1 / 1.333);
    var clear = Math.max(0.15, p.clarity);
    // Absorption close to pure water (red gone in metres, blue survives tens);
    // murkier water adds particles that absorb and scatter more evenly.
    var sigA = [0.34, 0.058, 0.014].map(function (v, i) { return v + [0.03, 0.025, 0.02][i] * (1 / clear - 1) * 2; });
    var sigS = [0.009, 0.012, 0.017].map(function (v) { return v / Math.sqrt(clear); });
    var sigT = sigA.map(function (v, i) { return v + sigS[i]; });
    var causDepth = 4.5;                       // typical lagoon depth
    var causShift = [sunRefr[0] / -sunRefr[1], sunRefr[2] / -sunRefr[1]];
    this.cloudOff[0] += dt * p.timeScale * 4.0; this.cloudOff[1] += dt * p.timeScale * 2.0;
    // swell and wave-group phases, wrapped in double precision
    var T = this.world.SWELL_T, shorePh = mod(TAU * this.time / T, TAU), groupPh = mod(TAU * this.time / (T * this.world.GROUP), TAU);

    var common = {
      uCam: cam, uSunDir: sun, uSunCol: sunCol, uTime: this.time,
      uGlowCol: [0.03, 0.4, 1.0].map(function (v) { return v * glowK; }), uNight: night,
      uDisp0: this.casc[0].disp[this.cur], uDisp1: this.casc[1].disp[this.cur], uDisp2: this.casc[2].disp[this.cur],
      uDer0: this.casc[0].der, uDer1: this.casc[1].der, uDer2: this.casc[2].der,
      uCascL: CASCADE_L, uN: FFT_N,
      uOff0: [mod(O[0], CASCADE_L[0]), mod(O[1], CASCADE_L[0])],
      uOff1: [mod(O[0], CASCADE_L[1]), mod(O[1], CASCADE_L[1])],
      uOff2: [mod(O[0], CASCADE_L[2]), mod(O[1], CASCADE_L[2])],
      uNoise: this.noise,
      uAtmo: this.atmo,
      uCloudOff: [mod(O[0] + this.cloudOff[0], 1100 * 256), mod(O[1] + this.cloudOff[1], 1100 * 256)],
      uCloudCover: p.clouds,
      uEnv: this.env,
      uSigT: sigT, uSigS: sigS, uSunRefr: sunRefr, uSunTrans: sunTrans,
      uWorld: this.worldTex, uWorldInv: 1 / this.world.SIZE, uMaxH: this.maxH,
      uWorldUV0: [(O[0] + this.world.HALF) / this.world.SIZE, (O[1] + this.world.HALF) / this.world.SIZE],
      uOffD: [mod(O[0], 4096), mod(O[1], 4096)], uOffR: [mod(O[0], 256 / 0.55 * 11), mod(O[1], 256 / 0.55 * 11)],
      uCaustic: this.caustic, uCausL: CAUSTIC_L, uCausDepth: causDepth, uCausStr: p.caustics,
      uShore: this.shoreTex, uShoreK: this.shoreKTex, uSurf: p.surf > 0 ? 1 : 0,
      uShoreWT: [Math.cos(shorePh), Math.sin(shorePh)], uGroupWT: [Math.cos(groupPh), Math.sin(groupPh)],
      uOffC: [mod(O[0], CAUSTIC_L), mod(O[1], CAUSTIC_L)], uCausShift: causShift,
      uProbe: this.probe,
    };

    // Sky LUT, only when the sun has moved.
    var key = sun.map(function (v) { return v.toFixed(5); }).join() + "," + lightK.toFixed(6);
    if (key !== this.atmoKey) {
      this.atmoKey = key;
      this._target(this.atmoFbo, 256, 128);
      this.progAtmo.use().set({ uSunDir: sun, uSunI: SUN_I * lightK, uSkyBoost: 2.0 });
      this._fullscreen();
    }

    // Ripples, then water height at the camera and the probe points.
    this._readProbes();
    this._stepRipples(dt * p.timeScale, camera.rippleCentre || camera.pos, O, common);
    this._stepSurf(dt * p.timeScale, camera.rippleCentre || camera.pos, O, common);
    this._stepLips(dt * p.timeScale, O, common);
    var pp = this.probePts;
    for (var pi = 0; pi < 64; pi++) {
      var pw = this.probeWorld && this.probeWorld[pi];
      pp[pi * 2] = pw ? pw[0] - O[0] : 0; pp[pi * 2 + 1] = pw ? pw[1] - O[1] : 0;
    }
    this._target(this.probeFbo, 64, 1);
    this.progProbe.use().set(common).set({ uProbePts: pp });
    this._fullscreen();
    if (!this.probeFence) {
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.probePBO);
      gl.readPixels(0, 0, 64, 1, gl.RGBA, gl.FLOAT, 0);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      this.probeFence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    }

    // Caustics for the current waves.
    this._target(this.causticFbo, 1024, 1024);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    var cm = this.causticMesh;
    this.progCaustic.use().set(common).set({
      uMargin: 2.5, uShift0: [causShift[0] * causDepth, causShift[1] * causDepth],
      uLod1: Math.max(0, Math.log2(cm.cell * FFT_N / CASCADE_L[1])), uLod2: Math.max(0, Math.log2(cm.cell * FFT_N / CASCADE_L[2])),
    });
    gl.bindVertexArray(cm.vao);
    gl.drawElements(gl.TRIANGLES, cm.count, gl.UNSIGNED_INT, 0);
    gl.disable(gl.BLEND);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, this.caustic); gl.generateMipmap(gl.TEXTURE_2D);

    // Sky environment for reflections.
    this._target(this.envFbo, 512, 256);
    this.progEnv.use().set(common);
    this._fullscreen();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, this.env); gl.generateMipmap(gl.TEXTURE_2D);

    // Plants: cull to the view and to the shadow box around the camera.
    var sunKey = sun.map(function (v) { return v.toFixed(4); }).join();
    this._plantSun(this.palms, sun, sunKey);
    this._plantSun(this.trees, sun, sunKey);
    var planes = frustumPlanes(vp);
    var inView = function (x, y, z, r) {
      if ((x - cam[0]) * (x - cam[0]) + (z - cam[2]) * (z - cam[2]) > 5000 * 5000) return false;
      for (var q = 0; q < 6; q++) { var pl = planes[q]; if (pl[0] * x + pl[1] * y + pl[2] * z + pl[3] < -r) return false; }
      return true;
    };
    // shadow box: SHADOW_R metres around the camera, snapped to texels so
    // shadows don't crawl as the camera moves
    var SR = 110, f = [-sun[0], -sun[1], -sun[2]], right = norm3(cross3(f, [0, 1, 0])), upv = cross3(right, f);
    var texel = 2 * SR / 2048, cr = dot3(cam, right), cu = dot3(cam, upv), cf = dot3(cam, f);
    cr = Math.round(cr / texel) * texel; cu = Math.round(cu / texel) * texel;
    var center = [right[0] * cr + upv[0] * cu + f[0] * cf, right[1] * cr + upv[1] * cu + f[1] * cf, right[2] * cr + upv[2] * cu + f[2] * cf];
    var lightView = lookDir([center[0] - f[0] * 600, center[1] - f[1] * 600, center[2] - f[2] * 600], f);
    var lightVP = mul4(ortho(-SR, SR, -SR, SR, 1, 1400), lightView);
    var inBox = function (x, y, z, r) {
      var dx = x - center[0], dy = y - center[1], dz = z - center[2];
      return Math.abs(dx * right[0] + dy * right[1] + dz * right[2]) < SR + r && Math.abs(dx * upv[0] + dy * upv[1] + dz * upv[2]) < SR + r;
    };
    var self2 = this;
    [this.palms, this.trees].forEach(function (k) { self2._fillPlants(k, "main", O, inView, cam); self2._fillPlants(k, "shadow", O, inBox); });
    var windRad = p.windDir * Math.PI / 180, plantU = { uWindXZ: [Math.cos(windRad), Math.sin(windRad)], uWindStr: Math.min(p.wind / 6, 2.5) };
    this._target(this.shadowFbo, 2048, 2048);
    gl.clearDepth(1); gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS);
    this.progPlantShadow.use().set(common).set(plantU).set({ uViewProj: lightVP });
    [this.palms, this.trees].forEach(function (k) {
      if (!k.shadow.n) return;
      gl.bindVertexArray(k.shadow.vao);
      gl.drawElementsInstanced(gl.TRIANGLES, k.shadow.count, gl.UNSIGNED_SHORT, 0, k.shadow.n);
    });
    // reef life near the camera (only worth drawing from low down)
    var camW = camera.pos, reefR = camW[1] < 0 ? 42 : camW[1] < 40 ? 32 : 0;
    if (reefR > 0) this._updateReef(camW, O, reefR); else this.reef.forEach(function (k) { k.n = 0; });
    // the animals
    var fauna = this.fauna;
    if (this.life && reefR > 0) {
      this.life.update(dt * p.timeScale, camW);
      fauna[0].n = this.life.writeFish(fauna[0].inst, O, fauna[0].max);
      fauna[1].n = this.life.writeTurtles(fauna[1].inst, O);
      fauna[2].n = this.life.writeRays(fauna[2].inst, O);
      fauna.forEach(function (k) { if (k.n) { gl.bindBuffer(gl.ARRAY_BUFFER, k.buf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, k.inst, 0, k.n * 12); } });
    } else fauna.forEach(function (k) { k.n = 0; });
    var dyn = [], dynamic = this.dynamic;
    Object.keys(dynamic).forEach(function (name) {
      var k = dynamic[name];
      k.n = k.fill ? Math.min(k.fill(k.inst, O), k.max) : 0;
      if (k.n) { gl.bindBuffer(gl.ARRAY_BUFFER, k.buf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, k.inst, 0, k.n * 12); dyn.push(k); }
    });
    this.progThingShadow.use().set(common).set({ uViewProj: lightVP, uGrowR: reefR });
    this._drawThings(this.progThingShadow, this.reef);
    this._drawThings(this.progThingShadow, dyn, { uGrowR: 0 });
    this._drawThings(this.progThingShadow, fauna, { uGrowR: 0 });
    gl.disable(gl.DEPTH_TEST);
    // world -> shadow texture coordinates (bias matrix folded in)
    var bias = new Float32Array([0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0.5, 0.5, 0.5, 1]);
    common.uShadowMap = this.shadowMap; common.uShadowMat = mul4(bias, lightVP); common.uShadowOn = 1;

    // Scene: background, terrain, plants, then the water surface.
    this._target(this.sceneFbo, w, h);
    gl.clearDepth(1); gl.clear(gl.DEPTH_BUFFER_BIT);
    this.progBack.use().set(common).set({ uInvViewProj: inv4(vp), uSunDisk: sunDisk });
    this._fullscreen();
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS);
    gl.bindVertexArray(this.waterMesh.vao);
    this.progTerrain.use().set(common).set({ uViewProj: vp });
    gl.drawElements(gl.TRIANGLES, this.waterMesh.count, gl.UNSIGNED_INT, 0);
    this.progPlant.use().set(common).set(plantU).set({ uViewProj: vp });
    [this.palms, this.trees].forEach(function (k) {
      ["main", "far"].forEach(function (pass) {
        if (!k[pass].n) return;
        gl.bindVertexArray(k[pass].vao);
        gl.drawElementsInstanced(gl.TRIANGLES, k[pass].count, gl.UNSIGNED_SHORT, 0, k[pass].n);
      });
    });
    this.progThing.use().set(common).set({ uViewProj: vp, uGrowR: reefR });
    this._drawThings(this.progThing, this.reef);
    this._drawThings(this.progThing, fauna, { uGrowR: 0 });
    this._drawThings(this.progThing, dyn, { uGrowR: 0 });
    // copy the opaque scene so the water can look through itself at objects
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.sceneFbo);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.copyFbo);
    gl.blitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    this._target(this.sceneFbo, w, h);
    this.progWater.use().set(common).set({ uViewProj: vp, uFoamStr: p.foam > 0 ? 0.6 + p.foam : 0, uSceneCopy: this.sceneCopy, uInvRes: [1 / w, 1 / h] });
    gl.bindVertexArray(this.waterMesh.vao);
    gl.drawElements(gl.TRIANGLES, this.waterMesh.count, gl.UNSIGNED_INT, 0);
    gl.disable(gl.DEPTH_TEST);
    if (p.surfSim) this._drawLips(common, vp, w, h, camera);
    this._target(this.sceneFbo, w, h);
    gl.enable(gl.DEPTH_TEST);
    // spray over everything, depth-tested but not written
    var spray = this.spray;
    spray.n = 0;
    if (spray.fill) {
      spray.data = spray.data || new Float32Array(4000 * 8);
      spray.n = Math.min(spray.fill(spray.data, O), 4000);
    }
    if (spray.n) {
      gl.bindBuffer(gl.ARRAY_BUFFER, spray.buf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, spray.data, 0, spray.n * 8);
      var fw = norm3(camera.fwd), rt = norm3(cross3(fw, camera.up || [0, 1, 0])), upc = cross3(rt, fw);
      gl.depthMask(false);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      this.progSpray.use().set(common).set({ uViewProj: vp, uCamRight: rt, uCamUp: upc });
      gl.bindVertexArray(spray.vao);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, spray.n);
      gl.disable(gl.BLEND);
      gl.depthMask(true);
    }
    gl.disable(gl.DEPTH_TEST);

    // Bloom: bright-pass + downsample chain, then additive upsample.
    var b = this.bloom;
    if (p.bloom) {
      var src = this.sceneTex, sw = w, sh = h;
      for (var i = 0; i < b.length; i++) {
        this._target(b[i].fbo, b[i].w, b[i].h);
        // only the sun and its glints, not the bright sky around it
        this.progDown.use().set({ uSrc: src, uTexel: [1 / sw, 1 / sh], uThreshold: i === 0 ? (14 - 12.5 * night) / this.exposureNow : 0 });
        this._fullscreen();
        src = b[i].tex; sw = b[i].w; sh = b[i].h;
      }
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
      for (i = b.length - 2; i >= 0; i--) {
        this._target(b[i].fbo, b[i].w, b[i].h);
        this.progUp.use().set({ uSrc: b[i + 1].tex, uTexel: [1 / b[i + 1].w, 1 / b[i + 1].h] });
        this._fullscreen();
      }
      gl.disable(gl.BLEND);
    }

    this._target(null, this.canvas.width, this.canvas.height);
    // White balance for overhead sunlight, as a camera would: midday sand reads white.
    var wb = sunTransmittance([0, 1, 0]);
    this.progComposite.use().set({
      uScene: this.sceneTex, uBloom: b[0].tex, uExposure: this.exposureNow, uBloomStr: p.bloom ? 0.12 + 0.2 * night : 0, uNight: night,
      uWhite: [wb[1] / wb[0], 1, wb[1] / wb[2]],
    });
    this._fullscreen();
    gl.bindVertexArray(null);
  };

  // Read back one cascade's displacement field (debug / tests).
  Renderer.prototype.readDisplacement = function (c) {
    var gl = this.gl, N = FFT_N, out = new Float32Array(N * N * 4);
    var fbo = this._mrt([this.casc[c].disp[this.cur]]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.readPixels(0, 0, N, N, gl.RGBA, gl.FLOAT, out);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fbo);
    return out;
  };

  return {
    Renderer: Renderer,
    DEFAULTS: DEFAULTS,
    FFT_N: FFT_N, CASCADE_L: CASCADE_L, CASCADE_CUT: CASCADE_CUT,
    gaussians: gaussians, buildSpectrum: buildSpectrum, spectrumVariance: spectrumVariance,
    spectrumM0: spectrumM0, spectrumK: spectrumK, seaState: seaState, dispersion: dispersion,
    fftStockham: fftStockham,
  };
});
