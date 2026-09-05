"use strict";
/*
 * shots.js — headless screenshot of the sim at chosen times, rendered the
 * same way index.html renders (dots / velocity dots) straight into PNG files.
 * Uses only node built-ins (zlib for the PNG datastream).
 *
 *   node shots.js            -> writes _shot_t0.png, _shot_vel_2s.png,
 *                               _shot_settled_5s.png
 */
var zlib = require("zlib");
var fs = require("fs");
var Sim = require("./sim.js");

// ---------------------------------------------------------------- PNG writer
var CRC_TABLE = (function () {
  var t = new Int32Array(256);
  for (var n = 0; n < 256; n++) {
    var c = n;
    for (var k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    t[n] = c;
  }
  return t;
})();
function crc32(buf) {
  var c = 0xFFFFFFFF;
  for (var i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}
function chunk(type, data) {
  var len = Buffer.alloc(4); len.writeUInt32BE(data.length, 0);
  var body = Buffer.concat([Buffer.from(type, "ascii"), data]);
  var crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(body), 0);
  return Buffer.concat([len, body, crc]);
}
function writePNG(path, w, h, rgb) {
  // rgb: Buffer of w*h*3 bytes
  var raw = Buffer.alloc(h * (1 + w * 3));
  for (var y = 0; y < h; y++) {
    raw[y * (1 + w * 3)] = 0; // filter: none
    rgb.copy(raw, y * (1 + w * 3) + 1, y * w * 3, (y + 1) * w * 3);
  }
  var ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; ihdr[9] = 2; // 8-bit RGB
  var png = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
    chunk("IHDR", ihdr),
    chunk("IDAT", zlib.deflateSync(raw, { level: 6 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
  fs.writeFileSync(path, png);
}

// ------------------------------------------------------------------ drawing
var W = 800, H = 600;
function canvas() {
  var b = Buffer.alloc(W * H * 3);
  for (var i = 0; i < W * H; i++) { b[i * 3] = 0x0e; b[i * 3 + 1] = 0x14; b[i * 3 + 2] = 0x1b; }
  return b;
}
function dot(buf, x, y, r, cr, cg, cb) {
  var x0 = Math.max(0, Math.floor(x - r)), x1 = Math.min(W - 1, Math.ceil(x + r));
  var y0 = Math.max(0, Math.floor(y - r)), y1 = Math.min(H - 1, Math.ceil(y + r));
  for (var py = y0; py <= y1; py++) {
    for (var px = x0; px <= x1; px++) {
      var dx = px - x, dy = py - y;
      if (dx * dx + dy * dy <= r * r) {
        var o = (py * W + px) * 3;
        buf[o] = cr; buf[o + 1] = cg; buf[o + 2] = cb;
      }
    }
  }
}
function bucketColor(t) { // index.html velocity ramp
  var r, g, b;
  if (t < 0.5) { r = 0; g = 90 + 165 * (t / 0.5); b = 220; }
  else { var u = (t - 0.5) / 0.5; r = 255 * u; g = 255 * (1 - u) * 0.9 + 40; b = 220 * (1 - u); }
  return [r | 0, g | 0, b | 0];
}
function shot(sim, path, mode) {
  var buf = canvas();
  for (var i = 0; i < sim.N; i++) {
    var cr = 77, cg = 180, cb = 255;   // #4db4ff
    if (mode === 2) {
      var s = Math.sqrt(sim.vx[i] * sim.vx[i] + sim.vy[i] * sim.vy[i]);
      var t = Math.min(1, s / 700);
      var c = bucketColor(t); cr = c[0]; cg = c[1]; cb = c[2];
    }
    dot(buf, sim.px[i], sim.py[i], shotDotR, cr, cg, cb);
  }
  writePNG(path, W, H, buf);
  console.log("wrote " + path + " at t=" + sim.time.toFixed(2) + "s");
}

// ------------------------------------------------------------------- capture
// Optional arguments: `node shots.js [count] [spacing]` writes suffixed files.
// With spacing given, the sim is rescaled via setScale and the count follows
// to preserve fluid volume (N ~ 1/s^2), mirroring the UI's particle-size knob.
var N = +process.argv[2] || 0;
var SZ = +process.argv[3] || 0;
var suf = (N ? "_" + N : "") + (SZ ? "_size" + SZ : "");
var sim = new Sim(N ? { count: N } : {});
if (SZ) {
  sim.setScale(SZ);
  var nTarget = Math.max(250, Math.min(4000, Math.round(sim.N * 81 / (sim.spacing * sim.spacing))));
  sim.setCount(nTarget);
  sim.reset();
}
var shotDotR = Math.max(1.2, sim.spacing / 3);
shot(sim, "_shot_t0" + suf + ".png", 1);
function advance(seconds) {
  var frames = Math.round(seconds * 60);
  for (var f = 0; f < frames; f++) sim.step(1 / 60);
}
advance(2 - sim.time); shot(sim, "_shot_vel_2s" + suf + ".png", 2);
advance(5 - sim.time); shot(sim, "_shot_settled_5s" + suf + ".png", 1);
