/*
 * play.js — things you can do to the water, simulated on the CPU.
 *
 *   - Thrown objects (beach ball, coconut, rock): rigid spheres with
 *     buoyancy from the submerged cap volume, air and water drag, and the
 *     down-slope push of the waves, so they bob, drift and get carried in
 *     the surf. They roll on sand and splash when they land.
 *   - Spray particles: from splashes, the hand, the boat's bow, and the
 *     lips of breaking waves (found from the same shore-wave fields the GPU
 *     uses, so the spray leaves the crests you see).
 *   - A small motor boat: buoyancy, pitch and roll that follow the water
 *     under the hull, planing, a keel, steering that leans into turns, wave
 *     surfing, jumps off crests, and grounding on sand.
 *
 * Water heights come from the renderer's probes (one frame late): each
 * frame this module lists the points it wants (probePoints) and reads back
 * what the GPU measured there last frame (setProbes). Disturbances it makes
 * (splashes, wakes) go to the GPU ripple simulation as drops.
 *
 * Pure JS: the browser uses it as globalThis.Play; ocean-test.js runs it
 * against an analytic sea.
 */
(function (global, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else global.Play = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var G = 9.81, TAU = Math.PI * 2, RHO_W = 1025, RHO_A = 1.2;
  var MAX_OBJ = 16, MAX_PROBES = 64, MAX_PART = 4000;
  // probe slots: 0 camera (renderer), 1..48 objects (3 each), 49..56 boat
  var OBJ_SLOT = 1, BOAT_SLOT = 49;
  var KINDS = [
    { name: "beach ball", r: 0.32, density: 55, cd: 0.5, code: 20 },
    { name: "coconut", r: 0.12, density: 620, cd: 0.6, code: 21 },
    { name: "rock", r: 0.2, density: 2600, cd: 0.6, code: 22 },
  ];
  // Hull probe points in boat space (x forward, z starboard).
  var HULL = [[2.0, 0], [-2.3, 0], [0, -0.9], [0, 0.9], [1.2, -0.6], [1.2, 0.6], [-2.0, -0.8], [-2.0, 0.8]];

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function clamp(x, a, b) { return x < a ? a : x > b ? b : x; }
  function wrapPi(a) { a = (a + Math.PI) % TAU; if (a < 0) a += TAU; return a - Math.PI; }
  // Volume fraction of a sphere of radius r under water whose surface is s
  // above the sphere's bottom (spherical cap).
  function capFraction(s, r) { s = clamp(s, 0, 2 * r); return s * s * (3 * r - s) / (4 * r * r * r); }
  function quatMul(a, b) {
    return [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
      a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
  }
  function quatAxis(ax, ay, az, ang) { var s = Math.sin(ang / 2); return [ax * s, ay * s, az * s, Math.cos(ang / 2)]; }
  function quatNorm(q) { var l = Math.hypot(q[0], q[1], q[2], q[3]) || 1; return [q[0] / l, q[1] / l, q[2] / l, q[3] / l]; }

  // world: World.build() result. ground(x, z): sea-floor/land height.
  function Playground(world, ground) {
    this.w = world;
    this.ground = ground || function (x, z) { return world.heightAt(world, x, z); };
    this.rnd = mulberry32(31337);
    this.objects = [];
    this.probeH = new Float32Array(MAX_PROBES);     // last measured surface heights
    this.probeOK = false;
    this.drops = [];
    this.pPos = new Float32Array(MAX_PART * 3);
    this.pVel = new Float32Array(MAX_PART * 3);
    this.pLife = new Float32Array(MAX_PART);
    this.pMax = new Float32Array(MAX_PART);
    this.pSize = new Float32Array(MAX_PART);
    this.pN = 0;
    this.boat = null;
    this.events = [];          // splashes for the sound: {x, z, size}
  }

  // ---------------------------------------------------------------- probes
  Playground.prototype.probePoints = function () {
    var pts = [], i;
    for (i = 0; i < MAX_PROBES; i++) pts.push([0, 0]);
    this.objects.forEach(function (o, k) {
      var s = OBJ_SLOT + 3 * k;
      pts[s] = [o.pos[0], o.pos[2]]; pts[s + 1] = [o.pos[0] + 0.5, o.pos[2]]; pts[s + 2] = [o.pos[0], o.pos[2] + 0.5];
    });
    var b = this.boat;
    if (b) {
      var c = Math.cos(b.hdg), s2 = Math.sin(b.hdg);
      HULL.forEach(function (h, k) { pts[BOAT_SLOT + k] = [b.pos[0] + c * h[0] - s2 * h[1], b.pos[2] + s2 * h[0] + c * h[1]]; });
    }
    this._pts = pts;
    return pts;
  };
  Playground.prototype.setProbes = function (heights) {
    if (!heights) return;
    for (var i = 0; i < MAX_PROBES; i++) this.probeH[i] = heights[i];
    this.probeOK = true;
  };
  // Water height and slope from an object's three probes; before the first
  // readback, a flat sea.
  Playground.prototype._water = function (k) {
    if (!this.probeOK) return { h: 0, sx: 0, sz: 0 };
    var s = OBJ_SLOT + 3 * k, h = this.probeH[s];
    return { h: h, sx: (this.probeH[s + 1] - h) / 0.5, sz: (this.probeH[s + 2] - h) / 0.5 };
  };

  // ------------------------------------------------------------- objects
  Playground.prototype.throwObject = function (kind, pos, vel) {
    var K = KINDS[kind];
    if (this.objects.length >= MAX_OBJ) this.objects.shift();
    var a = this.rnd() * TAU;
    this.objects.push({
      kind: kind, K: K, pos: pos.slice(), vel: vel.slice(), q: quatAxis(0, 1, 0, a),
      w: [(this.rnd() - 0.5) * 8, (this.rnd() - 0.5) * 8, (this.rnd() - 0.5) * 8], wet: 0, seed: this.rnd(),
    });
  };
  Playground.prototype._updateObjects = function (dt) {
    var self = this;
    // probes follow the objects, so an object's slot shifts when an older one goes
    this.objects.forEach(function (o, k) {
      var K = o.K, r = K.r, wat = self._water(k);
      var g = self.ground(o.pos[0], o.pos[2]);
      var waterH = Math.max(wat.h, g - 2);   // over dry land the probe means nothing
      var f = g > 0.3 ? 0 : capFraction(waterH - (o.pos[1] - r), r);
      // splash on entry
      if (f > 0 && o.wet === 0 && o.vel[1] < -1.5) self.splash(o.pos[0], waterH, o.pos[2], r, -o.vel[1]);
      o.wet = f;
      var ax = 0, ay = -G, az = 0;
      ay += G * f * RHO_W / K.density;                       // buoyancy
      // drag: a = -(3/8) (rho/rho_obj) Cd |v| v / r, water on the wet part
      var sp = Math.hypot(o.vel[0], o.vel[1], o.vel[2]);
      var kd = 0.375 * K.cd / r * (RHO_A * (1 - f) + RHO_W * f) / K.density * sp;
      var damp = Math.exp(-Math.min(kd, 60) * dt);
      // The pressure gradient under a wave pushes a floating body down the
      // slope with g x slope, scaled by how much of its weight the water carries
      // (Froude-Krylov): a beach ball rides the swell as surely as a coconut.
      var carried = Math.min(1, f * RHO_W / K.density);
      if (f > 0) { ax -= G * wat.sx * 0.9 * carried; az -= G * wat.sz * 0.9 * carried; }
      o.vel[0] = (o.vel[0] + ax * dt) * damp;
      o.vel[1] = (o.vel[1] + ay * dt) * damp;
      o.vel[2] = (o.vel[2] + az * dt) * damp;
      // extra heave damping at the surface (radiated waves)
      if (f > 0 && f < 1) o.vel[1] *= Math.exp(-2.5 * dt);
      o.pos[0] += o.vel[0] * dt; o.pos[1] += o.vel[1] * dt; o.pos[2] += o.vel[2] * dt;
      // ground: bounce and roll
      g = self.ground(o.pos[0], o.pos[2]);
      if (o.pos[1] < g + r) {
        o.pos[1] = g + r;
        var e = 0.4 * (K.density < 100 ? 1.4 : 0.6);
        if (o.vel[1] < 0) o.vel[1] = -o.vel[1] * e;
        if (Math.abs(o.vel[1]) < 0.3) o.vel[1] = 0;
        var fr = Math.exp(-(K.density > 1000 ? 4 : 1.2) * dt);
        o.vel[0] *= fr; o.vel[2] *= fr;
      }
      // spin toward rolling with the motion when touching water or ground
      var touch = f > 0 || o.pos[1] < g + r + 0.02;
      if (touch) {
        var tw = [o.vel[2] / r, 0, -o.vel[0] / r], kk = Math.min(1, 5 * dt);
        o.w[0] += (tw[0] - o.w[0]) * kk; o.w[1] *= 1 - kk * 0.5; o.w[2] += (tw[2] - o.w[2]) * kk;
      }
      var wl = Math.hypot(o.w[0], o.w[1], o.w[2]);
      if (wl > 1e-5) o.q = quatNorm(quatMul(quatAxis(o.w[0] / wl, o.w[1] / wl, o.w[2] / wl, wl * dt), o.q));
    });
    // lost ones (far away or sunk deep) go
    this.objects = this.objects.filter(function (o) { return o.pos[1] > -60; });
  };
  Playground.prototype.writeObjects = function (kind, out, O) {
    var n = 0;
    this.objects.forEach(function (o) {
      if (o.kind !== kind) return;
      var q = n * 12;
      out[q] = o.pos[0] - O[0]; out[q + 1] = o.pos[1]; out[q + 2] = o.pos[2] - O[1]; out[q + 3] = o.K.r;
      out[q + 4] = o.q[0]; out[q + 5] = o.q[1]; out[q + 6] = o.q[2]; out[q + 7] = o.q[3];
      out[q + 8] = o.seed; out[q + 9] = 0; out[q + 10] = 0; out[q + 11] = 0;
      n++;
    });
    return n;
  };

  // ----------------------------------------------------------- disturbances
  // A ripple drop: velocity kick dv (m/s, negative pushes the water down)
  // with gaussian radius r, plus foam.
  Playground.prototype.drop = function (x, z, r, dv, foam) {
    if (this.drops.length < 24) this.drops.push([x, z, r, dv, foam || 0]);
  };
  Playground.prototype.takeDrops = function () { var d = this.drops; this.drops = []; return d; };
  Playground.prototype.splash = function (x, y, z, r, speed) {
    var k = Math.min(speed * r, 3);
    this.drop(x, z, Math.max(r * 1.6, 0.3), -Math.min(0.6 + speed * r * 0.9, 3.5), Math.min(0.4 + k, 2.5));
    var n = Math.round(10 + 50 * k);
    for (var i = 0; i < n; i++) {
      var a = this.rnd() * TAU, s = (0.6 + this.rnd()) * (1 + speed * 0.25), up = (1.5 + this.rnd() * 3) * (0.6 + k * 0.5);
      this.emit(x + Math.cos(a) * r, y + 0.05, z + Math.sin(a) * r, Math.cos(a) * s, up, Math.sin(a) * s, 0.06 + this.rnd() * 0.12 * (1 + k), 0.6 + this.rnd() * 0.8);
    }
    this.events.push({ x: x, z: z, size: k });
  };

  // --------------------------------------------------------------- spray
  Playground.prototype.emit = function (x, y, z, vx, vy, vz, size, life) {
    var i = this.pN < MAX_PART ? this.pN++ : (this.rnd() * MAX_PART) | 0;
    this.pPos[i * 3] = x; this.pPos[i * 3 + 1] = y; this.pPos[i * 3 + 2] = z;
    this.pVel[i * 3] = vx; this.pVel[i * 3 + 1] = vy; this.pVel[i * 3 + 2] = vz;
    this.pLife[i] = life; this.pMax[i] = life; this.pSize[i] = size;
  };
  Playground.prototype._updateSpray = function (dt) {
    var P = this.pPos, V = this.pVel, L = this.pLife, n = this.pN;
    for (var i = 0; i < n; i++) {
      L[i] -= dt;
      var o = i * 3;
      V[o + 1] -= G * dt;
      var d = Math.exp(-(0.4 + this.pSize[i] * 4) * dt);       // fine mist slows fastest
      V[o] *= d; V[o + 1] *= d; V[o + 2] *= d;
      P[o] += V[o] * dt; P[o + 1] += V[o + 1] * dt; P[o + 2] += V[o + 2] * dt;
      this.pSize[i] *= 1 + 0.35 * dt;                           // drops spread into mist
      if (L[i] <= 0 || (P[o + 1] < -0.15 && V[o + 1] < 0)) {
        // swap-remove
        n--;
        if (i !== n) {
          P[o] = P[n * 3]; P[o + 1] = P[n * 3 + 1]; P[o + 2] = P[n * 3 + 2];
          V[o] = V[n * 3]; V[o + 1] = V[n * 3 + 1]; V[o + 2] = V[n * 3 + 2];
          L[i] = L[n]; this.pMax[i] = this.pMax[n]; this.pSize[i] = this.pSize[n];
          i--;
        }
      }
    }
    this.pN = n;
  };
  // Per particle: (x, y, z, size), (vx, vy, vz, alpha).
  Playground.prototype.writeSpray = function (out, O) {
    var n = this.pN;
    for (var i = 0; i < n; i++) {
      var q = i * 8, o = i * 3, t = this.pLife[i] / this.pMax[i];
      out[q] = this.pPos[o] - O[0]; out[q + 1] = this.pPos[o + 1]; out[q + 2] = this.pPos[o + 2] - O[1]; out[q + 3] = this.pSize[i];
      out[q + 4] = this.pVel[o]; out[q + 5] = this.pVel[o + 1]; out[q + 6] = this.pVel[o + 2];
      out[q + 7] = Math.min(1, t * 2.5) * Math.min(1, (1 - t) * 12 + 0.2);
    }
    return n;
  };
  // Spray thrown off the lips of breaking waves near the camera. Mirrors the
  // GPU shore wave: phase th = w t - w A (A = arrival time), height scaled by
  // the wave-group envelope; the lip leaves at th ~ 0 while the stage is
  // plunging.
  Playground.prototype._crestSpray = function (dt, time, cam, surf) {
    var w = this.w;
    if (!w.arrival || surf <= 0) return;
    var T = w.SWELL_T, om = TAU / T, og = om / w.GROUP, samples = Math.round(2400 * Math.min(dt * 60, 3));
    for (var s = 0; s < samples; s++) {
      var a = this.rnd() * TAU, r = 6 + 74 * Math.sqrt(this.rnd());
      var x = cam[0] + Math.cos(a) * r, z = cam[2] + Math.sin(a) * r;
      var st = w.bilinear(w.stage, x, z, 0);
      if (st < 0.92 || st > 1.5) continue;
      var A = w.bilinear(w.arrival, x, z, 0), th = wrapPi(om * (time - A));
      if (th < -0.1 || th > 0.06) continue;
      var H = w.bilinear(w.waveH, x, z, 0) * surf * (0.72 + 0.28 * Math.cos(og * (time - A)));
      if (H < 0.3) continue;
      var nx = w.bilinear(w.dirX, x, z, 0), nz = w.bilinear(w.dirZ, x, z, 0), nl = Math.hypot(nx, nz) || 1;
      nx /= nl; nz /= nl;
      var plunge = clamp((st - 0.9) * 4, 0, 1) * clamp((1.5 - st) * 3, 0, 1);
      for (var k = 0; k < 3; k++) {
        var c = Math.sqrt(G * Math.max(H, 0.3) * 1.6), fw = c * (0.3 + 0.4 * this.rnd()), up = (1 + 2.5 * this.rnd()) * Math.sqrt(H) * (0.5 + plunge);
        this.emit(x + nx * H * 0.3, H * 0.6, z + nz * H * 0.3, nx * fw + (this.rnd() - 0.5), up, nz * fw + (this.rnd() - 0.5), 0.1 + 0.25 * this.rnd() * H, 0.7 + 0.9 * this.rnd());
      }
    }
  };

  // ---------------------------------------------------------------- boat
  Playground.prototype.spawnBoat = function (x, z, hdg) {
    this.boat = { pos: [x, 0, z], hdg: hdg, pitch: 0, roll: 0, vel: [0, 0, 0], yawRate: 0, pitchV: 0, rollV: 0, throttle: 0, steer: 0, air: 0, grounded: false };
    return this.boat;
  };
  Playground.prototype.removeBoat = function () { this.boat = null; };
  Playground.prototype._updateBoat = function (dt) {
    var b = this.boat, self = this;
    if (!b) return;
    var H = [];
    for (var k = 0; k < HULL.length; k++) H.push(this.probeOK ? this.probeH[BOAT_SLOT + k] : 0);
    var c = Math.cos(b.hdg), s = Math.sin(b.hdg), fwd = [c, s], side = [-s, c];
    var gC = this.ground(b.pos[0], b.pos[2]), gB = this.ground(b.pos[0] + c * 2.2, b.pos[2] + s * 2.2);
    for (k = 0; k < H.length; k++) H[k] = Math.max(H[k], Math.min(gC, gB) - 3);
    var hBow = (H[0] + H[4] + H[5]) / 3, hStern = (H[1] + H[6] + H[7]) / 3, hPort = (H[2] + H[4] + H[6]) / 3, hStar = (H[3] + H[5] + H[7]) / 3;
    var hMean = (hBow + hStern) / 2;
    var speedF = b.vel[0] * c + b.vel[2] * s, speedS = b.vel[0] * side[0] + b.vel[2] * side[1];
    var draft = 0.28, d = hMean - b.pos[1] + draft, wet = clamp(d / draft, 0, 1);
    b.air = wet <= 0 ? b.air + dt : 0;
    // heave: buoyancy spring toward floating at the water line, plus planing lift
    var lift = Math.min(speedF * speedF * 0.004, 0.25);
    var ay = d > 0 ? G * Math.min(d, 0.9) / draft - G + lift * G : -G;
    b.vel[1] = (b.vel[1] + ay * dt) * Math.exp(-(d > 0 ? 3.2 : 0.05) * dt);
    // pitch and roll follow the water under the hull; planing lifts the bow
    var plane = clamp(speedF / 7, 0, 1), planePitch = 0.11 * plane * (1 - clamp((speedF - 9) / 8, 0, 0.6));
    var tPitch = Math.atan2(hBow - hStern, 4.3) + planePitch - b.throttle * 0.02;
    var tRoll = Math.atan2(hPort - hStar, 1.8) - b.steer * clamp(speedF, 0, 12) * 0.018;
    if (wet > 0) {
      b.pitchV += ((tPitch - b.pitch) * 22 - b.pitchV * 6) * dt * wet;
      b.rollV += ((tRoll - b.roll) * 16 - b.rollV * 5) * dt * wet;
    } else b.pitchV -= 0.35 * dt;                                // nose drops in the air
    b.pitch += b.pitchV * dt; b.roll += b.rollV * dt;
    // thrust needs the propeller in the water; keel kills sideways motion
    var prop = this.probeOK ? clamp((H[1] - (b.pos[1] - Math.sin(b.pitch) * 2.4) + 0.5) / 0.4, 0, 1) : 1;
    var thrust = b.throttle * (b.throttle > 0 ? 7.5 : 3.2) * prop;
    var drag = speedF * (0.12 + 0.028 * Math.abs(speedF)) * (wet > 0 ? 1 : 0.05);
    var aF = thrust - drag, aS = -speedS * (wet > 0 ? 3.5 : 0.1);
    // surf: the slope of the water under the hull pushes the boat down it
    var slF = (hBow - hStern) / 4.3, slS = (hStar - hPort) / 1.8;
    aF -= G * slF * 0.55 * wet; aS -= G * slS * 0.55 * wet;
    b.vel[0] += (fwd[0] * aF + side[0] * aS) * dt;
    b.vel[2] += (fwd[1] * aF + side[1] * aS) * dt;
    // steering: rudder authority grows with speed; reversing flips it
    var dir = speedF < -0.3 ? -1 : 1;
    var tYaw = b.steer * clamp(0.25 + Math.abs(speedF) * 0.09, 0, 0.85) * dir * (wet > 0 ? 1 : 0.2);
    b.yawRate += (tYaw - b.yawRate) * Math.min(1, 3 * dt);
    b.hdg += b.yawRate * dt;
    b.pos[0] += b.vel[0] * dt; b.pos[1] += b.vel[1] * dt; b.pos[2] += b.vel[2] * dt;
    // the keel on the sand: friction, and the hull rides up the beach
    c = Math.cos(b.hdg); s = Math.sin(b.hdg);
    var gk = Math.max(this.ground(b.pos[0], b.pos[2]), this.ground(b.pos[0] + c * 1.8, b.pos[2] + s * 1.8) - 0.15);
    b.grounded = false;
    if (b.pos[1] - 0.45 < gk) {
      b.grounded = true;
      b.pos[1] = gk + 0.45;
      if (b.vel[1] < 0) b.vel[1] = 0;
      var fr = Math.exp(-3.5 * dt);
      b.vel[0] *= fr; b.vel[2] *= fr;
    }
    // too steep to climb (rock, land): stop dead against it
    if (this.ground(b.pos[0] + c * 2.6, b.pos[2] + s * 2.6) > b.pos[1] + 0.5) {
      var vF = b.vel[0] * c + b.vel[2] * s;
      if (vF > 0) { b.vel[0] -= c * vF * 1.2; b.vel[2] -= s * vF * 1.2; }
    }
    // wake: the hull pushes water down along its length; foam at the stern
    var sp = Math.hypot(b.vel[0], b.vel[2]);
    if (wet > 0 && sp > 0.3) {
      // (rates per second, so the wake doesn't depend on the frame rate)
      for (k = 0; k < 4; k++) {
        var t = 2.0 - k * 1.3, kick = -Math.min(sp, 14) * 0.7 * dt * (k === 0 ? 1.4 : 1);
        this.drop(b.pos[0] + c * t, b.pos[2] + s * t, 0.9, kick, k === 3 ? Math.min(sp * 0.5, 5) * dt : 0);
      }
      // churned water behind the propeller
      if (Math.abs(b.throttle) > 0.05) this.drop(b.pos[0] - c * 3.0, b.pos[2] - s * 3.0, 0.6, 0, 6 * Math.abs(b.throttle) * dt);
    }
    // bow spray: sheets off both sides at speed, more when the bow slams down
    if (wet > 0 && speedF > 3.5) {
      var rate = (speedF - 3.5) * 9 + Math.max(-b.pitchV, 0) * 120;
      var count = Math.min(Math.floor(rate * dt + this.rnd()), 40);
      for (k = 0; k < count; k++) {
        var sd = this.rnd() < 0.5 ? -1 : 1, along = 0.6 + this.rnd() * 1.0;
        var px = b.pos[0] + c * along - s * sd * 0.85, pz = b.pos[2] + s * along + c * sd * 0.85;
        var out = 1.5 + this.rnd() * 2.5 + speedF * 0.12;
        this.emit(px, b.pos[1] - 0.05, pz, c * speedF * 0.35 - s * sd * out, 1.2 + this.rnd() * 2.2 + speedF * 0.08, s * speedF * 0.35 + c * sd * out, 0.05 + this.rnd() * 0.1, 0.5 + this.rnd() * 0.5);
      }
    }
    // landing after a jump
    if (b.air === 0 && b.vel[1] < -2.5 && wet > 0) { this.splash(b.pos[0], hMean, b.pos[2], 1.0, -b.vel[1]); }
    void self;
  };
  // Quaternion: yaw (about y, heading hdg toward +z for positive), pitch (bow up), roll.
  Playground.prototype.boatQuat = function () {
    var b = this.boat;
    var qy = quatAxis(0, 1, 0, -b.hdg), qp = quatAxis(0, 0, 1, b.pitch), qr = quatAxis(1, 0, 0, b.roll);
    return quatNorm(quatMul(qy, quatMul(qp, qr)));
  };
  Playground.prototype.writeBoat = function (out, O) {
    var b = this.boat;
    if (!b) return 0;
    var q = this.boatQuat();
    out[0] = b.pos[0] - O[0]; out[1] = b.pos[1]; out[2] = b.pos[2] - O[1]; out[3] = 1;
    out[4] = q[0]; out[5] = q[1]; out[6] = q[2]; out[7] = q[3];
    out[8] = 0; out[9] = 0; out[10] = b.throttle; out[11] = 0;
    return 1;
  };

  // ----------------------------------------------------------------- step
  Playground.prototype.update = function (dt, time, cam, surf) {
    dt = Math.min(dt, 0.05);
    if (dt <= 0) return;
    this._updateObjects(dt);
    this._updateBoat(dt);
    this._crestSpray(dt, time, cam, surf);
    this._updateSpray(dt);
  };

  // ---------------------------------------------------------------- meshes
  // Flat-shaded triangle soup in the shared vertex format (position, normal,
  // uv, part).
  function Soup() { this.V = []; }
  Soup.prototype.tri = function (a, b, c, part, uv) {
    var e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    var n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]], l = Math.hypot(n[0], n[1], n[2]) || 1;
    n = [n[0] / l, n[1] / l, n[2] / l];
    var V = this.V;
    [a, b, c].forEach(function (p, i) { V.push(p[0], p[1], p[2], n[0], n[1], n[2], uv ? uv[i * 2] : 0, uv ? uv[i * 2 + 1] : 0, part); });
  };
  Soup.prototype.quad = function (a, b, c, d, part) { this.tri(a, b, c, part); this.tri(a, c, d, part); };
  Soup.prototype.box = function (x0, x1, y0, y1, z0, z1, part) {
    var p = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
    this.quad(p[0], p[3], p[2], p[1], part); this.quad(p[4], p[5], p[6], p[7], part);
    this.quad(p[0], p[4], p[7], p[3], part); this.quad(p[1], p[2], p[6], p[5], part);
    this.quad(p[3], p[7], p[6], p[2], part); this.quad(p[0], p[1], p[5], p[4], part);
  };
  Soup.prototype.done = function () {
    var n = this.V.length / 9, idx = new Uint16Array(n);
    for (var i = 0; i < n; i++) idx[i] = i;
    return { vertices: new Float32Array(this.V), indices: idx };
  };
  // Unit sphere with smooth normals (uv = longitude, latitude).
  function sphereMesh(nLat, nLon, squash) {
    var V = [], I = [];
    for (var i = 0; i <= nLat; i++) for (var j = 0; j <= nLon; j++) {
      var th = i / nLat * Math.PI, ph = j / nLon * TAU, n = [Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph)];
      V.push(n[0], n[1] * (squash || 1), n[2], n[0], n[1] / (squash || 1), n[2], j / nLon, i / nLat, 0);
    }
    for (i = 0; i < nLat; i++) for (j = 0; j < nLon; j++) { var a = i * (nLon + 1) + j; I.push(a, a + nLon + 1, a + 1, a + 1, a + nLon + 1, a + nLon + 2); }
    return { vertices: new Float32Array(V), indices: new Uint16Array(I) };
  }
  // A 5.2 m centre-console skiff: hard-chine deep-V hull, bow at +x, water
  // line at y = 0. Parts: 0 hull, 2 deck, 3 white fittings, 4 motor, 5 glass.
  function boatMesh() {
    var S = new Soup(), st = [], NST = 14, i;
    for (i = 0; i <= NST; i++) {
      var x = -2.6 + 5.2 * i / NST, t = (x + 2.6) / 5.2;
      var bow = Math.max(0, (x - 0.6) / 2.0);
      var half = 1.0 * Math.sqrt(Math.max(0, 1 - Math.pow(bow, 2.2))) * (1 - 0.12 * (1 - t) * (1 - t));
      var keel = -0.42 + 0.75 * Math.pow(Math.max(0, (x - 1.2) / 1.4), 1.8), chineY = -0.12 + 0.55 * Math.pow(bow, 2), gun = 0.55 + 0.12 * Math.pow(bow, 1.5);
      if (i === NST) half = 0.001;
      st.push({ x: x, half: half, keel: Math.min(keel, gun - 0.05), chine: Math.min(chineY, gun - 0.05), gun: gun });
    }
    function P(s, k, side) {   // k: 0 keel, 1 chine, 2 gunwale
      if (k === 0) return [s.x, s.keel, 0];
      if (k === 1) return [s.x, s.chine, side * s.half * 0.82];
      return [s.x, s.gun, side * s.half];
    }
    for (i = 0; i < NST; i++) {
      var a = st[i], b = st[i + 1];
      [1, -1].forEach(function (sd) {
        // bottom (keel -> chine) and topsides (chine -> gunwale), outward-facing
        if (sd > 0) { S.quad(P(a, 0, sd), P(b, 0, sd), P(b, 1, sd), P(a, 1, sd), 0); S.quad(P(a, 1, sd), P(b, 1, sd), P(b, 2, sd), P(a, 2, sd), 0); }
        else { S.quad(P(a, 0, sd), P(a, 1, sd), P(b, 1, sd), P(b, 0, sd), 0); S.quad(P(a, 1, sd), P(a, 2, sd), P(b, 2, sd), P(b, 1, sd), 0); }
        // inner wall from deck to gunwale, and the gunwale cap
        var deck = 0.12, inA = [a.x, deck, sd * a.half * 0.9], inB = [b.x, deck, sd * b.half * 0.9];
        var tA = [a.x, a.gun, sd * a.half * 0.9], tB = [b.x, b.gun, sd * b.half * 0.9];
        if (sd > 0) { S.quad(inA, tA, tB, inB, 3); S.quad(tA, P(a, 2, sd), P(b, 2, sd), tB, 3); }
        else { S.quad(inA, inB, tB, tA, 3); S.quad(tA, tB, P(b, 2, sd), P(a, 2, sd), 3); }
      });
      // deck
      S.quad([a.x, 0.12, -a.half * 0.9], [a.x, 0.12, a.half * 0.9], [b.x, 0.12, b.half * 0.9], [b.x, 0.12, -b.half * 0.9], 2);
    }
    // transom
    var s0 = st[0];
    [1, -1].forEach(function (sd) {
      var k0 = P(s0, 0, sd), c0 = P(s0, 1, sd), g0 = P(s0, 2, sd), mid = [s0.x, s0.gun, 0];
      if (sd > 0) { S.tri(k0, mid, c0, 0); S.tri(c0, mid, g0, 0); } else { S.tri(k0, c0, mid, 0); S.tri(c0, g0, mid, 0); }
    });
    // centre console with windscreen, a bench seat, the outboard
    S.box(0.2, 0.95, 0.12, 1.0, -0.36, 0.36, 3);
    S.quad([0.95, 1.0, -0.36], [0.95, 1.0, 0.36], [0.75, 1.32, 0.36], [0.75, 1.32, -0.36], 5);
    S.box(-0.75, -0.2, 0.12, 0.52, -0.55, 0.55, 3);
    S.box(-2.95, -2.62, 0.25, 0.95, -0.2, 0.2, 4);
    S.box(-2.85, -2.7, -0.55, 0.25, -0.06, 0.06, 4);
    return S.done();
  }

  return {
    Playground: Playground, KINDS: KINDS, MAX_PROBES: MAX_PROBES, MAX_PART: MAX_PART, MAX_OBJ: MAX_OBJ,
    capFraction: capFraction, sphereMesh: sphereMesh, boatMesh: boatMesh,
  };
});
