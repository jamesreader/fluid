/*
 * life.js — the animals of the reef, simulated on the CPU.
 *
 * Fish swim in schools (boids: cohesion toward a target that wanders around
 * the school's home coral head, alignment, separation), hug the reef without
 * touching it, stay under the surface, and scatter when the camera swims
 * into them. Green turtles cruise the lagoon and come up to breathe; spotted
 * eagle rays glide low over the sand. Everything lives near the camera:
 * animals left far behind respawn out of sight at reef spots nearby.
 *
 * Pure JS: the browser uses it as globalThis.Life, ocean-test.js requires it.
 * Instances are written for ocean.js's reef-life shader: position + scale,
 * rotation quaternion, (seed, species, swim rate, phase).
 */
(function (global, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else global.Life = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var TAU = Math.PI * 2;
  // name, fish per school, length (m), cruise speed (m/s), body-wave rate
  var SPECIES = [
    { name: "yellow tang", n: 22, len: 0.24, speed: 0.55, wag: 9 },
    { name: "blue chromis", n: 46, len: 0.12, speed: 0.5, wag: 13 },
    { name: "sergeant major", n: 24, len: 0.18, speed: 0.5, wag: 11 },
    { name: "parrotfish", n: 7, len: 0.55, speed: 0.45, wag: 6 },
    { name: "butterflyfish", n: 10, len: 0.17, speed: 0.4, wag: 10 },
    { name: "bluefin jack", n: 34, len: 0.45, speed: 1.1, wag: 7 },
  ];
  var SCHOOLS = [0, 1, 1, 2, 3, 4, 5, 0, 2];
  var RANGE = 75, RESPAWN = 105;

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // Quaternion turning +x toward (dx, dy, dz): yaw about y, then pitch.
  function headingQuat(dx, dy, dz, out, o) {
    var h = Math.sqrt(dx * dx + dz * dz) || 1e-6, yaw = Math.atan2(-dz, dx), pitch = Math.atan2(dy, h);
    var cy = Math.cos(yaw / 2), sy = Math.sin(yaw / 2), cp = Math.cos(pitch / 2), sp = Math.sin(pitch / 2);
    // q = qy(yaw) * qz(pitch)
    out[o] = -sy * sp; out[o + 1] = sy * cp; out[o + 2] = cy * sp; out[o + 3] = cy * cp;
  }

  // ground(x, z) -> sea-floor height; defaults to the world heightfield (the
  // renderer passes one that includes its coral-rock relief).
  function Ecosystem(world, seed, ground) {
    this.w = world;
    this.ground = ground || function (x, z) { return world.heightAt(world, x, z); };
    this.rnd = mulberry32(seed || 2024);
    this.time = 0;
    var self = this;
    this.schools = SCHOOLS.map(function (sp) {
      var S = SPECIES[sp], n = S.n;
      return { sp: sp, S: S, n: n, pos: new Float32Array(n * 3), vel: new Float32Array(n * 3), seed: new Float32Array(n), home: [0, 0, 0], target: [0, 0, 0], alive: false, wander: self.rnd() * TAU };
    });
    this.turtles = [0, 1, 2].map(function () { return { pos: [0, 0, 0], yaw: 0, pitch: 0, speed: 0.45, alive: false, breathe: 20 + self.rnd() * 40, goal: [0, 0, 0], seed: self.rnd() }; });
    this.rays = [0, 1, 2].map(function () { return { pos: [0, 0, 0], yaw: 0, pitch: 0, speed: 0.7, alive: false, goal: [0, 0, 0], seed: self.rnd() }; });
  }

  // A spot under water near (cx, cz): depth in [dMin, dMax] m, preferring
  // coral when wantCoral. Tries random points at radius r0..r1.
  Ecosystem.prototype.spot = function (cx, cz, r0, r1, dMin, dMax, wantCoral) {
    var w = this.w, best = null, bestScore = -1;
    for (var t = 0; t < 40; t++) {
      var a = this.rnd() * TAU, r = r0 + (r1 - r0) * Math.sqrt(this.rnd());
      var x = cx + Math.cos(a) * r, z = cz + Math.sin(a) * r, g = w.heightAt(w, x, z);
      if (-g < dMin || -g > dMax) continue;
      var score = wantCoral ? w.bilinear(w.coral, x, z, 0) + 0.05 * this.rnd() : this.rnd();
      if (score > bestScore) { bestScore = score; best = [x, g, z]; }
    }
    return best;
  };
  Ecosystem.prototype._spawnSchool = function (sc, cam, near) {
    var p = this.spot(cam[0], cam[2], near ? 8 : 45, near ? 45 : 70, 1.6, 22, true);
    if (!p) { sc.alive = false; return; }
    var S = sc.S, ground = p[1];
    sc.home = [p[0], ground + 0.6 + S.len * 2, p[2]];
    sc.target = sc.home.slice();
    for (var i = 0; i < sc.n; i++) {
      var a = this.rnd() * TAU, r = Math.sqrt(this.rnd()) * (0.5 + sc.n * 0.04);
      sc.pos[i * 3] = sc.home[0] + Math.cos(a) * r;
      sc.pos[i * 3 + 1] = Math.min(sc.home[1] + (this.rnd() - 0.5) * 0.8, -0.6);
      sc.pos[i * 3 + 2] = sc.home[2] + Math.sin(a) * r;
      sc.vel[i * 3] = Math.cos(sc.wander) * S.speed; sc.vel[i * 3 + 1] = 0; sc.vel[i * 3 + 2] = Math.sin(sc.wander) * S.speed;
      sc.seed[i] = this.rnd();
    }
    sc.alive = true;
  };

  // Advance everything by dt seconds around the camera at world position cam.
  Ecosystem.prototype.update = function (dt, cam) {
    dt = Math.min(dt, 0.1);
    if (dt <= 0) return;
    this.time += dt;
    var w = this.w, self = this, i, j;
    this.schools.forEach(function (sc, si) {
      var dx = sc.home[0] - cam[0], dz = sc.home[2] - cam[2];
      if (!sc.alive || dx * dx + dz * dz > RESPAWN * RESPAWN) self._spawnSchool(sc, cam, !sc.everSpawned);
      sc.everSpawned = true;
      if (!sc.alive) return;
      var S = sc.S, n = sc.n, P = sc.pos, V = sc.vel;
      // the school's target loops slowly around its home coral head
      sc.wander += dt * (0.05 + 0.02 * (si % 3));
      var rad = 3 + 0.05 * n;
      sc.target[0] = sc.home[0] + Math.cos(sc.wander) * rad;
      sc.target[2] = sc.home[2] + Math.sin(sc.wander * 1.3) * rad;
      var tg = self.ground(sc.target[0], sc.target[2]);
      sc.target[1] = Math.min(Math.max(tg + 0.5 + S.len * 2.5, sc.home[1] - 1.5), -0.8);
      var ax = 0, ay = 0, az = 0;
      for (i = 0; i < n; i++) { ax += V[i * 3]; ay += V[i * 3 + 1]; az += V[i * 3 + 2]; }
      ax /= n; ay /= n; az /= n;
      var sep = S.len * 2.2, sep2 = sep * sep;
      for (i = 0; i < n; i++) {
        var o = i * 3, px = P[o], py = P[o + 1], pz = P[o + 2];
        var fx = (sc.target[0] - px) * 0.35, fy = (sc.target[1] - py) * 0.5, fz = (sc.target[2] - pz) * 0.35;
        fx += (ax - V[o]) * 0.8; fy += (ay - V[o + 1]) * 0.8; fz += (az - V[o + 2]) * 0.8;
        for (j = 0; j < n; j++) {
          if (j === i) continue;
          var ex = px - P[j * 3], ey = py - P[j * 3 + 1], ez = pz - P[j * 3 + 2], d2 = ex * ex + ey * ey + ez * ez;
          if (d2 < sep2 && d2 > 1e-8) { var k = (sep2 - d2) / (sep2 * Math.sqrt(d2)) * 3.0; fx += ex * k; fy += ey * k; fz += ez * k; }
        }
        // reef below, surface above
        var g = self.ground(px, pz), clear = py - g;
        if (clear < 0.35 + S.len) fy += (0.35 + S.len - clear) * 8;
        if (py > -0.45) fy -= (py + 0.45) * 8;
        // the diver: scatter if the camera comes close
        var cx = px - cam[0], cy = py - cam[1], cz = pz - cam[2], cd2 = cx * cx + cy * cy + cz * cz, flee = 0;
        if (cd2 < 16 && cd2 > 1e-6) { var cd = Math.sqrt(cd2); flee = (4 - cd) / 4; var fk = flee * 14 / cd; fx += cx * fk; fy += cy * fk * 0.5; fz += cz * fk; }
        var vx = V[o] + fx * dt, vy = V[o + 1] + fy * dt, vz = V[o + 2] + fz * dt;
        var sp = Math.sqrt(vx * vx + vy * vy + vz * vz) || 1e-6, maxS = S.speed * (1.6 + flee * 4), minS = S.speed * 0.4;
        var cl = sp > maxS ? maxS / sp : sp < minS ? minS / sp : 1;
        vx *= cl; vy *= cl * 0.6; vz *= cl;
        V[o] = vx; V[o + 1] = vy; V[o + 2] = vz;
        P[o] = px + vx * dt; P[o + 1] = Math.min(py + vy * dt, -0.3); P[o + 2] = pz + vz * dt;
        if (P[o + 1] < g + 0.08) P[o + 1] = g + 0.08;
      }
    });
    // turtles: cruise between waypoints in the lagoon, surface now and then to breathe
    this.turtles.forEach(function (t) {
      var dx = t.pos[0] - cam[0], dz = t.pos[2] - cam[2];
      if (!t.alive || dx * dx + dz * dz > RESPAWN * RESPAWN) {
        var p = self.spot(cam[0], cam[2], t.alive ? 50 : 12, t.alive ? 75 : 50, 2.5, 18, false);
        if (!p) return;
        t.pos = [p[0], p[1] + 1.5, p[2]]; t.yaw = self.rnd() * TAU; t.goal = p; t.alive = true;
      }
      var gdx = t.goal[0] - t.pos[0], gdz = t.goal[2] - t.pos[2];
      if (gdx * gdx + gdz * gdz < 9) t.goal = self.spot(t.pos[0], t.pos[2], 10, 40, 2.5, 18, false) || t.goal;
      steer(t, t.goal, 0.25, dt);
      t.breathe -= dt;
      var g = self.ground(t.pos[0], t.pos[2]);
      var wantY = t.breathe < 0 ? -0.15 : Math.min(g + 1.2 + 1.5 * Math.sin(self.time * 0.1 + t.seed * 9), -1.2);
      if (t.breathe < -6) t.breathe = 30 + self.rnd() * 40;
      t.pitch += ((wantY - t.pos[1]) * 0.25 - t.pitch) * Math.min(1, dt);
      t.pitch = Math.max(-0.5, Math.min(0.5, t.pitch));
      move(t, dt, g + 0.35);
    });
    // rays: glide low over the sand
    this.rays.forEach(function (r) {
      var dx = r.pos[0] - cam[0], dz = r.pos[2] - cam[2];
      if (!r.alive || dx * dx + dz * dz > RESPAWN * RESPAWN) {
        var p = self.spot(cam[0], cam[2], r.alive ? 50 : 10, r.alive ? 75 : 45, 2.0, 14, false);
        if (!p) return;
        r.pos = [p[0], p[1] + 0.8, p[2]]; r.yaw = self.rnd() * TAU; r.goal = p; r.alive = true;
      }
      var gdx = r.goal[0] - r.pos[0], gdz = r.goal[2] - r.pos[2];
      if (gdx * gdx + gdz * gdz < 9) r.goal = self.spot(r.pos[0], r.pos[2], 12, 45, 2.0, 14, false) || r.goal;
      steer(r, r.goal, 0.3, dt);
      var g = self.ground(r.pos[0], r.pos[2]);
      r.pitch += (((g + 0.9) - r.pos[1]) * 0.3 - r.pitch) * Math.min(1, dt);
      r.pitch = Math.max(-0.3, Math.min(0.3, r.pitch));
      move(r, dt, g + 0.4);
    });
  };
  function steer(a, goal, rate, dt) {
    var want = Math.atan2(goal[2] - a.pos[2], goal[0] - a.pos[0]);
    var d = ((want - a.yaw) % TAU + TAU * 1.5) % TAU - Math.PI;
    a.yaw += Math.max(-rate, Math.min(rate, d)) * dt;
  }
  function move(a, dt, floor) {
    var c = Math.cos(a.pitch);
    a.pos[0] += Math.cos(a.yaw) * c * a.speed * dt;
    a.pos[1] = Math.max(Math.min(a.pos[1] + Math.sin(a.pitch) * a.speed * dt, -0.12), floor);
    a.pos[2] += Math.sin(a.yaw) * c * a.speed * dt;
  }

  // Instance writers (12 floats each, positions relative to the grid origin O).
  Ecosystem.prototype.writeFish = function (out, O, max) {
    var n = 0;
    for (var s = 0; s < this.schools.length; s++) {
      var sc = this.schools[s];
      if (!sc.alive) continue;
      for (var i = 0; i < sc.n && n < max; i++, n++) {
        var o = i * 3, q = n * 12, len = sc.S.len * (0.85 + 0.3 * sc.seed[i]);
        out[q] = sc.pos[o] - O[0]; out[q + 1] = sc.pos[o + 1]; out[q + 2] = sc.pos[o + 2] - O[1]; out[q + 3] = len;
        headingQuat(sc.vel[o], sc.vel[o + 1], sc.vel[o + 2], out, q + 4);
        var sp = Math.hypot(sc.vel[o], sc.vel[o + 1], sc.vel[o + 2]);
        out[q + 8] = sc.seed[i]; out[q + 9] = sc.sp; out[q + 10] = sc.S.wag * (0.6 + 0.8 * sp / sc.S.speed); out[q + 11] = sc.seed[i] * TAU;
      }
    }
    return n;
  };
  function writeAgents(list, out, O, size) {
    var n = 0;
    list.forEach(function (a) {
      if (!a.alive) return;
      var q = n * 12, c = Math.cos(a.pitch);
      out[q] = a.pos[0] - O[0]; out[q + 1] = a.pos[1]; out[q + 2] = a.pos[2] - O[1]; out[q + 3] = size * (0.85 + 0.3 * a.seed);
      headingQuat(Math.cos(a.yaw) * c, Math.sin(a.pitch), Math.sin(a.yaw) * c, out, q + 4);
      out[q + 8] = a.seed; out[q + 9] = 0; out[q + 10] = 1; out[q + 11] = a.seed * TAU;
      n++;
    });
    return n;
  }
  Ecosystem.prototype.writeTurtles = function (out, O) { return writeAgents(this.turtles, out, O, 1.0); };
  Ecosystem.prototype.writeRays = function (out, O) { return writeAgents(this.rays, out, O, 1.5); };
  Ecosystem.prototype.fishCount = function () { var n = 0; this.schools.forEach(function (s) { if (s.alive) n += s.n; }); return n; };

  return { Ecosystem: Ecosystem, SPECIES: SPECIES, headingQuat: headingQuat };
});
