/*
 * sound.js — the island's soundscape, synthesised with WebAudio (no files).
 *
 *   surf   brown-ish noise, low-passed: the roar of whitewater near you
 *   crash  band-passed noise bursts as wave lips break close by
 *   wind   high-passed noise, rising with wind speed and altitude
 *   engine two detuned oscillators through a low-pass, pitched by the
 *          boat's throttle and speed, fading with distance
 *   splash one-shot noise bursts for things hitting the water
 *   night  a chorus of crickets from the island, after dark
 *
 * Under water everything goes through a closing low-pass and a muffled hum.
 * Browsers only allow audio after a user gesture, so start() is called from
 * a click.
 */
(function (global, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else global.Sound = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  var TAU = 2 * Math.PI;

  function Soundscape() { this.ctx = null; this.on = false; this.lastCrash = 0; }

  Soundscape.prototype.start = function () {
    if (this.ctx) { this.on = true; this.ctx.resume(); this.master.gain.setTargetAtTime(0.9, this.ctx.currentTime, 0.3); return; }
    var AC = (typeof window !== "undefined") && (window.AudioContext || window.webkitAudioContext);
    if (!AC) return;
    var ctx = this.ctx = new AC(), self = this;
    // 4 s of white noise, and a brown (integrated) copy for the deep roar
    var len = ctx.sampleRate * 4, white = ctx.createBuffer(1, len, ctx.sampleRate), brown = ctx.createBuffer(1, len, ctx.sampleRate);
    var wd = white.getChannelData(0), bd = brown.getChannelData(0), last = 0;
    for (var i = 0; i < len; i++) { wd[i] = Math.random() * 2 - 1; last = (last + 0.02 * wd[i]) / 1.02; bd[i] = last * 3.5; }
    this.white = white;
    function loop(buf) { var s = ctx.createBufferSource(); s.buffer = buf; s.loop = true; s.start(0, Math.random() * 3); return s; }
    function filter(type, f, q) { var b = ctx.createBiquadFilter(); b.type = type; b.frequency.value = f; b.Q.value = q || 0.7; return b; }
    function gain(v) { var g = ctx.createGain(); g.gain.value = v; return g; }
    this.master = gain(0);
    this.under = filter("lowpass", 18000, 0.5);
    this.master.connect(this.under); this.under.connect(ctx.destination);
    // surf roar
    this.surfG = gain(0); this.surfF = filter("lowpass", 700, 0.4);
    loop(brown).connect(this.surfF); this.surfF.connect(this.surfG); this.surfG.connect(this.master);
    // breaking hiss
    this.hissG = gain(0); var hb = filter("bandpass", 1800, 0.5);
    loop(white).connect(hb); hb.connect(this.hissG); this.hissG.connect(this.master);
    // wind
    this.windG = gain(0); var wh = filter("highpass", 500, 0.5), wl = filter("lowpass", 2600, 0.5);
    this.windF = wl;
    loop(white).connect(wh); wh.connect(wl); wl.connect(this.windG); this.windG.connect(this.master);
    // engine
    this.engG = gain(0); this.engF = filter("lowpass", 700, 1.2);
    this.osc1 = ctx.createOscillator(); this.osc1.type = "sawtooth"; this.osc1.frequency.value = 40;
    this.osc2 = ctx.createOscillator(); this.osc2.type = "square"; this.osc2.frequency.value = 20.4;
    var o2g = gain(0.4);
    this.osc1.connect(this.engF); this.osc2.connect(o2g); o2g.connect(this.engF);
    this.engF.connect(this.engG); this.engG.connect(this.master);
    this.osc1.start(); this.osc2.start();
    // crickets: a few singers, each a ~4.5 kHz tone gated into trills of
    // three pulses, repeating at its own rate; baked into a 6 s loop
    var cl = ctx.sampleRate * 6, cb = ctx.createBuffer(2, cl, ctx.sampleRate), sr = ctx.sampleRate;
    var singers = [[4420, 0.61, 0.0, 0.9, 0.3], [4710, 0.83, 0.21, 0.6, 0.75], [4180, 0.72, 0.47, 0.45, 0.5], [5030, 1.1, 0.13, 0.3, 0.15], [4560, 0.52, 0.36, 0.35, 0.9]];
    for (var ch = 0; ch < 2; ch++) {
      var cd = cb.getChannelData(ch);
      for (i = 0; i < cl; i++) {
        var tt = i / sr, v = 0;
        for (var q = 0; q < singers.length; q++) {
          var sg = singers[q], ph = (tt / sg[1] + sg[2]) % 1, tp = ph * sg[1];   // time into this singer's chirp
          if (tp < 0.105) {
            var pp = tp % 0.035, env = pp < 0.02 ? Math.sin(Math.PI * pp / 0.02) : 0;
            var pan = ch === 0 ? 1 - sg[4] : sg[4];
            v += env * Math.sin(TAU * sg[0] * tt) * sg[3] * (0.4 + 0.6 * pan);
          }
        }
        cd[i] = v * 0.22;
      }
    }
    this.nightG = gain(0);
    var cs = ctx.createBufferSource(); cs.buffer = cb; cs.loop = true; cs.start(0);
    cs.connect(this.nightG); this.nightG.connect(this.master);
    // underwater hum
    this.humG = gain(0); var hf = filter("lowpass", 160, 0.8);
    loop(brown).connect(hf); hf.connect(this.humG); this.humG.connect(this.under);
    this.on = true;
    this.master.gain.setTargetAtTime(0.9, ctx.currentTime, 0.5);
    void self;
  };
  Soundscape.prototype.stop = function () {
    if (!this.ctx) return;
    this.on = false;
    this.master.gain.setTargetAtTime(0, this.ctx.currentTime, 0.2);
    this.humG.gain.setTargetAtTime(0, this.ctx.currentTime, 0.2);
  };

  // s: { surf 0..1, crash 0..1, wind 0..1, under bool, engine: {throttle, speed, dist} | null,
  //      night 0..1, land 0..1 (how near the island's shore) }
  Soundscape.prototype.update = function (s) {
    if (!this.ctx || !this.on) return;
    var t = this.ctx.currentTime, k = 0.15;
    this.surfG.gain.setTargetAtTime(0.55 * s.surf, t, 0.4);
    this.surfF.frequency.setTargetAtTime(380 + 900 * s.surf, t, 0.5);
    this.hissG.gain.setTargetAtTime(0.06 * s.surf + 0.5 * s.crash, t, 0.08);
    this.windG.gain.setTargetAtTime(0.035 + 0.18 * s.wind, t, 0.6);
    this.windF.frequency.setTargetAtTime(1600 + 2500 * s.wind, t, 0.8);
    var e = s.engine;
    if (e) {
      var rpm = 32 + 50 * Math.abs(e.throttle) + 2.6 * e.speed;
      this.osc1.frequency.setTargetAtTime(rpm, t, k); this.osc2.frequency.setTargetAtTime(rpm * 0.51, t, k);
      this.engF.frequency.setTargetAtTime(380 + 900 * Math.abs(e.throttle), t, k);
      this.engG.gain.setTargetAtTime((0.05 + 0.13 * Math.abs(e.throttle)) / (1 + e.dist / 15), t, k);
    } else this.engG.gain.setTargetAtTime(0, t, 0.3);
    this.under.frequency.setTargetAtTime(s.under ? 420 : 18000, t, 0.08);
    this.humG.gain.setTargetAtTime(s.under ? 0.25 : 0, t, 0.2);
    var nk = Math.max(0, Math.min(1, ((s.night || 0) - 0.4) / 0.5));
    this.nightG.gain.setTargetAtTime(s.under ? 0 : 0.16 * nk * nk * (0.15 + 0.85 * (s.land || 0)), t, 0.8);
  };
  Soundscape.prototype.splash = function (dist, size) {
    if (!this.ctx || !this.on) return;
    var ctx = this.ctx, t = ctx.currentTime, src = ctx.createBufferSource(), bp = ctx.createBiquadFilter(), g = ctx.createGain();
    src.buffer = this.white;
    bp.type = "bandpass"; bp.frequency.value = 700 + 900 / (1 + size); bp.Q.value = 0.8;
    var v = Math.min(0.9, 0.25 + 0.3 * size) / (1 + dist / 12), dur = 0.25 + 0.35 * Math.min(size, 2);
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(v, t + 0.008); g.gain.exponentialRampToValueAtTime(0.0005, t + dur);
    src.connect(bp); bp.connect(g); g.connect(this.master);
    src.start(t, Math.random() * 3, dur + 0.05);
  };

  return { Soundscape: Soundscape };
});
