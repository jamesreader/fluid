// Headless scene harness for the ocean pages (SwiftShader WebGL2).
//
//   node tools/scene.js v1|v2 out-prefix steps.json [w h]
//
// steps: [{viewport: [w, h], eval: "js", fps: 30, frames: n, shot: "name"}]
// run in order; frames are hand-stepped at `fps` (sticky, default 30). An
// eval's result is printed as JSON. SwiftShader is slow: warm the surf up at
// a tiny viewport, then resize for the shots.
const { chromium } = require('/opt/node22/lib/node_modules/playwright');
const fs = require('fs'), path = require('path');
const [,, which, prefix, stepsFile, W = '640', H = '360'] = process.argv;
const page = path.resolve(__dirname, '..', which === 'v2' ? 'v2/ocean.html' : 'ocean.html');
(async () => {
  const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: +W, height: +H }, deviceScaleFactor: 1 });
  p.on('console', m => { if (m.type() === 'error') console.log('[console.error]', m.text().slice(0, 1500)); });
  p.on('pageerror', e => console.log('[pageerror]', e.message.slice(0, 1500)));
  await p.addInitScript(() => { try { localStorage.clear(); } catch (e) {} window.__q = []; window.requestAnimationFrame = (cb) => { window.__q.push(cb); return 1; }; });
  await p.goto('file://' + page);
  await p.waitForFunction(() => window.__lagoon, null, { timeout: 120000, polling: 250 });
  await p.evaluate(() => {
    document.body.classList.add('fs'); window.__t = performance.now(); window.__fps = 30;
    window.__step = (n) => { for (let i = 0; i < n; i++) { window.__t += 1000 / window.__fps; const q = window.__q; window.__q = []; q.forEach(cb => cb(window.__t)); } };
  });
  for (const st of JSON.parse(fs.readFileSync(stepsFile, 'utf8'))) {
    const t0 = Date.now();
    if (st.viewport) { await p.setViewportSize({ width: st.viewport[0], height: st.viewport[1] }); await p.waitForTimeout(200); }
    const out = await p.evaluate((st) => { if (st.fps) window.__fps = st.fps; let r = null; if (st.eval) r = eval(st.eval); window.__step(st.frames || 0); return r === undefined ? null : r; }, st);
    if (out !== null) console.log(JSON.stringify(out));
    if (st.shot) { await p.screenshot({ path: `${prefix}_${st.shot}.png`, timeout: 300000 }); console.log(st.shot, ((Date.now() - t0) / 1000).toFixed(1) + 's'); }
  }
  await b.close();
})();
