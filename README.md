# SPH Liquid

A real-time 2D fluid simulation you can poke, tilt and splash — smoothed-particle
hydrodynamics (Müller 2003: poly6 density, spiky pressure, viscosity laplacian)
in a few hundred lines of plain JavaScript on a canvas. No framework, no build
step, no dependencies.

![velocity-colored dam break](_shot_vel_2s.png)

It's an installable PWA: offline-capable service worker, app icon, fullscreen
mode that harmonizes with the browser's own, and device-tilt gravity on
phones and tablets.

## What's inside

- `sim.js` — the physics core (UMD: browser global + Node export). Weakly
  compressible SPH with a uniform spatial-hash grid and CSR neighbour table.
  Rest density is computed from the kernel's lattice sum at rest spacing, so
  pressure at rest is exactly zero by construction.
- `index.html` — the whole UI: slide-out options panel, brush/attract/repel
  controls, live particle count and resolution knobs, palettes, metaball
  rendering.
- `test.js` — headless physics assertions (settling energy, boundaries,
  neighbour search vs brute force, "is it a fluid *body*" regression, live
  resize, resolution rescale, surface tension).
- `shots.js` / `serve.js` — deterministic PNG screenshots, zero-dependency
  dev server.
- `sw.js` / `manifest.webmanifest` / `icons/` — PWA layer.

## Run it

Open `index.html` in a browser. Or for the full experience (install, offline,
sensors):

```
node serve.js        # prints a LAN URL for your phone
node test.js         # the physics suite
node shots.js        # regenerate screenshots
```

## Tuning invariants

If you adjust things, keep these true: particle size and smoothing radius stay
in a ~2.9 ratio, the CFL number `c·dt/h` stays near 0.2, and rest density comes
from the lattice sum, never a magic number. The comments in `sim.js` explain
why each one exists — they were learned the hard way.

Keys: `Space` pause · `R` re-dam · `G` gravity · `O` options · `F` fullscreen ·
`1/2/3` render mode.

## Clear Lagoon: the 3D ocean

![clear water over a white-sand lagoon](_shot_ocean.jpg)

`ocean.html` is a second, separate sim: a real-time 3D ocean you can fly over
and dive under, with clear tropical water above a white sand floor. WebGL2,
still no build step or dependencies.

- **Waves** are Tessendorf's FFT ocean. A fetch-limited JONSWAP wind sea with
  finite-depth dispersion and the TMA shallow-water correction is split over
  three cascades (233 m, 36 m, 12 m tiles at 256²). The GPU advances the
  spectra and inverse-FFTs them every frame (Stockham radix-2, one fragment
  pass per stage) into choppy displacement, slopes, and a Jacobian that
  seeds whitecap foam.
- **Clear water**: each surface pixel refracts its view ray with Snell's law
  and traces it to the sand. Absorption is close to pure sea water, so red is
  gone after a few metres and blue survives tens: shallow sand bars read pale
  turquoise, and the deeper channels read blue.
- **Caustics** come from the same waves: a mesh of sun rays is refracted
  onto the floor and each triangle brightens by how much its area shrank.
- **Sky**: single-scattering atmosphere baked into a small lookup table per
  sun position (blue noon to orange sunset), a cumulus layer, a GGX sun glint
  and bloom.
- **Under water**: Snell's window, total internal reflection of the floor,
  and blue fog.

Controls: click for mouse look (or drag) · `WASD` move · `Space`/`E` up ·
`Q`/`C` down · `Shift` fast · wheel speed · `P` pause · `R` reset view ·
`H` panel · `F` fullscreen. On touch: drag to look, two-finger drag to fly,
pinch to climb or dive. The panel sets wind, fetch, choppiness, foam, depth,
floor relief, clarity, caustics, sun position, clouds and render quality.

```
node ocean-test.js   # spectrum energy, FFT index math, packing, bathymetry
```
