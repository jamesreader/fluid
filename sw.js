/* Service worker for SPH Liquid: network-first (always fresh while online),
 * falling back to cache when offline.
 *
 * VERSION DISCIPLINE: bump CACHE below on every deploy that changes
 * index.html, sim.js, or any ocean page or module (v1 or v2/). The name change is what forces
 * every installed copy to evict its stale shell on next visit — the
 * byte-change in THIS file is also what makes browsers re-check at all (they
 * don't re-fetch an unchanged sw.js). Forgetting the bump strands old installs
 * on old physics mixed with fresh HTML, which reads as "UI fine, simulation
 * dead". */
var CACHE = "sphfluid-v8";
var ASSETS = [
  "./",
  "./index.html",
  "./sim.js",
  "./ocean.html",
  "./ocean.js",
  "./world.js",
  "./life.js",
  "./play.js",
  "./sound.js",
  "./v2/ocean.html",
  "./v2/ocean.js",
  "./v2/world.js",
  "./v2/life.js",
  "./v2/play.js",
  "./v2/sound.js",
  "./manifest.webmanifest",
  "./icons/icon-192.png",
  "./icons/icon-512.png"
];

self.addEventListener("install", function (e) {
  e.waitUntil(
    caches.open(CACHE).then(function (c) { return c.addAll(ASSETS); })
      .then(function () { return self.skipWaiting(); })
  );
});

self.addEventListener("activate", function (e) {
  e.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.filter(function (k) { return k !== CACHE; })
                          .map(function (k) { return caches.delete(k); }));
    }).then(function () { return self.clients.claim(); })
  );
});

self.addEventListener("fetch", function (e) {
  if (e.request.method !== "GET" || !e.url.startsWith(self.location.origin)) return;
  e.respondWith(
    fetch(e.request).then(function (res) {
      // Only cache good, same-origin responses — a cached 404 or an opaque
      // redirect would poison the offline fallback.
      if (res.ok && res.type === "basic") {
        var copy = res.clone();
        caches.open(CACHE).then(function (c) { c.put(e.request, copy); });
      }
      return res;
    }).catch(function () {
      return caches.match(e.request).then(function (hit) {
        return hit || caches.match("./index.html");   // offline app shell
      });
    })
  );
});
