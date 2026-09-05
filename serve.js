#!/usr/bin/env node
"use strict";
/*
 * serve.js — zero-dependency LAN file server for the fluid sim.
 *
 *   node serve.js [port]        (default 8080)
 *
 * Serves this folder over http on all interfaces and prints the URLs your
 * phone/tablet can use (must be on the same Wi-Fi). no-cache everywhere so
 * a refreshed tablet always sees the newest sim.js during development.
 */
var http = require("http");
var fs = require("fs");
var path = require("path");
var os = require("os");

var PORT = (+process.argv[2] || 8080);
var ROOT = __dirname;
var TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json",
  ".webmanifest": "application/manifest+json",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".css": "text/css; charset=utf-8",
};

http.createServer(function (req, res) {
  var p;
  try { p = decodeURIComponent(req.url.split("?")[0]); } catch (e) { res.writeHead(400).end("bad url"); return; }
  if (p === "/") p = "/index.html";
  var f = path.normalize(path.join(ROOT, p));
  if (f.indexOf(ROOT) !== 0) { res.writeHead(403).end("forbidden"); return; }   // no traversal
  fs.readFile(f, function (err, data) {
    if (err) { res.writeHead(404).end("not found"); return; }
    res.writeHead(200, {
      "Content-Type": TYPES[path.extname(f).toLowerCase()] || "application/octet-stream",
      "Cache-Control": "no-cache",
    });
    res.end(data);
  });
}).listen(PORT, "0.0.0.0", function () {
  console.log("SPH Liquid serving " + ROOT);
  var nets = os.networkInterfaces(), seen = {};
  Object.keys(nets).forEach(function (name) {
    nets[name].forEach(function (a) {
      if (a.family === "IPv4" && !a.internal && !seen[a.address]) {
        seen[a.address] = 1;
        console.log("  this device : http://localhost:" + PORT);
        console.log("  phone/tablet: http://" + a.address + ":" + PORT + "   (same Wi-Fi, name may vary by interface)");
      }
    });
  });
  console.log("Interface names above are hints; if a URL is refused, try the IPv4 shown by ipconfig.");
});
