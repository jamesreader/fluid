#!/usr/bin/env python3
"""Build Clear Lagoon (v1) or Clear Lagoon II (v2) as one self-contained page.

    python3 tools/artifact.py v1 out.html [--full]
    python3 tools/artifact.py v2 out.html [--full]

--full keeps the document skeleton (doctype, head, body) for hosts that take
a complete page; without it the page body is left for a host that wraps it.

Inlines the page's modules, drops what only works on the deployed site
(manifest, icons, service worker, the v1/v2 compare link, the link to the
2D sim), and checks nothing was left half-cut. Run from the repo root.
"""
import re, sys

which, out, full = sys.argv[1], sys.argv[2], '--full' in sys.argv[3:]
root, up, title = ('.', '', 'Clear Lagoon') if which == 'v1' else ('v2', '../', 'Clear Lagoon II')
html = open(root + '/ocean.html').read()

def cut(a, b='', count=1):
    global html
    assert html.count(a) == count, (html.count(a), a[:60])
    html = html.replace(a, b)

def cut_re(pattern):
    global html
    m = re.search(pattern, html, re.S)
    assert m, pattern
    html = html[:m.start()] + '\n' + html[m.end():]

skeleton = ['<!DOCTYPE html>\n', '<html lang="en">\n', '<head>\n', '</head>\n', '<body>\n', '</body>\n', '</html>\n',
            '<meta charset="utf-8">\n',
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n']
for tag in ([] if full else skeleton) + ['<meta name="theme-color" content="#0b0f14">\n',
            '<link rel="manifest" href="%smanifest.webmanifest">\n' % up,
            '<link rel="icon" type="image/png" sizes="192x192" href="%sicons/icon-192.png">\n' % up]:
    cut(tag)
for name in ['world', 'life', 'play', 'sound', 'ocean']:
    cut('<script src="%s.js?v=@@VER@@"></script>' % name, '<script>\n' + open('%s/%s.js' % (root, name)).read() + '</script>')
cut(' <a href="%sindex.html">2D SPH fluid →</a>' % up)
cut_re(r'\n    <a class="tbtn" id="ui-compare" href="[^"]*" title="[^"]*">[^<]*</a>')
cut_re(r'\n  document\.getElementById\("ui-compare"\)\.addEventListener\("click", function \(e\) \{.*?\n  \}\);')
cut_re(r'\nif \("serviceWorker" in navigator\) \{.*?\n\}')
assert '<title>%s</title>' % title in html[:8192]
assert 'serviceWorker' not in html and '@@VER@@' not in html and 'id="ui-compare"' not in html
open(out, 'w').write(html)
print(len(html), 'bytes')
