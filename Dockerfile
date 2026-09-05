FROM nginx:1-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html sim.js sw.js manifest.webmanifest /usr/share/nginx/html/
COPY icons /usr/share/nginx/html/icons
# Stamp @@VER@@ in index.html with this build's identity (Coolify passes
# GIT_COMMIT/BUILD_DATE as build args). Version-stamped sim.js/sw.js URLs make
# stale-asset mixing impossible: the HTML is served no-cache, so a fresh page
# always requests freshly-named scripts no CDN can answer from old cache.
ARG GIT_COMMIT=local
ARG BUILD_DATE=0
RUN V="$( [ "$GIT_COMMIT" = local ] && echo "$BUILD_DATE" || echo "$GIT_COMMIT" )" && \
    sed -i "s/@@VER@@/$(printf %.7s "$V")/g" /usr/share/nginx/html/index.html
EXPOSE 80
