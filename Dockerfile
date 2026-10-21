FROM nginx:1-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html sim.js ocean.html ocean.js sw.js manifest.webmanifest /usr/share/nginx/html/
COPY icons /usr/share/nginx/html/icons
# Stamp @@VER@@ in index.html and ocean.html with this build's identity.
# Coolify may pass GIT_COMMIT/BUILD_DATE as empty build args, so the fallback
# chain is commit -> date -> build timestamp; the greps print the result into
# the deploy log so a silent no-stamp can never happen on our watch.
ARG GIT_COMMIT=
ARG BUILD_DATE=
RUN VER=""; \
    [ -z "$GIT_COMMIT" ] || [ "$GIT_COMMIT" = local ] || [ "$GIT_COMMIT" = 0 ] || VER="$(printf %.7s "$GIT_COMMIT")"; \
    if [ -z "$VER" ]; then VER="$(date -u +%s)"; fi; \
    sed -i "s/@@VER@@/$VER/g" /usr/share/nginx/html/index.html /usr/share/nginx/html/ocean.html; \
    grep -o 'sim\.js?v=[^"]*' /usr/share/nginx/html/index.html; \
    grep -o 'ocean\.js?v=[^"]*' /usr/share/nginx/html/ocean.html
EXPOSE 80
