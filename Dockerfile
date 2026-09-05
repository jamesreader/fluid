FROM nginx:1-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html sim.js sw.js manifest.webmanifest /usr/share/nginx/html/
COPY icons /usr/share/nginx/html/icons
# Stamp @@VER@@ in index.html with this build's identity. Coolify may pass
# GIT_COMMIT/BUILD_DATE as empty build args, so the fallback chain is
# commit -> date -> build timestamp; the grep prints the result into the
# deploy log so a silent no-stamp can never happen on our watch.
ARG GIT_COMMIT=
ARG BUILD_DATE=
RUN V="$GIT_COMMIT"; \
    if [ -z "$V" ] || [ "$V" = local ] || [ "$V" = 0 ]; then V="$BUILD_DATE"; fi; \
    if [ -z "$V" ] || [ "$V" = 0 ]; then V="$(date -u +%s)"; fi; \
    sed -i "s/@@VER@@/$(printf %.7s "$V")/g" /usr/share/nginx/html/index.html; \
    grep -o 'sim\.js?v=[^"]*' /usr/share/nginx/html/index.html
EXPOSE 80
