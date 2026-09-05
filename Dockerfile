FROM nginx:1-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html sim.js sw.js manifest.webmanifest /usr/share/nginx/html/
COPY icons /usr/share/nginx/html/icons
EXPOSE 80
