# deploy/docker/ui.Dockerfile  (build context: repository root)
# node:24-alpine as of 2026-09-21
FROM node@sha256:ebfe2f90462722a7a4de65e91990e97fe0d401c70e0e762c5b53302f905ec1c1 AS build
WORKDIR /app
COPY apps/assistant/package.json apps/assistant/package-lock.json ./
RUN npm ci
COPY apps/assistant/ .
RUN npm run build

# nginx:1.29-alpine as of 2026-09-21
FROM nginx@sha256:5616878291a2eed594aee8db4dade5878cf7edcb475e59193904b198d9b830de
COPY --from=build /app/dist/assistant/browser /usr/share/nginx/html
COPY deploy/docker/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
