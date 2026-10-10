# deploy/docker/ui.Dockerfile  (build context: repository root)
# node:24-alpine as of 2026-09-21
# The build stage runs on the builder's own platform. Its output is static HTML,
# CSS and JavaScript, identical on every architecture, so only the nginx stage
# below is built per platform. Running npm ci for arm64 under QEMU emulation
# crashed with an illegal instruction and hung the v0.3.1 and v0.4.0 releases.
FROM --platform=$BUILDPLATFORM node@sha256:ebfe2f90462722a7a4de65e91990e97fe0d401c70e0e762c5b53302f905ec1c1 AS build
# The repository layout is kept under /repo because the app compiles the
# TypeScript SDK from its source through a path mapping to
# ../../packages/sdk-typescript/src (Phase 9, decision D22). The SDK has no
# runtime dependencies, so its source is all the build needs from it.
WORKDIR /repo/apps/assistant
COPY apps/assistant/package.json apps/assistant/package-lock.json ./
RUN npm ci
COPY packages/sdk-typescript/src /repo/packages/sdk-typescript/src
COPY apps/assistant/ .
RUN npm run build

# nginx:1.29-alpine as of 2026-09-21
FROM nginx@sha256:5616878291a2eed594aee8db4dade5878cf7edcb475e59193904b198d9b830de
COPY --from=build /repo/apps/assistant/dist/assistant/browser /usr/share/nginx/html
COPY deploy/docker/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
