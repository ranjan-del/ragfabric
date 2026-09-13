# deploy/docker/ui.Dockerfile  (build context: repository root)
FROM node:24-alpine AS build
WORKDIR /app
COPY apps/assistant/package.json apps/assistant/package-lock.json ./
RUN npm ci
COPY apps/assistant/ .
RUN npm run build

FROM nginx:1.29-alpine
COPY --from=build /app/dist/frontend/browser /usr/share/nginx/html
COPY deploy/docker/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
