# Frontend image for the RagFabric (Angular).

# Build stage
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

# Serve stage
FROM nginx:alpine
# The Angular application builder emits the site under dist/frontend/browser.
COPY --from=build /app/dist/frontend/browser /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
