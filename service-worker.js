const CACHE_NAME = "present-shell-v1";
const APP_SHELL = [
  "./",
  "./index.html",
  "./app.js",
  "./style.css",
  "./favicon.svg"
];

let networkLockdown = false;

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(APP_SHELL))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("message", (event) => {
  if (event.data?.type === "SET_NETWORK_LOCKDOWN") {
    networkLockdown = !!event.data.enabled;
  }
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  const sameOrigin = url.origin === self.location.origin;

  if (event.request.mode === "navigate") {
    networkLockdown = false;
  }

  if (networkLockdown && !sameOrigin) {
    event.respondWith(Response.error());
    return;
  }

  if (sameOrigin && event.request.method === "GET") {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) return cached;
        return fetch(event.request).then((response) => {
          const copy = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copy));
          return response;
        });
      })
    );
  }
});
