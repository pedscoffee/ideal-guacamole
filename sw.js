const CACHE_VERSION = "present-pwa-v4"; // Bumped version to force update
const APP_CACHE = `${CACHE_VERSION}-app`;
const RUNTIME_CACHE = `${CACHE_VERSION}-runtime`;

const APP_SHELL = [
  "./",
  "./index.html",
  "./style.css",
  "./app.js",
  "./favicon.svg",
  "./manifest.webmanifest"
];

// Only cache the JS library CDNs.
// DO NOT cache Hugging Face, GitHub, or MLC domains. The ML libraries handle those natively.
const CACHEABLE_REMOTE_HOSTS = new Set([
  "esm.run",
  "cdn.jsdelivr.net"
]);

function isCacheableRemoteHost(hostname) {
  return CACHEABLE_REMOTE_HOSTS.has(hostname);
}

function isCacheableRequest(request) {
  if (request.method !== "GET") return false;
  const url = new URL(request.url);
  
  // Cache local assets
  if (url.origin === self.location.origin) return true;
  
  // Cache only approved CDNs (ignoring all model endpoints)
  return url.protocol === "https:" && isCacheableRemoteHost(url.hostname);
}

async function putIfUsable(cache, request, response) {
  if (!response) return response;
  if (response.ok || response.type === "opaque" || response.type === "cors") {
    try {
      await cache.put(request, response.clone());
    } catch {
      // Swallow quota errors cleanly
    }
  }
  return response;
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(APP_CACHE)
      .then((cache) => cache.addAll(APP_SHELL))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const expected = new Set([APP_CACHE, RUNTIME_CACHE]);
    const keys = await caches.keys();
    // Clear out the old v3 caches that are bogged down with massive duplicated model weights
    await Promise.all(keys.map((key) => expected.has(key) ? undefined : caches.delete(key)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // If the request isn't for our app shell or UI CDNs, let the browser handle it normally.
  if (!isCacheableRequest(request)) return;

  // Handle HTML navigation
  if (request.mode === "navigate") {
    event.respondWith((async () => {
      try {
        const response = await fetch(request);
        await putIfUsable(await caches.open(APP_CACHE), request, response);
        return response;
      } catch {
        return (await caches.match(request)) || caches.match("./index.html");
      }
    })());
    return;
  }

  // Handle local CSS, JS, SVG assets
  if (url.origin === self.location.origin) {
    event.respondWith((async () => {
      const cached = await caches.match(request);
      try {
        const response = await fetch(request);
        return putIfUsable(await caches.open(APP_CACHE), request, response);
      } catch {
        if (cached) return cached;
        throw new Error("Requested local asset is not cached.");
      }
    })());
    return;
  }

  // Handle external CDNs (esm.run, jsdelivr)
  event.respondWith((async () => {
    const cached = await caches.match(request);
    if (cached) return cached;

    try {
      const response = await fetch(request);
      return putIfUsable(await caches.open(RUNTIME_CACHE), request, response);
    } catch (err) {
      throw err;
    }
  })());
});

self.addEventListener("message", (event) => {
  if (event.data?.type === "PRESENT_CACHE_STATUS") {
    event.waitUntil((async () => {
      const port = event.ports?.[0];
      const appCache = await caches.open(APP_CACHE);
      const runtimeCache = await caches.open(RUNTIME_CACHE);
      const appRequests = await appCache.keys();
      const runtimeRequests = await runtimeCache.keys();
      port?.postMessage({
        type: "PRESENT_CACHE_STATUS",
        appAssets: appRequests.length,
        runtimeAssets: runtimeRequests.length
      });
    })());
  }
});