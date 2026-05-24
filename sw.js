const CACHE_VERSION = "present-pwa-v2";
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

const CACHEABLE_REMOTE_HOSTS = new Set([
  "esm.run",
  "cdn.jsdelivr.net",
  "huggingface.co",
  "raw.githubusercontent.com",
  "github.com",
  "objects.githubusercontent.com",
  "media.githubusercontent.com",
  "mlc.ai",
  "webllm.mlc.ai",
  "cas-bridge.xethub.hf.co",
  "transfer.xethub.hf.co"
]);

function isCacheableRemoteHost(hostname) {
  return CACHEABLE_REMOTE_HOSTS.has(hostname)
    || hostname.endsWith(".huggingface.co")
    || hostname.endsWith(".xethub.hf.co")
    || hostname.endsWith(".githubusercontent.com")
    || hostname.endsWith(".mlc.ai");
}

function isCacheableRequest(request) {
  if (request.method !== "GET") return false;

  const url = new URL(request.url);
  if (url.origin === self.location.origin) return true;
  return url.protocol === "https:" && isCacheableRemoteHost(url.hostname);
}

async function putIfUsable(cache, request, response) {
  if (!response) return response;
  if (response.ok || response.type === "opaque" || response.type === "cors") {
    await cache.put(request, response.clone());
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
    await Promise.all(keys.map((key) => expected.has(key) ? undefined : caches.delete(key)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (!isCacheableRequest(request)) return;

  const url = new URL(request.url);

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

  event.respondWith((async () => {
    const cached = await caches.match(request);
    if (cached) return cached;

    const response = await fetch(request);
    return putIfUsable(await caches.open(RUNTIME_CACHE), request, response);
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
