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

// WebLLM and Transformers.js manage their own Cache API entries for large model weight
// files. Intercepting those in the SW and trying to cache them a second time causes
// competing cache.put() calls on the same response stream, which produces ERR_FAILED.
// Skip SW-level caching for these files and let the libraries handle them directly.
function isModelWeightFile(url) {
  return /\.(bin|safetensors|gguf|ot)(\?.*)?$/i.test(url.pathname);
}

async function putIfUsable(cache, request, response) {
  if (!response) return response;
  if (response.ok || response.type === "opaque" || response.type === "cors") {
    try {
      await cache.put(request, response.clone());
    } catch {
      // Swallow NetworkError / QuotaExceededError thrown when a large response stream
      // is interrupted mid-read during caching. The live response is still returned.
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

    try {
      const response = await fetch(request);
      // Skip SW-level caching for model weight shards (.bin, .safetensors, etc.).
      // WebLLM / Transformers.js manage those caches directly; double-caching
      // the same response stream causes ERR_FAILED on the model download.
      if (isModelWeightFile(url)) return response;
      return putIfUsable(await caches.open(RUNTIME_CACHE), request, response);
    } catch (err) {
      // Network failure with nothing cached — propagate so the library sees it.
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
