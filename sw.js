// Smart DCA Service Worker — v3 (Binance migration; bumped to invalidate v2 cache)
// - Push notification support (JSON or plain text)
// - Network-first fetch with offline fallback (required by Chrome to install PWA)

const CACHE_NAME = "smartdca-v3";
const SHELL_FILES = [
    "./",
    "./index.html",
    "./manifest.json",
    "./icon-192.png",
    "./icon-512.png",
    "./apple-touch-icon.png",
];

// Pre-cache shell on install
self.addEventListener("install", (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_FILES).catch(() => {}))
    );
    self.skipWaiting();
});

// Clean old caches on activate
self.addEventListener("activate", (event) => {
    event.waitUntil(
        caches.keys().then((keys) =>
            Promise.all(
                keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
            )
        ).then(() => self.clients.claim())
    );
});

// Network-first fetch handler (required for Chrome PWA install criteria)
self.addEventListener("fetch", (event) => {
    // Only handle GET requests on same origin
    if (event.request.method !== "GET") return;

    const url = new URL(event.request.url);
    // Skip API/Edge Function calls — always go to network
    if (url.hostname.includes("supabase.co") ||
        url.hostname.includes("googleapis.com") ||
        url.hostname.includes("query1.finance.yahoo.com")) {
        return;
    }

    event.respondWith(
        fetch(event.request)
            .then((res) => {
                // Only cache successful same-origin requests
                if (res.ok && url.origin === self.location.origin) {
                    const clone = res.clone();
                    caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone).catch(() => {}));
                }
                return res;
            })
            .catch(() => caches.match(event.request).then((c) => c || caches.match("./index.html")))
    );
});

// ─── Push notifications ───
self.addEventListener("push", function (event) {
    if (!event.data) return;

    let title = "Smart DCA";
    let body = "";
    let url = "./";

    try {
        const payload = event.data.json();
        if (payload && typeof payload === "object") {
            title = payload.title || title;
            body = payload.body || "";
            url = payload.url || "./";
        }
    } catch (_) {
        body = event.data.text();
    }

    const options = {
        body: body,
        icon: "./app-icon.png",
        badge: "./app-icon.png",
        vibrate: [100, 50, 100],
        data: { url: url, dateOfArrival: Date.now() }
    };

    event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener("notificationclick", function (event) {
    event.notification.close();
    const targetUrl = (event.notification.data && event.notification.data.url) || "./";

    event.waitUntil(
        clients.matchAll({ type: "window", includeUncontrolled: true }).then((wins) => {
            for (const w of wins) {
                if ("focus" in w) {
                    w.focus();
                    if ("navigate" in w) {
                        try { w.navigate(targetUrl); } catch (_) {}
                    }
                    return;
                }
            }
            return clients.openWindow(targetUrl);
        })
    );
});
