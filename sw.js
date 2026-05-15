// Smart DCA Service Worker
// Supports both legacy plain-text push payloads and new JSON format:
//   { "title": "...", "body": "...", "url": "https://dca.hellokai07.com/#/forum/abc" }

self.addEventListener('push', function (event) {
    if (!event.data) return;

    let title = 'Smart DCA';
    let body = '';
    let url = './';

    // Try JSON payload first; fall back to plain text for legacy notifications
    try {
        const payload = event.data.json();
        if (payload && typeof payload === 'object') {
            title = payload.title || title;
            body = payload.body || '';
            url = payload.url || './';
        }
    } catch (_) {
        body = event.data.text();
    }

    const options = {
        body: body,
        icon: './app-icon.png',
        badge: './app-icon.png',
        vibrate: [100, 50, 100],
        data: {
            url: url,
            dateOfArrival: Date.now()
        }
    };

    event.waitUntil(
        self.registration.showNotification(title, options)
    );
});

self.addEventListener('notificationclick', function (event) {
    event.notification.close();
    const targetUrl = (event.notification.data && event.notification.data.url) || './';

    event.waitUntil(
        clients.matchAll({ type: 'window', includeUncontrolled: true }).then((wins) => {
            // If a window is already open, focus it and navigate
            for (const w of wins) {
                if ('focus' in w) {
                    w.focus();
                    if ('navigate' in w) {
                        try { w.navigate(targetUrl); } catch (_) {}
                    }
                    return;
                }
            }
            // Otherwise open new window
            return clients.openWindow(targetUrl);
        })
    );
});

// Minimal fetch handler — required by Chrome to qualify as installable PWA.
// We do NOT cache anything here (network-first/pass-through) to keep things simple.
self.addEventListener('fetch', (event) => {
    // Pass-through; let the browser handle the request normally.
    // Adding the listener (even no-op) is enough for Chrome's PWA install criteria.
});
