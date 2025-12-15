
self.addEventListener('push', function (event) {
    if (event.data) {
        const data = event.data.text(); // Assume text payload for simplicity, or json
        const options = {
            body: data,
            icon: './app-icon.png',
            badge: './app-icon.png',
            vibrate: [100, 50, 100],
            data: {
                dateOfArrival: Date.now(),
                primaryKey: 1
            }
        };
        event.waitUntil(
            self.registration.showNotification('Smart DCA', options)
        );
    }
});

self.addEventListener('notificationclick', function (event) {
    event.notification.close();
    event.waitUntil(
        clients.openWindow('./')
    );
});
