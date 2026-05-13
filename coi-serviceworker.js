/* Enables SharedArrayBuffer (→ ORT multi-threaded WASM) on GitHub Pages by
   injecting COOP/COEP headers via a Service Worker.
   Based on gzuidhof/coi-serviceworker (MIT). */
(() => {
  'use strict';

  if (typeof window === 'undefined') {
    // ---- Service Worker context -------------------------------------------
    self.addEventListener('install', () => self.skipWaiting());
    self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

    self.addEventListener('fetch', (e) => {
      const req = e.request;
      if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
      e.respondWith(
        fetch(req).then((res) => {
          if (res.status === 0) return res;
          const h = new Headers(res.headers);
          // 'credentialless' is safer than 'require-corp' for pages that
          // load CDN resources (fonts, ort.min.js) — no CORP header needed.
          h.set('Cross-Origin-Embedder-Policy', 'credentialless');
          h.set('Cross-Origin-Opener-Policy', 'same-origin');
          return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
        })
      );
    });
    return;
  }

  // ---- Main-thread context -------------------------------------------------
  if (!window.crossOriginIsolated) {
    navigator.serviceWorker
      .register(document.currentScript.src)
      .then((reg) => {
        // First visit: SW just installed but doesn't control the page yet → reload.
        if (!navigator.serviceWorker.controller) window.location.reload();
      })
      .catch((err) => console.warn('[coi-sw] registration failed:', err));
  }
})();
