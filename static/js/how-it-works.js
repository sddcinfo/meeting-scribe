/* How It Works — Pipeline node activation + language toggle
   Pure vanilla JS. No dependencies. */

(function () {
  'use strict';

  // First act: tell the CSS that JS is alive.  The `.reveal` opacity-0
  // starting state is gated on <html class="js-on">, so if this line
  // doesn't run (script error, content blocker, whatever) content
  // stays fully visible instead of being hidden behind a never-fired
  // IntersectionObserver.  Done before the rest of the IIFE body so
  // even a bug below this line leaves the page usable.
  document.documentElement.classList.add('js-on');

  // Pipeline node activation
  const stageToNode = new Map();
  document.querySelectorAll('.pipeline-node[data-stage]').forEach((node) => {
    stageToNode.set(node.dataset.stage, node);
  });

  const nodeObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        const stage = e.target.dataset.stage;
        const node = stageToNode.get(stage);
        if (!node) return;
        if (e.isIntersecting) {
          node.classList.add('active');
        } else {
          node.classList.remove('active');
        }
      });
    },
    { threshold: 0.2 }
  );

  document.querySelectorAll('.stage[data-stage]').forEach((el) => {
    nodeObserver.observe(el);
  });

  // Language toggle — sets data-lang attribute on body and persists
  // the choice in localStorage so it carries across the three
  // editorial pages (how-it-works, benchmarking, hardware-scaling).
  // URL hash takes precedence so deep-links keep working.
  const langBtns = document.querySelectorAll('.lang-btn');
  const LANG_STORAGE_KEY = 'meeting-scribe.editorial.lang';

  function _storeLang(lang) {
    try {
      window.localStorage.setItem(LANG_STORAGE_KEY, lang);
    } catch (_e) {
      // Private mode / storage disabled: silently degrade — the hash
      // and media-query fallbacks below still give a usable default.
    }
  }

  function _readStoredLang() {
    try {
      const v = window.localStorage.getItem(LANG_STORAGE_KEY);
      return v && ['en', 'ja', 'both'].includes(v) ? v : null;
    } catch (_e) {
      return null;
    }
  }

  function setLang(lang) {
    document.body.dataset.lang = lang;
    langBtns.forEach((b) => b.classList.toggle('active', b.dataset.lang === lang));
    // Update URL hash without scrolling
    history.replaceState(null, '', '#lang=' + lang);
    _storeLang(lang);
  }

  langBtns.forEach((btn) => {
    btn.addEventListener('click', () => setLang(btn.dataset.lang));
  });

  // Initialize — URL hash wins (deep-link), then stored choice from a
  // sibling editorial page, then a mobile-friendly default.
  function getInitialLang() {
    const hash = location.hash.replace('#', '');
    const params = new URLSearchParams(hash);
    const urlLang = params.get('lang');
    if (urlLang && ['en', 'ja', 'both'].includes(urlLang)) {
      return urlLang;
    }
    const stored = _readStoredLang();
    if (stored) {
      return stored;
    }
    return window.matchMedia('(max-width: 640px)').matches ? 'en' : 'both';
  }

  setLang(getInitialLang());

  // Smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener('click', (e) => {
      const target = document.querySelector(link.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // Viewport-diagnostics beacon — captures the device viewport into
  // the scribe server's /api/diag/listeners log so we can verify CSS
  // breakpoints against real devices.  Uses navigator.sendBeacon (the
  // fire-and-forget API purpose-built for telemetry) with a fetch
  // fallback.  Skip entirely on the public GitHub Pages mirror — there
  // is no /api/diag/listener endpoint there; without the skip every
  // demo-page load logs a 405 to the visitor's console.
  try {
    const _isGhPages = location.hostname === "github.io"
      || location.hostname.endsWith(".github.io");
    if (!_isGhPages) {
      const beacon = {
        client_id: 'how-it-works-' + Math.random().toString(36).slice(2, 10),
        page: 'how-it-works',
        viewport_w: window.innerWidth,
        viewport_h: window.innerHeight,
        screen_w: (screen && screen.width) || 0,
        screen_h: (screen && screen.height) || 0,
        dpr: window.devicePixelRatio || 1,
        orientation:
          (screen && screen.orientation && screen.orientation.type) || 'unknown',
        ua_short: (navigator.userAgent || '').slice(0, 200),
      };
      const payload = JSON.stringify(beacon);
      let sent = false;
      try {
        if (navigator.sendBeacon) {
          const blob = new Blob([payload], { type: 'application/json' });
          sent = navigator.sendBeacon('/api/diag/listener', blob);
        }
      } catch (_e) { sent = false; }
      if (!sent) {
        try {
          fetch('/api/diag/listener', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: payload,
          }).catch(() => {});
        } catch (_e) { /* content blocker, refused, etc. */ }
      }
    }
  } catch (_e) { /* best-effort only */ }

  // Scroll-reveal for `.reveal` elements (stage cards, memory bar, etc).
  // Uses a once-only observer so revealed-then-scrolled-past items stay
  // visible — there is no need to fade them out when they leave.  Falls
  // back to immediately-visible if IntersectionObserver is unsupported
  // (very old WebKit) so the content still renders.
  const reveals = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window && reveals.length > 0) {
    const revealObserver = new IntersectionObserver(
      (entries, obs) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('in-view');
            obs.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12 }
    );
    reveals.forEach((el) => revealObserver.observe(el));
  } else {
    reveals.forEach((el) => el.classList.add('in-view'));
  }
})();
