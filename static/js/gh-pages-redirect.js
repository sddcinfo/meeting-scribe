// GitHub Pages serves static/ as the site root — redirect to docs.
// Extracted from inline <script> in static/index.html so the strict CSP
// (`script-src 'self'`) blocks no expected behavior.
(function () {
  if (location.hostname.endsWith("github.io")) {
    location.replace("how-it-works.html");
  }
})();
