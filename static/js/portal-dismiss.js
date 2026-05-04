// Captive-portal dismiss flow — extracted from inline <script> in
// static/portal.html so the CSP `script-src 'self'` policy holds.
//
// Sequence:
//   1. Fill in the URL display + the "Open" button's href.
//   2. On Open: drop the scribe_portal cookie, send a no-cors HEAD/GET
//      to /captive-dismiss on port 80 so the redirector flips this
//      client's iOS probe to "Success", then window.open() the page in
//      real Safari (the CNA permits window.open).
(function () {
  var url = location.protocol + "//" + location.host + "/";
  var urlDisplay = document.getElementById("url-display");
  var openBtn = document.getElementById("open-btn");
  if (urlDisplay) urlDisplay.textContent = url;
  if (openBtn) openBtn.href = url;

  if (openBtn) {
    openBtn.addEventListener("click", function (e) {
      e.preventDefault();
      document.cookie = "scribe_portal=done;path=/;max-age=86400";
      fetch("http://" + location.hostname + "/captive-dismiss", {
        mode: "no-cors",
      }).catch(function () {});
      window.open(url, "_blank");
      var h1 = document.querySelector("h1");
      var p = document.querySelector("p");
      if (h1) h1.textContent = "Opening Safari...";
      if (p) p.textContent = "You can close this window.";
    });
  }
})();
