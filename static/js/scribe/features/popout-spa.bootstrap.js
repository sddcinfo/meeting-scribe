// Meeting Scribe — Pop-out SPA bootstrap.
//
// Fires after the admin SPA boot orchestrator stamps the
// `popout-view` / `view-only` body classes so they're in place by the
// time we mutate the DOM here. The `?popout=` URL flag is the same
// one every other bootstrap consults — keep them in lockstep.
//
// Skipped on the admin SPA so the popout-only DOM (popout header,
// slide viewer, terminal panel mount) never touches the admin view.

import { bootPopoutSpa } from "./popout-spa.js";

const POPOUT_MODE = new URLSearchParams(location.search).get("popout");
if (POPOUT_MODE) {
  bootPopoutSpa();
}
