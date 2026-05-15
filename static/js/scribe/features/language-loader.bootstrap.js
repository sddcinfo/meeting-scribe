// Meeting Scribe — language-registry bootstrap.
//
// Fires the IIFE that hydrates the setup-screen language dropdowns +
// caches the server's default pair for MeetingsManager. Skipped under
// `?popout=view` since popouts get their language registry via the
// pop-out SPA's own /api/languages fetch.

import { bootLanguageLoader } from "./language-loader.js";

const POPOUT_MODE = new URLSearchParams(location.search).get("popout");
if (!POPOUT_MODE) {
  bootLanguageLoader();
}
