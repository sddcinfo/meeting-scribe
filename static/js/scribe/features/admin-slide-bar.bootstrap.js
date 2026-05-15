// Meeting Scribe — Admin slide bar bootstrap.
//
// The admin slide preview is desktop-only — popout windows render
// their own slide surface. Gate on the `?popout` URL flag so two DOM
// mutators don't fight for the same elements.

import { bootAdminSlideBar } from "./admin-slide-bar.js";

const POPOUT_MODE = new URLSearchParams(location.search).get("popout");
if (!POPOUT_MODE) {
  bootAdminSlideBar();
}
