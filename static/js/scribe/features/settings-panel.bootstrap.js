// Meeting Scribe — Settings panel bootstrap.
//
// Fires once after the admin SPA boot so DOM nodes
// (#settings-panel, #btn-settings, …) are already in the document by
// the time this wires its listeners.

import { bootSettingsPanel } from "./settings-panel.js";

bootSettingsPanel();
