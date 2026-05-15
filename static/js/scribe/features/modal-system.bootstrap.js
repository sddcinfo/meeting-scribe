// Meeting Scribe — Modal system (bootstrap).
//
// Stamps the five window-contract surfaces consumed by slide-viewer.js,
// terminal-panel.js, and the dozen+ inline ``onclick="closeModal()"``
// handlers that live in modal markup strings across the codebase
// (meetings-manager, meeting-controls, speaker-review,
// meeting-tools-modal, recording-lifecycle, …).

import {
  alertDialog,
  closeAllModals,
  closeModal,
  confirmDialog,
  promptDialog,
} from "./modal-system.js";

window.alertDialog = alertDialog;
window.confirmDialog = confirmDialog;
window.promptDialog = promptDialog;
window.closeAllModals = closeAllModals;
// ``closeModal`` is the surface that inline ``onclick="closeModal()"``
// markup expects on every confirm-dialog OK button across the SPA, so
// it must be published on window.
window.closeModal = closeModal;
