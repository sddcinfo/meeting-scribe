// Meeting Scribe — Audio-out listener (bootstrap).
//
// Constructs the AudioOutListener singleton + stamps
// ``window.audioOutListener`` so consumers (the Listen button click
// handler, the language/mode pickers, the recording start/stop side
// effects) share one instance. Also kicks off the 2 s telemetry poll
// that pushes a state snapshot to /api/diag/listener.

import {
  AudioOutListener,
  postListenerSnapshot,
} from "./audio-out.js";

const audioOutListener = new AudioOutListener();
window.audioOutListener = audioOutListener;

// Telemetry: push the live state every 2 s so an operator can see
// what's happening with ``scripts/scribe_trace.py --listeners``
// without asking the user to read text off the screen.
setInterval(() => postListenerSnapshot(audioOutListener), 2000);
