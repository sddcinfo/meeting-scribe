// Meeting Scribe — Audio player (bootstrap).
//
// Constructs the AudioPlayer singleton (which binds listeners to the
// #player-bar DOM) and stamps ``window.audioPlayer`` so cross-feature
// consumers (slide-viewer.js, showSpeakerModal,
// MeetingsManager.openMeeting, …) share one instance.

import { AudioPlayer } from "./audio-player.js";

const audioPlayer = new AudioPlayer();
window.audioPlayer = audioPlayer;
