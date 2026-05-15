// Meeting Scribe — meeting timer (pure module).
//
// Wall-clock `mm:ss` ticker shown in the meeting-mode header. The
// `timer` singleton lives in state.js and is wired to:
//   * `btn-record` start handler → `timer.start()`
//   * `btn-stop` / error rollback → `timer.stop()`
//   * `btn-clear` → `timer.reset()`
//
// Side-effect-free: the class only mutates `el.textContent` and an
// internal interval id. No top-level window publishes, no DOM lookups
// at module load.

export class MeetingTimer {
  constructor(el) {
    this.el = el;
    this.startTime = 0;
    this.interval = null;
  }

  start() {
    this.startTime = Date.now();
    this.interval = setInterval(() => this._tick(), 1000);
  }

  stop() {
    clearInterval(this.interval);
  }

  reset() {
    this.stop();
    if (this.el) this.el.textContent = "00:00";
  }

  _tick() {
    if (!this.el) return;
    const s = Math.floor((Date.now() - this.startTime) / 1000);
    this.el.textContent =
      `${String(Math.floor(s / 60)).padStart(2, "0")}` +
      `:${String(s % 60).padStart(2, "0")}`;
  }
}
