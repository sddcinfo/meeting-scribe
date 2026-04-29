/**
 * SlideViewer — displays original + translated slide PNGs side-by-side
 * with navigation, upload, and WebSocket sync.
 *
 * Usage:
 *   const viewer = new SlideViewer(container, meetingId, { isAdmin: true });
 *   viewer.connect(wsConnection);  // attach to existing WS for events
 */

class SlideViewer {
  constructor(container, meetingId, opts = {}) {
    this._container = container;
    this._meetingId = meetingId;
    this._isAdmin = opts.isAdmin || false;
    this._apiBase = opts.apiBase || location.origin;

    this._deckId = null;
    this._totalSlides = 0;
    this._currentIndex = 0;
    this._meta = null;
    this._visible = false;

    this._build();
  }

  _build() {
    this._container.innerHTML = '';
    this._container.className = 'slide-viewer';

    // Upload area (admin only)
    if (this._isAdmin) {
      this._uploadArea = document.createElement('div');
      this._uploadArea.className = 'sv-upload';
      this._uploadArea.innerHTML =
        '<label class="sv-upload-btn">' +
        '<input type="file" accept=".pptx" style="display:none">' +
        'Upload PPTX</label>';
      this._container.appendChild(this._uploadArea);

      const fileInput = this._uploadArea.querySelector('input[type=file]');
      fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) this._handleUpload(e.target.files[0]);
      });

      // Drag and drop
      this._uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        this._uploadArea.classList.add('sv-drag');
      });
      this._uploadArea.addEventListener('dragleave', () => {
        this._uploadArea.classList.remove('sv-drag');
      });
      this._uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        this._uploadArea.classList.remove('sv-drag');
        if (e.dataTransfer.files.length > 0) this._handleUpload(e.dataTransfer.files[0]);
      });
    }

    // Progress indicator
    this._progressEl = document.createElement('div');
    this._progressEl.className = 'sv-progress';
    this._progressEl.style.display = 'none';
    this._container.appendChild(this._progressEl);

    // Slides display area
    this._slidesArea = document.createElement('div');
    this._slidesArea.className = 'sv-slides';
    this._slidesArea.style.display = 'none';

    // Original slide
    const origPane = document.createElement('div');
    origPane.className = 'sv-pane sv-pane-orig';
    this._origImg = document.createElement('img');
    this._origImg.className = 'sv-img';
    this._origImg.alt = 'Original slide';
    origPane.appendChild(this._origImg);
    this._slidesArea.appendChild(origPane);

    // Translated slide
    const transPane = document.createElement('div');
    transPane.className = 'sv-pane sv-pane-trans';
    this._transImg = document.createElement('img');
    this._transImg.className = 'sv-img';
    this._transImg.alt = 'Translated slide';
    this._transSpinner = document.createElement('div');
    this._transSpinner.className = 'sv-spinner';
    this._transSpinner.textContent = 'Translating...';
    transPane.appendChild(this._transImg);
    transPane.appendChild(this._transSpinner);
    this._slidesArea.appendChild(transPane);

    this._container.appendChild(this._slidesArea);

    // Navigation bar
    this._navBar = document.createElement('div');
    this._navBar.className = 'sv-nav';
    this._navBar.style.display = 'none';

    this._prevBtn = document.createElement('button');
    this._prevBtn.className = 'sv-nav-btn';
    this._prevBtn.textContent = '\u25C0';
    this._prevBtn.addEventListener('click', () => this.prev());

    this._slideLabel = document.createElement('span');
    this._slideLabel.className = 'sv-nav-label';

    this._nextBtn = document.createElement('button');
    this._nextBtn.className = 'sv-nav-btn';
    this._nextBtn.textContent = '\u25B6';
    this._nextBtn.addEventListener('click', () => this.next());

    this._navBar.appendChild(this._prevBtn);
    this._navBar.appendChild(this._slideLabel);
    this._navBar.appendChild(this._nextBtn);
    this._container.appendChild(this._navBar);

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (!this._visible) return;
      if (e.key === 'ArrowLeft') this.prev();
      else if (e.key === 'ArrowRight') this.next();
    });
  }

  async _handleUpload(file) {
    // Prefer the styled alertDialog primitive (exposed on window by
    // scribe-app.js). Fall back to window.alert only when slide-viewer
    // is loaded before scribe-app, so we never silently swallow an
    // upload rejection.
    const notify = (title, msg) =>
      (window.alertDialog ? window.alertDialog(title, msg) : Promise.resolve(window.alert(`${title}\n\n${msg}`)));
    if (!file.name.toLowerCase().endsWith('.pptx')) {
      await notify('Unsupported file', 'Only .pptx files are accepted.');
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      await notify('File too large', 'Maximum upload size is 50 MB.');
      return;
    }

    this._showProgress('Uploading...');
    if (this._uploadArea) this._uploadArea.style.display = 'none';

    const formData = new FormData();
    formData.append('file', file);

    try {
      const resp = await fetch(
        `${this._apiBase}/api/meetings/${this._meetingId}/slides/upload`,
        { method: 'POST', body: formData },
      );

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error || `Upload failed: ${resp.status}`);
      }

      const data = await resp.json();
      this._deckId = data.deck_id;
      this._showProgress('Processing slides...');
    } catch (err) {
      this._showProgress(`Error: ${err.message}`);
      if (this._uploadArea) this._uploadArea.style.display = '';
    }
  }

  _showProgress(text) {
    this._progressEl.textContent = text;
    this._progressEl.style.display = '';
  }

  _hideProgress() {
    this._progressEl.style.display = 'none';
  }

  _updateNav() {
    this._slideLabel.textContent = `${this._currentIndex + 1} / ${this._totalSlides}`;
    this._prevBtn.disabled = this._currentIndex <= 0;
    this._nextBtn.disabled = this._currentIndex >= this._totalSlides - 1;
  }

  _loadSlide(index) {
    this._currentIndex = index;
    this._updateNav();

    const base = `${this._apiBase}/api/meetings/${this._meetingId}/slides/${index}`;
    const bustParam = this._deckId ? `?d=${this._deckId}` : '';

    this._origImg.src = `${base}/original${bustParam}`;
    this._origImg.onerror = () => { this._origImg.style.opacity = '0.3'; };
    this._origImg.onload = () => { this._origImg.style.opacity = '1'; };

    this._transImg.src = `${base}/translated${bustParam}`;
    this._transImg.onerror = () => {
      this._transImg.style.display = 'none';
      this._transSpinner.style.display = '';
    };
    this._transImg.onload = () => {
      this._transImg.style.display = '';
      this._transSpinner.style.display = 'none';
    };
  }

  show(totalSlides, deckId) {
    this._totalSlides = totalSlides;
    this._deckId = deckId;
    this._currentIndex = 0;
    this._visible = true;

    this._hideProgress();
    if (this._uploadArea) this._uploadArea.style.display = 'none';
    this._slidesArea.style.display = '';
    this._navBar.style.display = '';

    this._loadSlide(0);
  }

  hide() {
    this._visible = false;
    this._slidesArea.style.display = 'none';
    this._navBar.style.display = 'none';
    if (this._uploadArea) this._uploadArea.style.display = '';
    this._hideProgress();
  }

  next() {
    if (this._currentIndex < this._totalSlides - 1) {
      const newIndex = this._currentIndex + 1;
      this._loadSlide(newIndex);
      if (this._isAdmin) this._sendSlideAdvance(newIndex);
    }
  }

  prev() {
    if (this._currentIndex > 0) {
      const newIndex = this._currentIndex - 1;
      this._loadSlide(newIndex);
      if (this._isAdmin) this._sendSlideAdvance(newIndex);
    }
  }

  goTo(index) {
    if (index >= 0 && index < this._totalSlides) {
      this._loadSlide(index);
    }
  }

  async _sendSlideAdvance(index) {
    try {
      await fetch(
        `${this._apiBase}/api/meetings/${this._meetingId}/slides/current`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ index }),
        },
      );
    } catch { /* best effort */ }
  }

  /** Handle WebSocket events from the server. */
  handleWsEvent(data) {
    switch (data.type) {
      case 'slide_deck_changed':
        this._deckId = data.deck_id;
        this._totalSlides = data.total_slides;
        this.show(data.total_slides, data.deck_id);
        break;

      case 'slide_change':
        if (data.deck_id === this._deckId && !this._isAdmin) {
          this.goTo(data.slide_index);
        }
        break;

      case 'slide_job_progress':
        if (data.stage === 'complete') {
          // Reload current slide to pick up translated version
          this._loadSlide(this._currentIndex);
        } else {
          const label = data.stage.replace(/_/g, ' ');
          const progress = data.progress ? ` (${data.progress})` : '';
          this._showProgress(`${label}${progress}`);
        }
        break;
    }
  }

  /** Poll for existing slide state (for late-joining clients). */
  async checkExisting() {
    try {
      const resp = await fetch(
        `${this._apiBase}/api/meetings/${this._meetingId}/slides`,
      );
      if (!resp.ok) return;

      const meta = await resp.json();
      if (meta.total_slides > 0 && meta.deck_id) {
        this._deckId = meta.deck_id;
        this._totalSlides = meta.total_slides;
        this._currentIndex = meta.current_slide_index || 0;
        this.show(meta.total_slides, meta.deck_id);
        this._loadSlide(this._currentIndex);
      }
    } catch { /* no slides available */ }
  }
}
