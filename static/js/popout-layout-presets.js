/* Three popout layout presets.
 *
 * Curated for the simplest mental model the user actually wants:
 *   * `translate`    — transcript only (just the translation column).
 *   * `translator`   — Presentation: transcript on top, slides below.
 *                      (Slug stays `translator` to preserve any
 *                      localStorage state from prior versions.)
 *   * `triple`       — Triple stack: transcript, slides, terminal.
 *
 * Ids namespaced per-preset so ratio overrides persisted in localStorage
 * don't collide across presets.
 */

(function () {
  'use strict';

  const { leaf, split } = window.PopoutLayout;

  const translate = {
    slug: 'translate',
    label: 'Translate',
    description: 'Translation only — transcript fills the window.',
    tree: leaf('transcript'),
  };

  const translator = {
    slug: 'translator',
    label: 'Presentation',
    description: 'Translation top, original + translated slides side-by-side below.',
    tree: split(
      'v', 0.55,
      leaf('transcript'),
      leaf('slides'),
      'translator:main',
    ),
  };

  const triple = {
    slug: 'triple',
    label: 'Triple stack',
    description: 'Translation, slides, terminal — stacked vertically.',
    tree: split(
      'v', 0.45,
      leaf('transcript'),
      split(
        'v', 0.55,
        leaf('slides'),
        leaf('terminal'),
        'triple:bottom',
      ),
      'triple:main',
    ),
  };

  const PRESETS = { translate, translator, triple };
  const PRESET_ORDER = ['translate', 'translator', 'triple'];

  function get(slug) {
    return PRESETS[slug] || PRESETS.translator;
  }

  function hasTerminal(slug) {
    const preset = get(slug);
    return window.PopoutLayout.collectPanels(preset.tree).includes('terminal');
  }

  function hasSlides(slug) {
    const preset = get(slug);
    return window.PopoutLayout.collectPanels(preset.tree).includes('slides');
  }

  function hasTranscript(slug) {
    const preset = get(slug);
    return window.PopoutLayout.collectPanels(preset.tree).includes('transcript');
  }

  window.PopoutLayoutPresets = {
    PRESETS, PRESET_ORDER, get, hasTerminal, hasSlides, hasTranscript,
  };
})();
