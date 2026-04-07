/* How It Works — Pipeline node activation + language toggle
   Pure vanilla JS. No dependencies. */

(function () {
  'use strict';

  // Pipeline node activation
  const stageToNode = new Map();
  document.querySelectorAll('.pipeline-node[data-stage]').forEach((node) => {
    stageToNode.set(node.dataset.stage, node);
  });

  const nodeObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        const stage = e.target.dataset.stage;
        const node = stageToNode.get(stage);
        if (!node) return;
        if (e.isIntersecting) {
          node.classList.add('active');
        } else {
          node.classList.remove('active');
        }
      });
    },
    { threshold: 0.2 }
  );

  document.querySelectorAll('.stage[data-stage]').forEach((el) => {
    nodeObserver.observe(el);
  });

  // Language toggle — sets data-lang attribute on body
  const langBtns = document.querySelectorAll('.lang-btn');

  function setLang(lang) {
    document.body.dataset.lang = lang;
    langBtns.forEach((b) => b.classList.toggle('active', b.dataset.lang === lang));
  }

  langBtns.forEach((btn) => {
    btn.addEventListener('click', () => setLang(btn.dataset.lang));
  });

  // Initialize
  setLang('both');

  // Smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener('click', (e) => {
      const target = document.querySelector(link.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
})();
