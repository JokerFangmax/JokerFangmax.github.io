// MathJax Configuration
//
// This file shadows `themes/academic/assets/js/mathjax-config.js` (Hugo's
// `resources.Get` prefers the site-level `assets/` directory). It keeps the
// theme's original tex/loader options and adds one extension: the
// `startup.pageReady` hook below runs `window.eqrefEnhance()` (defined in
// static/js/eqref.js, loaded synchronously in the head) BEFORE MathJax's
// initial typeset. That function wraps textual equation references like
// "式 (2.1)" in hoverable spans whose hidden child re-states the equation's
// TeX, so the tooltips get typeset together with the page. If anything fails
// there, math rendering proceeds normally.
//
// v2 to v3 upgrade notes (from the theme):
// - The CommonHTML.linebreaks option is not yet implemented (but may be in a future release)
// - The TeX.noUndefined.attributes option is not yet implemented (but may be in a future release)
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: false,
    packages: {'[+]': ['noerrors']}
  },
  loader: {
    load: ['[tex]/noerrors']
  },
  startup: {
    pageReady: function () {
      try {
        if (typeof window.eqrefEnhance === 'function') window.eqrefEnhance();
      } catch (err) {
        // Tooltips are an enhancement; never block math rendering.
      }
      return MathJax.startup.defaultPageReady();
    }
  }
};
