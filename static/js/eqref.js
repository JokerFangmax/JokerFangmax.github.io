/* Equation-reference hover tooltips (see assets/js/mathjax-config.js for the
 * pageReady hook that invokes this BEFORE MathJax's initial typeset).
 *
 * Three passes over .article-style text nodes:
 *   1. Collect display equations $$...$$ carrying a numeric \tag{N} -> map N to TeX.
 *   2. Wrap textual references to N -- "式 (2.1)", "式 2.14", "Eq.10" -- in
 *      <span class="eqref">label<span class="eqref-tip">\[TeX\]</span></span>.
 *   3. Symbol tooltips: UNTAGGED display blocks that begin with a definition
 *      "SYMBOL = ..." (e.g. \tilde\mu_t = ..., \qquad \tilde\beta_t = ...)
 *      register SYMBOL; inline math that is exactly "$SYMBOL$" or
 *      "$SYMBOL(args)$" gets the same hover treatment. Only "derived" symbols
 *      (containing \tilde or a theta subscript) are registered so that
 *      ubiquitous plain symbols ($x_t$, $\varepsilon$, ...) stay unadorned.
 *   4. Author-specified tooltips: a post may hand-write
 *      <span class="eqref" data-eqref-tip="TeX">claim</span>. The attribute
 *      carries the TeX verbatim (Goldmark never touches attribute values, so
 *      backslashes survive) and may hold a multi-line derivation chain -- use
 *      this when the "verification" of a claim is a chain rather than one
 *      numbered equation. Runs last so the injected tip text is not rescanned
 *      by the passes above.
 *
 * MathJax then typesets the page including the hidden tooltip spans (the
 * wrapped "$SYMBOL$" remains ordinary inline math inside the span). Pages
 * without \tag or qualifying definitions are left untouched; cross-post
 * references deliberately stay plain.
 */
(function () {
  'use strict';

  var BLOCK_RE = /\$\$([\s\S]+?)\$\$/g;
  var TAG_RE = /\\tag\{([^}]*)\}/;
  var NUMERIC_TAG_RE = /^\d+(?:\.\d+)?$/;
  var MATH_REGION_RE = /\$\$[\s\S]+?\$\$|\$[^$]+\$/g;
  // "式 (2.1)" / "式（2.1）" / "式 2.1" / "Eq.10" / "Eq. (10)"; trailing
  // full-width ） is deliberately NOT consumed (it usually closes an outer
  // （...） around the reference).
  var REF_RE = /(式|Eq\.?)\s*(?:\(\s*(\d+(?:\.\d+)?)\s*\)|（\s*(\d+(?:\.\d+)?)\s*）|(\d+(?:\.\d+)?))/g;
  // References preceded by these characters are false positives: 公式/模式/格式/
  // 形式/算式/项式 (as in 多项式) followed by an unrelated number.
  var BAD_PREFIX_RE = /[公模方格形算项]/;
  // A definable symbol: one or more \commands with an optional subscript
  // (backslash allowed in the subscript: \mu_\theta, \mu_{\theta}).
  var SYMBOL_RE = /^(?:\\[a-zA-Z]+\s*)+(?:_\{?\\?[a-zA-Z]+\}?)?$/;
  // Only "derived" symbols get tooltips (teacher/student quantities like
  // \tilde\mu_t, \mu_\theta); plain symbols ($x_t$, $\theta$, $\cos\theta$)
  // appear too often and would turn the page into confetti.
  function isDerivedSymbol(sym) {
    if (sym.indexOf('\\tilde') !== -1) return true;
    var sub = sym.match(/_\{?(\\?[a-zA-Z]+)\}?$/);
    return !!(sub && sub[1].replace(/^\\/, '').indexOf('theta') !== -1);
  }
  var QQ_SPLIT_RE = /\\qquad/;

  function collectTextNodes(root) {
    var nodes = [];
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: function (node) {
        var el = node.parentElement;
        while (el && el !== root) {
          var tag = el.tagName;
          if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'CODE' || tag === 'PRE') {
            return NodeFilter.FILTER_REJECT;
          }
          el = el.parentElement;
        }
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    var node;
    while ((node = walker.nextNode())) nodes.push(node);
    return nodes;
  }

  function buildEquationMap(root) {
    var eqMap = {};
    collectTextNodes(root).forEach(function (node) {
      var data = node.nodeValue;
      if (data.indexOf('$$') === -1 || data.indexOf('\\tag') === -1) return;
      var match;
      BLOCK_RE.lastIndex = 0;
      while ((match = BLOCK_RE.exec(data))) {
        var tagMatch = TAG_RE.exec(match[1]);
        if (!tagMatch) continue;
        var tag = tagMatch[1].trim();
        if (!NUMERIC_TAG_RE.test(tag)) continue;
        if (!eqMap[tag]) {
          eqMap[tag] = match[1].replace(TAG_RE, '').replace(/\s+/g, ' ').trim();
        }
      }
    });
    return eqMap;
  }

  function makeEqrefSpan(labelText, tex) {
    var span = document.createElement('span');
    span.className = 'eqref';
    span.setAttribute('tabindex', '0');
    span.appendChild(document.createTextNode(labelText));
    var tip = document.createElement('span');
    tip.className = 'eqref-tip';
    tip.setAttribute('aria-hidden', 'true');
    tip.textContent = '\\[' + tex + '\\]';
    span.appendChild(tip);
    // Keep click-toggled state from sticking after the pointer leaves.
    span.addEventListener('mouseleave', function () {
      span.classList.remove('is-open');
    });
    return span;
  }

  // Pass 3 data: untagged display blocks that open with "SYMBOL = ..." define
  // SYMBOL. A block may define several symbols separated by \qquad. First
  // definition wins (the canonical statement usually comes first).
  function buildSymbolMap(root) {
    var symMap = {};
    collectTextNodes(root).forEach(function (node) {
      var data = node.nodeValue;
      if (data.indexOf('$$') === -1 || data.indexOf('=') === -1) return;
      var match;
      BLOCK_RE.lastIndex = 0;
      while ((match = BLOCK_RE.exec(data))) {
        var body = match[1];
        if (TAG_RE.test(body)) continue;
        var tex = body.replace(/\s+/g, ' ').trim();
        body.split(QQ_SPLIT_RE).forEach(function (segment) {
          var def = segment.match(/^\s*([^=]{1,60}?)\s*=\s*(?!=)/);
          if (!def) return;
          var sym = def[1].trim();
          if (!isDerivedSymbol(sym)) return;
          if (!SYMBOL_RE.test(sym)) return;
          var key = sym.replace(/\s+/g, '');
          if (!symMap[key]) symMap[key] = tex;
        });
      }
    });
    return symMap;
  }

  function wrapSymbols(root, symMap) {
    var keys = Object.keys(symMap);
    if (!keys.length) return;
    collectTextNodes(root).forEach(function (node) {
      var data = node.nodeValue;
      if (data.indexOf('$') === -1) return;

      // Inline-math regions only ($$...$$ display blocks are skipped).
      var candidates = [];
      var regionMatch;
      MATH_REGION_RE.lastIndex = 0;
      while ((regionMatch = MATH_REGION_RE.exec(data))) {
        if (regionMatch[0].charAt(1) === '$') continue;
        var inner = regionMatch[0].slice(1, -1).replace(/\s+/g, '');
        var hit = null;
        // Longest symbol wins (e.g. \mu_\theta before \mu).
        keys.forEach(function (key) {
          if (inner === key || (inner.indexOf(key) === 0 && inner.charAt(key.length) === '(')) {
            if (!hit || key.length > hit.key.length) hit = {key: key};
          }
        });
        if (hit) {
          candidates.push({start: regionMatch.index, end: regionMatch.index + regionMatch[0].length,
                           text: regionMatch[0], key: hit.key});
        }
      }
      if (!candidates.length) return;

      var fragment = document.createDocumentFragment();
      var pos = 0;
      candidates.forEach(function (cand) {
        if (cand.start > pos) {
          fragment.appendChild(document.createTextNode(data.slice(pos, cand.start)));
        }
        fragment.appendChild(makeEqrefSpan(cand.text, symMap[cand.key]));
        pos = cand.end;
      });
      if (pos < data.length) {
        fragment.appendChild(document.createTextNode(data.slice(pos)));
      }
      node.parentNode.replaceChild(fragment, node);
    });
  }

  function wrapReferences(root, eqMap) {
    collectTextNodes(root).forEach(function (node) {
      var data = node.nodeValue;
      if (data.indexOf('式') === -1 && data.indexOf('Eq') === -1) return;

      // Spans of this node already claimed by $...$ / $$...$$ math.
      var regions = [];
      var regionMatch;
      MATH_REGION_RE.lastIndex = 0;
      while ((regionMatch = MATH_REGION_RE.exec(data))) {
        regions.push([regionMatch.index, regionMatch.index + regionMatch[0].length]);
      }
      var inMath = function (index) {
        for (var i = 0; i < regions.length; i++) {
          if (index >= regions[i][0] && index < regions[i][1]) return true;
        }
        return false;
      };

      var matches = [];
      var match;
      REF_RE.lastIndex = 0;
      while ((match = REF_RE.exec(data))) {
        var start = match.index;
        var num = match[2] || match[3] || match[4];
        if (!num || !eqMap[num]) continue;
        if (inMath(start)) continue;
        if (match[1] === '式' && start > 0 && BAD_PREFIX_RE.test(data[start - 1])) continue;
        matches.push({start: start, end: start + match[0].length, text: match[0], num: num});
      }
      if (!matches.length) return;

      var fragment = document.createDocumentFragment();
      var pos = 0;
      matches.forEach(function (ref) {
        if (ref.start > pos) {
          fragment.appendChild(document.createTextNode(data.slice(pos, ref.start)));
        }
        fragment.appendChild(makeEqrefSpan(ref.text, eqMap[ref.num]));
        pos = ref.end;
      });
      if (pos < data.length) {
        fragment.appendChild(document.createTextNode(data.slice(pos)));
      }
      node.parentNode.replaceChild(fragment, node);
    });
  }

  // Pass 4: author-specified proof tooltips (see header comment). The tip
  // text is injected AFTER the other passes, so e.g. a "式 2.2" mention in
  // the chain is not re-wrapped into a nested tooltip.
  function activateAuthoredTips(root) {
    var spans = root.querySelectorAll('.eqref[data-eqref-tip]');
    Array.prototype.forEach.call(spans, function (span) {
      var tex = span.getAttribute('data-eqref-tip');
      span.removeAttribute('data-eqref-tip');
      if (!tex) return;
      span.setAttribute('tabindex', '0');
      var tip = document.createElement('span');
      tip.className = 'eqref-tip';
      tip.setAttribute('aria-hidden', 'true');
      tip.textContent = '\\[' + tex + '\\]';
      span.appendChild(tip);
      // Same as makeEqrefSpan: keep click-toggled state from sticking.
      span.addEventListener('mouseleave', function () {
        span.classList.remove('is-open');
      });
    });
  }

  window.eqrefEnhance = function () {
    var root = document.querySelector('.article-style');
    if (!root || root.getAttribute('data-eqref') === 'done') return;
    root.setAttribute('data-eqref', 'done');

    var eqMap = buildEquationMap(root);
    if (Object.keys(eqMap).length) {
      wrapReferences(root, eqMap);
    }
    var symMap = buildSymbolMap(root);
    if (Object.keys(symMap).length) {
      wrapSymbols(root, symMap);
    }
    activateAuthoredTips(root);

    // Touch support: hover does not exist on touchscreens; tap toggles.
    // (Tapping is skipped while the user is selecting text.)
    document.addEventListener('click', function (event) {
      var ref = event.target && event.target.closest ? event.target.closest('.eqref') : null;
      if (!ref || !ref.classList.contains('eqref')) return;
      var selection = window.getSelection ? window.getSelection() : null;
      if (selection && String(selection).length) return;
      ref.classList.toggle('is-open');
    });
  };
})();
