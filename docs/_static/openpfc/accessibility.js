// SPDX-FileCopyrightText: 2026 Jukka Aho
// SPDX-License-Identifier: AGPL-3.0-or-later
// Native horizontal scrolling must also be reachable from the keyboard.
(() => {
  let pending = false;
  const update = () => {
    pending = false;
    for (const element of document.querySelectorAll('pre, .article-body table, .table-wrapper, .math.display, mjx-container[display="true"]')) {
      const overflowing = element.scrollWidth > element.clientWidth + 1;
      if (overflowing && !element.hasAttribute('tabindex')) {
        element.tabIndex = 0;
        element.dataset.scrollFocus = 'true';
      } else if (!overflowing && element.dataset.scrollFocus === 'true') {
        element.removeAttribute('tabindex');
        delete element.dataset.scrollFocus;
      }
    }
  };
  const schedule = () => {
    if (!pending) { pending = true; requestAnimationFrame(update); }
  };
  window.addEventListener('resize', schedule);
  window.addEventListener('load', schedule);
  new MutationObserver(schedule).observe(document.documentElement, { childList: true, subtree: true });
  document.fonts.ready.then(schedule);
  schedule();
})();
