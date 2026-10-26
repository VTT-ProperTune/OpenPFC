// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Complete the keyboard contract of sphinxcontrib-mermaid's fullscreen viewer.
(() => {
  let active = null;
  let returnFocus = null;
  let background = [];
  const focusable = () => [...active.querySelectorAll(
    'a[href], button, input, select, textarea, [tabindex]'
  )].filter(element => !element.disabled && element.tabIndex >= 0 &&
    element.getClientRects().length);
  const update = () => {
    const modal = document.querySelector('.mermaid-fullscreen-modal.active');
    if (modal === active) return;
    if (active) {
      active = null;
      for (const [element, inert, hidden] of background) {
        element.inert = inert;
        if (hidden === null) element.removeAttribute('aria-hidden');
        else element.setAttribute('aria-hidden', hidden);
      }
      background = [];
      if (returnFocus?.isConnected) returnFocus.focus({ preventScroll: true });
    }
    if (modal) {
      returnFocus = document.activeElement;
      active = modal;
      background = [...document.body.children]
        .filter(element => element !== modal && !['SCRIPT', 'STYLE', 'LINK'].includes(element.tagName))
        .map(element => [element, element.inert, element.getAttribute('aria-hidden')]);
      for (const [element] of background) {
        element.inert = true;
        element.setAttribute('aria-hidden', 'true');
      }
      modal.querySelector('.mermaid-fullscreen-close').focus();
    }
  };
  document.addEventListener('keydown', event => {
    if (!active || event.key !== 'Tab') return;
    const items = focusable();
    const first = items[0], last = items[items.length - 1];
    if (!active.contains(document.activeElement) ||
        event.shiftKey && document.activeElement === first ||
        !event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      (event.shiftKey ? last : first)?.focus();
    }
  }, true);
  document.addEventListener('focusin', event => {
    if (active && !active.contains(event.target)) focusable()[0]?.focus();
  });
  new MutationObserver(update).observe(document.body, {
    childList: true, subtree: true, attributes: true, attributeFilter: ['class']
  });
  update();
})();
