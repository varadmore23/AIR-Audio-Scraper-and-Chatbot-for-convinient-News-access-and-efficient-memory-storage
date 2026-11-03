/* Shared page behaviour for Home and Newsroom: theme toggle + persistence, scroll reveals, count-up. */
(function () {
  const KEY = 'vani-theme';
  const root = document.documentElement;
  const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function store(theme) { try { localStorage.setItem(KEY, theme); } catch (e) { /* private mode */ } }

  function setTheme(theme) {
    if (!reduced()) {
      root.classList.add('theme-anim');
      clearTimeout(setTheme.t);
      setTheme.t = setTimeout(() => root.classList.remove('theme-anim'), 500);
    }
    root.dataset.theme = theme;
    store(theme);
    document.querySelectorAll('.theme-toggle').forEach(b => {
      b.setAttribute('aria-pressed', String(theme === 'dark'));
      b.setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
      b.title = b.getAttribute('aria-label');
    });
  }
  window.vaniToggleTheme = () => setTheme(root.dataset.theme === 'dark' ? 'light' : 'dark');

  // Keep in step with another open tab.
  window.addEventListener('storage', e => { if (e.key === KEY && e.newValue) root.dataset.theme = e.newValue; });

  function countUp(el) {
    const target = Number(el.dataset.count);
    if (!Number.isFinite(target)) return;
    if (reduced() || target === 0) { el.textContent = String(target); return; }
    const dur = 1400, t0 = performance.now();
    const step = now => {
      const p = Math.min(1, (now - t0) / dur), eased = 1 - Math.pow(1 - p, 3);
      el.textContent = String(Math.round(target * eased));
      if (p < 1) requestAnimationFrame(step);
    };
    el.textContent = '0';
    requestAnimationFrame(step);
  }

  // Reveal [data-reveal] and start [data-count] when they enter the viewport.
  window.vaniObserve = function (scope) {
    const els = (scope || document).querySelectorAll('[data-reveal]:not(.in), [data-count]:not([data-counted])');
    if (!('IntersectionObserver' in window) || reduced()) {
      els.forEach(el => { el.classList.add('in'); if (el.dataset.count != null) { el.dataset.counted = '1'; el.textContent = el.dataset.count; } });
      return;
    }
    const io = new IntersectionObserver(entries => entries.forEach(en => {
      if (!en.isIntersecting) return;
      const el = en.target;
      el.classList.add('in');
      if (el.dataset.count != null && !el.dataset.counted) { el.dataset.counted = '1'; countUp(el); }
      io.unobserve(el);
    }), { threshold: 0.18, rootMargin: '0px 0px -40px 0px' });
    els.forEach(el => io.observe(el));
  };

  document.addEventListener('DOMContentLoaded', () => {
    setTheme(root.dataset.theme === 'dark' ? 'dark' : 'light');
    root.classList.remove('theme-anim');
    document.querySelectorAll('.theme-toggle').forEach(b => b.addEventListener('click', window.vaniToggleTheme));
    window.vaniObserve();
  });
})();
