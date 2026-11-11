/*
 * One-click light/dark switch for the Antenna Solver MoM documentation.
 *
 * Replaces the three-entry "Color mode" dropdown of pydata-sphinx-theme:
 *   - by default the page follows the visitor's system setting ("auto");
 *   - a click switches to the other mode;
 *   - when that other mode is the system's own, the page goes back to
 *     following the system, so it changes again if the system does.
 *
 * It uses the theme's own state: data-mode ("auto" | "light" | "dark") and
 * data-theme ("light" | "dark") on <html>, and the localStorage keys "mode"
 * and "theme" that the theme's start-up script reads before the first paint
 * (no flash of the wrong colours on the next page).
 */
(function () {
  "use strict";

  var root = document.documentElement;
  var systemDark = window.matchMedia("(prefers-color-scheme: dark)");

  function systemTheme() {
    return systemDark.matches ? "dark" : "light";
  }
  function storageSet(key, value) {
    try { window.localStorage.setItem(key, value); } catch (e) { /* private mode */ }
  }

  // What the theme's own setTheme() does, for a given mode.
  function apply(mode) {
    var theme = mode === "auto" ? systemTheme() : mode;
    root.dataset.mode = mode;
    root.dataset.theme = theme;
    document.querySelectorAll(".dropdown-menu").forEach(function (el) {
      el.classList.toggle("dropdown-menu-dark", theme === "dark");
    });
    storageSet("mode", mode);
    storageSet("theme", theme);
    updateButtons();
  }

  // The icon (moon in light mode, sun in dark mode) is switched by custom.css
  // from data-theme; only the text for screen readers and the tooltip change here.
  function updateButtons() {
    var label = root.dataset.theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
    document.querySelectorAll(".mom-theme-toggle").forEach(function (btn) {
      btn.setAttribute("aria-label", label);
      btn.setAttribute("title", label);
    });
  }

  function toggle() {
    var next = root.dataset.theme === "dark" ? "light" : "dark";
    apply(next === systemTheme() ? "auto" : next);
  }

  // The theme listens to system changes only while in "auto"; when the
  // visitor has chosen a mode, keep it (run after the theme's own handler).
  systemDark.addEventListener("change", function () {
    window.setTimeout(function () {
      apply(root.dataset.mode === "auto" ? "auto" : root.dataset.mode);
    }, 0);
  });

  function init() {
    var mode = root.dataset.mode;
    apply(mode === "light" || mode === "dark" ? mode : "auto");
    document.querySelectorAll(".mom-theme-toggle").forEach(function (btn) {
      btn.addEventListener("click", toggle);
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
