/*
 * Resizable sidebars for the Antenna Solver MoM documentation.
 *
 * A thin handle is inserted between each sidebar and the page content:
 *   - drag it (mouse, pen or touch) to resize the column;
 *   - double-click it to go back to the theme's default width;
 *   - with the keyboard: Tab to the handle, then Left / Right arrows
 *     (Shift for bigger steps), Home / End for the minimum / maximum.
 * The width is remembered in this browser (localStorage), per column.
 *
 * The widths are passed to the theme through CSS custom properties set on
 * <html> (--mom-sidebar-primary-width, --mom-sidebar-secondary-width), which
 * custom.css maps onto pydata-sphinx-theme's own variables. Nothing happens on
 * narrow screens, where the sidebars are drawers, nor while the left sidebar
 * is collapsed with the theme's "Collapse Sidebar" button.
 */
(function () {
  "use strict";

  var COLUMNS = {
    primary: {
      sidebarId: "pst-primary-sidebar",
      cssVar: "--mom-sidebar-primary-width",
      storageKey: "mom.sidebar.primary.px",
      label: "Resize the navigation column",
      media: "(min-width: 960px)",
      minRem: 12,
      maxViewport: 0.4,
      maxRem: 40,
      side: "after"           // handle on the right of the sidebar
    },
    secondary: {
      sidebarId: "pst-secondary-sidebar",
      cssVar: "--mom-sidebar-secondary-width",
      storageKey: "mom.sidebar.secondary.px",
      label: "Resize the “On this page” column",
      media: "(min-width: 1200px)",
      minRem: 10,
      maxViewport: 0.3,
      maxRem: 30,
      side: "before"          // handle on the left of the sidebar
    }
  };
  var STEP_PX = 16;
  var root = document.documentElement;

  function storageGet(key) {
    try { return window.localStorage.getItem(key); } catch (e) { return null; }
  }
  function storageSet(key, value) {
    try { window.localStorage.setItem(key, value); } catch (e) { /* private mode */ }
  }
  function storageRemove(key) {
    try { window.localStorage.removeItem(key); } catch (e) { /* private mode */ }
  }

  function remPx() {
    return parseFloat(window.getComputedStyle(root).fontSize) || 16;
  }
  function bounds(col) {
    var rem = remPx();
    var max = Math.min(col.maxRem * rem, col.maxViewport * window.innerWidth);
    var min = col.minRem * rem;
    return { min: min, max: Math.max(min, max) };
  }
  function clamp(col, px) {
    var b = bounds(col);
    return Math.round(Math.min(b.max, Math.max(b.min, px)));
  }
  function apply(col, px) {
    root.style.setProperty(col.cssVar, px + "px");
  }

  // Restore the saved widths at once (this script runs in <head>), so the page
  // is laid out with them from the first paint.
  Object.keys(COLUMNS).forEach(function (name) {
    var saved = parseFloat(storageGet(COLUMNS[name].storageKey));
    if (isFinite(saved) && saved > 0) {
      apply(COLUMNS[name], saved);
    }
  });

  function setupHandle(name) {
    var col = COLUMNS[name];
    var sidebar = document.getElementById(col.sidebarId);
    if (!sidebar || !sidebar.children.length ||
        sidebar.classList.contains("no-sidebar") || sidebar.classList.contains("hide-on-wide")) {
      return;   // no column to resize on this page
    }

    var handle = document.createElement("div");
    handle.className = "mom-resizer mom-resizer--" + name;
    handle.setAttribute("role", "separator");
    handle.setAttribute("aria-orientation", "vertical");
    handle.setAttribute("aria-controls", col.sidebarId);
    handle.setAttribute("aria-label", col.label);
    handle.setAttribute("tabindex", "0");
    handle.title = "Drag to resize · double-click to reset";
    sidebar.insertAdjacentElement(col.side === "after" ? "afterend" : "beforebegin", handle);

    // Moving the handle to the right widens the left column and narrows the right one.
    var direction = col.side === "after" ? 1 : -1;
    var media = window.matchMedia(col.media);

    // The width last asked for. The theme animates the left column's width,
    // so measuring the element in the middle of that animation would make
    // successive key presses (or a drag started right after one) lose steps.
    var target = null;
    var saved = parseFloat(storageGet(col.storageKey));
    if (isFinite(saved) && saved > 0) { target = clamp(col, saved); }

    function currentWidth() {
      return target !== null ? target : sidebar.getBoundingClientRect().width;
    }
    function updateAria() {
      var b = bounds(col);
      handle.setAttribute("aria-valuemin", String(Math.round(b.min)));
      handle.setAttribute("aria-valuemax", String(Math.round(b.max)));
      handle.setAttribute("aria-valuenow", String(Math.round(currentWidth())));
    }
    function setWidth(px, save) {
      target = clamp(col, px);
      apply(col, target);
      if (save) { storageSet(col.storageKey, String(target)); }
      updateAria();
    }

    var startX = 0;
    var startWidth = 0;
    handle.addEventListener("pointerdown", function (event) {
      if (!media.matches || event.button !== 0) { return; }
      event.preventDefault();
      startX = event.clientX;
      startWidth = currentWidth();
      handle.setPointerCapture(event.pointerId);
      handle.classList.add("is-active");
      root.classList.add("mom-resizing");
    });
    handle.addEventListener("pointermove", function (event) {
      if (!handle.hasPointerCapture(event.pointerId)) { return; }
      setWidth(startWidth + direction * (event.clientX - startX), false);
    });
    function endDrag(event) {
      if (!handle.hasPointerCapture(event.pointerId)) { return; }
      handle.releasePointerCapture(event.pointerId);
      handle.classList.remove("is-active");
      root.classList.remove("mom-resizing");
      setWidth(currentWidth(), true);
    }
    handle.addEventListener("pointerup", endDrag);
    handle.addEventListener("pointercancel", endDrag);

    handle.addEventListener("dblclick", function () {
      target = null;
      root.style.removeProperty(col.cssVar);
      storageRemove(col.storageKey);
      updateAria();
    });

    handle.addEventListener("keydown", function (event) {
      if (!media.matches) { return; }
      var step = event.shiftKey ? 4 * STEP_PX : STEP_PX;
      var b = bounds(col);
      var width = currentWidth();
      if (event.key === "ArrowRight") { width += direction * step; }
      else if (event.key === "ArrowLeft") { width -= direction * step; }
      else if (event.key === "Home") { width = b.min; }
      else if (event.key === "End") { width = b.max; }
      else { return; }
      event.preventDefault();
      setWidth(width, true);
    });

    // Keep a saved width inside the allowed range when the window changes size.
    window.addEventListener("resize", function () {
      var stored = parseFloat(storageGet(col.storageKey));
      if (isFinite(stored) && stored > 0 && media.matches) {
        target = clamp(col, stored);
        apply(col, target);
      }
      updateAria();
    });
    updateAria();
  }

  function init() {
    setupHandle("primary");
    setupHandle("secondary");
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
