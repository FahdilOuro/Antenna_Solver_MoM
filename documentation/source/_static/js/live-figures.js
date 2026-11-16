/*
 * Interactive (Plotly) versions of generated figures.
 *
 * A figure notebook exports a Plotly figure twice: a static image, shown by a
 * normal {figure} directive (caption, number, click to enlarge), and an HTML
 * fragment wrapped in <div class="mom-live-figure pst-js-only">, included with
 * a {raw} html block placed right before that {figure}. The page loads
 * plotly.js once (../_static/plotly/plotly.min.js, copied there by conf.py).
 *
 * With JavaScript, this script moves each live view into the figure that
 * follows it, in place of the image and of its "enlarge" link (whose overlay
 * would otherwise catch the mouse when rotating or zooming). The caption and
 * the figure number stay. Without JavaScript the live view is hidden
 * (pst-js-only) and the static image is shown as usual.
 */
(function () {
  "use strict";

  function nextFigure(el) {
    var next = el.nextElementSibling;
    while (next && next.tagName !== "FIGURE") {
      if (next.classList.contains("mom-live-figure")) { return null; }
      next = next.nextElementSibling;
    }
    return next;
  }

  function init() {
    document.querySelectorAll(".mom-live-figure").forEach(function (live) {
      var figure = nextFigure(live);
      if (!figure) { return; }
      var link = figure.querySelector("a.image-reference");
      var image = link || figure.querySelector("img");
      if (!image) { return; }
      image.replaceWith(live);
      figure.classList.add("mom-live");
      var plot = live.querySelector(".plotly-graph-div");
      if (plot && window.Plotly && window.Plotly.Plots) {
        window.Plotly.Plots.resize(plot);
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
