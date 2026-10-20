// tabs (results tables and video groups)
document.querySelectorAll(".tabs").forEach((list) => {
  const buttons = list.querySelectorAll("[data-panel]");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => {
        const on = b === btn;
        b.setAttribute("aria-selected", String(on));
        const p = document.getElementById(b.dataset.panel);
        p.hidden = !on;
        p.querySelectorAll("video[data-autoplay]").forEach((v) => (on ? obs.observe(v) : (v.pause(), obs.unobserve(v))));
      });
    });
  });
});

// play videos only while they are on screen
const obs = new IntersectionObserver((entries) => {
  entries.forEach((e) => {
    const v = e.target;
    if (e.isIntersecting) { v.preload = "auto"; v.play().catch(() => {}); } else { v.pause(); }
  });
}, { threshold: 0.35 });
document.querySelectorAll(".video-panel:not([hidden]) video[data-autoplay]").forEach((v) => obs.observe(v));

// copy BibTeX
const copyButton = document.querySelector("[data-copy-bib]");
const bibtex = document.querySelector("#bibtex");
if (copyButton && bibtex) {
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(bibtex.textContent.trim());
      const original = copyButton.textContent;
      copyButton.textContent = "Copied";
      window.setTimeout(() => { copyButton.textContent = original; }, 1400);
    } catch { copyButton.textContent = "Select text"; }
  });
}

// thumbnail strips: swap the panel's player source
document.querySelectorAll(".video-panel").forEach((panel) => {
  const video = panel.querySelector("video");
  const cap = panel.querySelector(".video-card figcaption");
  panel.querySelectorAll(".thumb").forEach((t) => {
    t.addEventListener("click", () => {
      panel.querySelectorAll(".thumb").forEach((o) => o.setAttribute("aria-pressed", String(o === t)));
      video.poster = t.dataset.poster;
      video.src = t.dataset.src;
      cap.textContent = t.dataset.title;
      video.play().catch(() => {});
    });
  });
});
