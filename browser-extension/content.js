// Verdict content script
// Runs after the browser has rendered the page (real session, real JS,
// no bot-detection wall) — reads review text straight from the DOM.

const DEFAULT_API_BASE = "http://127.0.0.1:5000"; // replace with your deployed Render URL

function getSelectors() {
  if (location.hostname.includes("amazon")) {
    return { review: "span[data-hook='review-body']", container: "div[data-hook='review']" };
  }
  if (location.hostname.includes("flipkart")) {
    return { review: "div._27M-vq, div.t-ZTKy", container: "div.col.EPCmJX, div._27M-vq" };
  }
  return null;
}

function badge(label, confidence) {
  const el = document.createElement("span");
  el.className = "verdict-badge verdict-badge--" + label;
  el.textContent = (label === "fake" ? "⚑ LIKELY FAKE" : "✓ LIKELY GENUINE") + ` · ${confidence}%`;
  return el;
}

async function analyzeAndTag(reviewEl, apiBase) {
  const text = reviewEl.textContent.trim();
  if (text.length < 15 || reviewEl.dataset.verdictDone) return;
  reviewEl.dataset.verdictDone = "1";
  try {
    const res = await fetch(apiBase + "/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) return;
    const data = await res.json();
    reviewEl.parentElement.insertBefore(badge(data.label, data.confidence), reviewEl);
  } catch (e) {
    // API unreachable (likely not deployed / wrong URL) — fail silently on page,
    // details available in the extension popup / console.
    console.warn("Verdict: could not reach API at", apiBase, e);
  }
}

chrome.storage.sync.get(["apiBase"], (cfg) => {
  const apiBase = cfg.apiBase || DEFAULT_API_BASE;
  const sel = getSelectors();
  if (!sel) return;

  const scan = () => {
    document.querySelectorAll(sel.review).forEach((el) => analyzeAndTag(el, apiBase));
  };

  scan();
  // Amazon/Flipkart load more reviews via infinite scroll / pagination —
  // re-scan when new nodes show up.
  const observer = new MutationObserver(() => scan());
  observer.observe(document.body, { childList: true, subtree: true });
});
