const input = document.getElementById("apiBase");
const status = document.getElementById("status");

chrome.storage.sync.get(["apiBase"], (cfg) => {
  input.value = cfg.apiBase || "http://127.0.0.1:5000";
});

document.getElementById("save").addEventListener("click", () => {
  const value = input.value.trim().replace(/\/$/, "");
  chrome.storage.sync.set({ apiBase: value }, () => {
    status.textContent = "Saved. Reload the product page to apply.";
  });
});
