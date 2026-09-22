/**
 * NETRA Chrome Extension - Background Service Worker (Manifest V3)
 */

const API_BASE = "http://127.0.0.1:8000";

chrome.runtime.onInstalled.addListener(() => {
  // Create context menus for quick scanning
  chrome.contextMenus.create({
    id: "netra-scan-page",
    title: "Scan Page URL with NETRA",
    contexts: ["page", "link"]
  });

  chrome.contextMenus.create({
    id: "netra-scan-selection",
    title: "Analyze Text with NETRA",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "netra-scan-page") {
    const targetUrl = info.linkUrl || info.pageUrl || (tab && tab.url);
    if (!targetUrl) return;

    await scanAndNotify({
      subject: `Context Scan: ${targetUrl}`,
      body_text: `Target URL: ${targetUrl}`,
      urls: [targetUrl],
      sender: ""
    }, tab);
  } else if (info.menuItemId === "netra-scan-selection") {
    const selectedText = info.selectionText;
    if (!selectedText) return;

    await scanAndNotify({
      subject: "Selected Text Scan",
      body_text: selectedText,
      urls: [],
      sender: ""
    }, tab);
  }
});

async function scanAndNotify(payload, tab) {
  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) return;
    const data = await res.json();
    const verdict = (data.classification || "UNKNOWN").toUpperCase();

    // Update extension badge on tab
    if (tab && tab.id) {
      if (verdict === "LEGITIMATE") {
        await chrome.action.setBadgeText({ text: "SAFE", tabId: tab.id });
        await chrome.action.setBadgeBackgroundColor({ color: "#00c07a", tabId: tab.id });
      } else if (verdict === "SUSPICIOUS") {
        await chrome.action.setBadgeText({ text: "WARN", tabId: tab.id });
        await chrome.action.setBadgeBackgroundColor({ color: "#fd9a46", tabId: tab.id });
      } else {
        await chrome.action.setBadgeText({ text: "PHISH", tabId: tab.id });
        await chrome.action.setBadgeBackgroundColor({ color: "#fc6d7b", tabId: tab.id });
      }
    }
  } catch (err) {
    console.error("[NETRA Service Worker] Failed to scan:", err);
  }
}
