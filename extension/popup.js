/**
 * NETRA Chrome Extension - Popup Controller
 * Implements ClickUp-style high-contrast design system interactions.
 */

const API_BASE = "http://127.0.0.1:8000";

// DOM Elements
const inputSender = document.getElementById("input-sender");
const inputSubject = document.getElementById("input-subject");
const inputBody = document.getElementById("input-body");
const btnScanEmail = document.getElementById("btn-scan-email");

const btnAutofetchEmail = document.getElementById("btn-autofetch-email");
const autofetchBtnText = document.getElementById("autofetch-btn-text");

const presetSafe = document.getElementById("preset-safe");
const presetPhish = document.getElementById("preset-phish");

// Result Elements
const resultsContainer = document.getElementById("results-container");
const verdictPill = document.getElementById("verdict-pill");
const riskScoreVal = document.getElementById("risk-score-val");
const riskProgressBar = document.getElementById("risk-progress-bar");

const signalTypoStatus = document.getElementById("signal-typo-status");
const signalUrgencyStatus = document.getElementById("signal-urgency-status");
const signalAuthStatus = document.getElementById("signal-auth-status");
const signalUrlsStatus = document.getElementById("signal-urls-status");

const inferenceTimeBadge = document.getElementById("inference-time-badge");

// Explanation Elements
const btnExplain = document.getElementById("btn-explain");
const explainContainer = document.getElementById("explain-container");
const explainSummary = document.getElementById("explain-summary");
const factorBar = document.getElementById("factor-bar");
const factorList = document.getElementById("factor-list");
const tokensPhish = document.getElementById("tokens-phish");
const tokensLegit = document.getElementById("tokens-legit");
const btnHighlight = document.getElementById("btn-highlight");
const btnClearHighlight = document.getElementById("btn-clear-highlight");
const explainNote = document.getElementById("explain-note");

// /explain runs Integrated Gradients: a few seconds on CPU, longer for long emails.
const EXPLAIN_TIMEOUT_MS = 60000;
const FACTOR_COLORS = {
  wording: "var(--color-brand-violet)",
  links: "var(--color-phish-red)",
  sender_auth: "var(--color-warn-amber)",
  reply_to: "var(--color-signal-blue)"
};

let currentActiveTab = null;
let lastExtracted = null;    // email fetched from the open Gmail / Outlook tab
let lastPayload = null;      // payload of the last /predict call
let lastExplanation = null;

// Initialize on load
document.addEventListener("DOMContentLoaded", async () => {
  setupPresets();
  setupEventListeners();
  await checkServerHealth();
  await detectCurrentTab();
});

function setupPresets() {
  presetSafe.addEventListener("click", () => {
    inputSender.value = "sarah.connor@cyberdyne.com";
    inputSubject.value = "Weekly project status update and meeting minutes";
    inputBody.value = "Hi team, please find attached the weekly notes and roadmap review for sprint 14. Next sync will be on Friday at 10 AM.";
  });

  presetPhish.addEventListener("click", () => {
    inputSender.value = "service@paypa1-security.com";
    inputSubject.value = "URGENT: Unauthorized login detected on your PayPal account!";
    inputBody.value = "Dear valued user, an unknown login attempt from Russia was detected. Verify your credentials immediately at http://paypal-security-verification.tk/login or your account will be permanently closed within 24 hours.";
  });
}

function setupEventListeners() {
  btnScanEmail.addEventListener("click", handleEmailScan);
  if (btnAutofetchEmail) {
    btnAutofetchEmail.addEventListener("click", handleAutofetchEmail);
  }
  btnExplain.addEventListener("click", handleExplain);
  btnHighlight.addEventListener("click", handleHighlight);
  btnClearHighlight.addEventListener("click", handleClearHighlight);
}

async function checkServerHealth() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);
    
    await fetch(`${API_BASE}/health`, { signal: controller.signal });
    clearTimeout(timeoutId);
  } catch (err) {
    console.warn("Backend offline or unreachable:", err);
  }
}

async function detectCurrentTab() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab && tab.url) {
      currentActiveTab = tab;
      try {
        const parsed = new URL(tab.url);
        if (parsed.hostname.includes("mail.google.com")) {
          autofetchBtnText.textContent = "Fetch Open Gmail Message";
        } else if (parsed.hostname.includes("outlook")) {
          autofetchBtnText.textContent = "Fetch Open Outlook Message";
        }
      } catch (e) {
        // ignore
      }
    }
  } catch (err) {
    console.warn("Tabs query error:", err);
  }
}

// In-page extraction function executed directly inside the active tab DOM
function extractEmailFromDOM() {
  try {
    const host = window.location.hostname;
    let subject = "";
    let sender = "";
    let bodyText = "";
    const urls = [];
    let source = "Webpage";

    // 1. Gmail
    if (host.includes("mail.google.com")) {
      source = "Gmail";

      // Subject
      const subjEl = document.querySelector("h2.hP") ||
                     document.querySelector("[data-thread-perm-id] h2") ||
                     document.querySelector("h2[tabindex='-1']");
      if (subjEl && subjEl.innerText.trim()) {
        subject = subjEl.innerText.trim();
      } else {
        subject = document.title.replace(/ - Gmail.*$/i, "").replace(/ - [^-]+ - Gmail.*$/i, "").trim();
      }

      // Sender: .gD or elements with email attribute
      const senderEl = document.querySelector("span.gD[email]") ||
                       document.querySelector("span[email]") ||
                       document.querySelector(".gD") ||
                       document.querySelector(".go");
      if (senderEl) {
        sender = senderEl.getAttribute("email") || senderEl.innerText.trim();
      }

      // Body: .a3s is standard for message body
      const bodyEls = document.querySelectorAll(".a3s");
      if (bodyEls && bodyEls.length > 0) {
        const activeBody = bodyEls[bodyEls.length - 1];
        bodyText = activeBody.innerText.trim();
        const links = activeBody.querySelectorAll("a[href]");
        links.forEach(a => {
          if (a.href && (a.href.startsWith("http://") || a.href.startsWith("https://"))) {
            urls.push(a.href);
          }
        });
      }
    }
    // 2. Outlook Web
    else if (host.includes("outlook")) {
      source = "Outlook";
      const subjEl = document.querySelector("[role='heading'][aria-level='2']") ||
                     document.querySelector("[data-testid='messageHeaderSubject']") ||
                     document.querySelector("div[aria-label='Reading Pane'] [title]");
      if (subjEl) subject = (subjEl.getAttribute("title") || subjEl.innerText).trim();

      const senderEl = document.querySelector("[data-testid='SenderPersona'] span[title*='@']") ||
                       document.querySelector("span[title*='@']") ||
                       document.querySelector(".O365_sender");
      if (senderEl) sender = (senderEl.getAttribute("title") || senderEl.innerText).trim();

      const bodyEl = document.querySelector("[aria-label='Message body']") ||
                     document.querySelector(".ItemPartView") ||
                     document.querySelector(".allowTextSelection");
      if (bodyEl) {
        bodyText = bodyEl.innerText.trim();
        const links = bodyEl.querySelectorAll("a[href]");
        links.forEach(a => {
          if (a.href && (a.href.startsWith("http://") || a.href.startsWith("https://"))) {
            urls.push(a.href);
          }
        });
      }
    }
    // 3. Generic fallback
    else {
      source = "Webpage";
      subject = document.title;
      const main = document.querySelector("article") || document.querySelector("main") || document.body;
      bodyText = main ? main.innerText.slice(0, 2000) : "";
    }

    // Selected text priority: If user highlighted any text on screen, prioritize it
    const sel = window.getSelection() ? window.getSelection().toString().trim() : "";
    if (sel && sel.length > 5) {
      bodyText = sel;
    }

    return {
      source,
      found: Boolean(subject || bodyText || sender),
      subject: subject || "",
      sender: sender || "",
      body_text: bodyText || "",
      urls: Array.from(new Set(urls))
    };
  } catch (err) {
    return { source: "Error", found: false, error: err.message };
  }
}

async function handleAutofetchEmail() {
  if (!currentActiveTab || !currentActiveTab.id) {
    alert("No active tab found.");
    return;
  }

  btnAutofetchEmail.disabled = true;
  const originalText = autofetchBtnText.textContent;
  autofetchBtnText.textContent = "Extracting email...";

  try {
    // Direct in-page script execution with a strict 2.5s safety timeout
    const executionPromise = chrome.scripting.executeScript({
      target: { tabId: currentActiveTab.id },
      func: extractEmailFromDOM
    });

    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error("Extraction timed out")), 2500)
    );

    const results = await Promise.race([executionPromise, timeoutPromise]);
    const response = results && results[0] ? results[0].result : null;

    if (response && response.found) {
      lastExtracted = response;
      if (response.sender) inputSender.value = response.sender;
      if (response.subject) inputSubject.value = response.subject;
      if (response.body_text) inputBody.value = response.body_text;

      btnAutofetchEmail.classList.add("success");
      autofetchBtnText.textContent = `✓ Fetched from ${response.source}!`;
      setTimeout(() => {
        btnAutofetchEmail.classList.remove("success");
        autofetchBtnText.textContent = originalText;
      }, 2500);
    } else {
      alert("No email content could be detected. If you are on Gmail or Outlook, make sure an email thread is opened.");
      autofetchBtnText.textContent = originalText;
    }
  } catch (err) {
    console.error("Auto-fetch error:", err);
    alert("Could not extract email automatically.\nTip: Open an email in Gmail or highlight text on the page, then click Auto-Fetch.");
    autofetchBtnText.textContent = originalText;
  } finally {
    btnAutofetchEmail.disabled = false;
  }
}

async function handleEmailScan() {
  const bodyText = inputBody.value.trim();
  const subject = inputSubject.value.trim();
  const sender = inputSender.value.trim();

  if (!bodyText && !subject && !sender) {
    alert("Please enter email body, subject, or sender to analyze.");
    return;
  }

  const payload = {
    subject: subject,
    body_text: bodyText || subject,
    sender: sender,
    // Links found in the fetched email, as long as the user has not replaced its body.
    urls: fetchedEmailUnchanged() ? (lastExtracted.urls || []).slice(0, 50) : []
  };

  btnScanEmail.disabled = true;
  btnScanEmail.textContent = "Analyzing...";

  try {
    await runInference(payload);
  } finally {
    btnScanEmail.disabled = false;
    btnScanEmail.textContent = "Analyze Email";
  }
}

async function runInference(payload) {
  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      throw new Error(`Inference error: HTTP ${res.status}`);
    }

    const data = await res.json();
    lastPayload = payload;
    resetExplanation();
    displayResults(data);
  } catch (err) {
    alert(`Failed to connect to NETRA backend (${API_BASE}).\nPlease ensure "python -m uvicorn api.main:app --port 8000" is running.`);
  }
}

function displayResults(data) {
  resultsContainer.classList.remove("hidden");

  // Verdict Pill
  const verdict = (data.classification || "UNKNOWN").toUpperCase();
  verdictPill.textContent = verdict;
  verdictPill.className = "verdict-pill";

  if (verdict === "LEGITIMATE") {
    verdictPill.classList.add("pill-legit");
    riskProgressBar.style.backgroundColor = "var(--color-emerald)";
  } else if (verdict === "SUSPICIOUS") {
    verdictPill.classList.add("pill-suspicious");
    riskProgressBar.style.backgroundColor = "var(--color-warn-amber)";
  } else {
    verdictPill.classList.add("pill-phish");
    riskProgressBar.style.backgroundColor = "var(--color-phish-red)";
  }

  // Risk Score & Progress Bar
  const score = data.risk_score !== undefined ? data.risk_score : 0;
  const pct = Math.min(100, Math.max(0, (score * 100))).toFixed(1);
  riskScoreVal.textContent = `${pct}%`;
  riskProgressBar.style.width = `${pct}%`;

  // Signals
  const signals = data.signals || {};

  // Typosquatting
  if (signals.typosquatting_detected) {
    signalTypoStatus.textContent = "Detected";
    signalTypoStatus.className = "signal-status flagged";
  } else {
    signalTypoStatus.textContent = "Clean";
    signalTypoStatus.className = "signal-status";
  }

  // Urgency (a payment / credential request under pressure is shown here too)
  if (signals.bec_detected) {
    signalUrgencyStatus.textContent = "Payment Request";
    signalUrgencyStatus.className = "signal-status flagged";
  } else if (signals.urgency_detected) {
    signalUrgencyStatus.textContent = "High Urgency";
    signalUrgencyStatus.className = "signal-status flagged";
  } else {
    signalUrgencyStatus.textContent = "Normal";
    signalUrgencyStatus.className = "signal-status";
  }

  // Header Auth
  if (signals.header_auth_failed) {
    signalAuthStatus.textContent = "Failed";
    signalAuthStatus.className = "signal-status flagged";
  } else {
    signalAuthStatus.textContent = "Passed / None";
    signalAuthStatus.className = "signal-status";
  }

  // Suspicious URLs (plain http plus the link checks added after the Tier-1 audit)
  const suspUrls = (signals.suspicious_urls || 0) + (signals.ip_url_detected ? 1 : 0) +
    (signals.shortened_url_detected ? 1 : 0) + (signals.executable_link ? 1 : 0) +
    (signals.risky_file_link ? 1 : 0);
  if (suspUrls > 0) {
    signalUrlsStatus.textContent = `${suspUrls} Flagged`;
    signalUrlsStatus.className = "signal-status flagged";
  } else {
    signalUrlsStatus.textContent = "0 Detected";
    signalUrlsStatus.className = "signal-status";
  }

  // Metadata
  const timeTextEl = document.getElementById("inference-time-text");
  if (timeTextEl) {
    timeTextEl.textContent = `${(data.processing_time_ms || 0).toFixed(1)} ms`;
  }

  // Rule layer: say when it changed the model verdict (textContent only).
  const adjustmentNote = document.getElementById("adjustment-note");
  if (adjustmentNote) {
    const modelPct = ((data.model_score || 0) * 100).toFixed(1);
    let text = "";
    if (data.adjustment === "trusted_sender") {
      text = `Model alone: ${modelPct}%. Lowered by the sender check: ${data.adjustment_reason || "trusted sender"}.`;
    } else if (data.adjustment === "signal_override") {
      text = `Model alone: ${modelPct}%. Raised by ${data.adjustment_reason || "threat signals"}.`;
    }
    adjustmentNote.textContent = text;
    adjustmentNote.classList.toggle("hidden", !text);
  }

  // Smooth scroll to results
  resultsContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ---------------------------------------------------------------------------
// Explanation (POST /explain)
// ---------------------------------------------------------------------------
function fetchedEmailUnchanged() {
  return Boolean(lastExtracted && lastExtracted.body_text &&
                 inputBody.value.trim() === lastExtracted.body_text.trim());
}

function resetExplanation() {
  lastExplanation = null;
  explainContainer.classList.add("hidden");
  btnExplain.disabled = false;
  btnExplain.textContent = "Why this verdict?";
}

async function handleExplain() {
  if (!lastPayload) return;
  btnExplain.disabled = true;
  btnExplain.textContent = "Explaining… (a few seconds)";

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), EXPLAIN_TIMEOUT_MS);
  try {
    const res = await fetch(`${API_BASE}/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...lastPayload, top_k: 8 }),
      signal: controller.signal
    });
    if (res.status === 429) {
      throw new Error("Too many explanations this minute. Try again shortly.");
    }
    if (!res.ok) {
      throw new Error(`Explain error: HTTP ${res.status}`);
    }
    const data = await res.json();
    lastExplanation = data.explanation;
    renderExplanation(data.explanation);
    btnExplain.textContent = "Explanation";
  } catch (err) {
    btnExplain.disabled = false;
    btnExplain.textContent = "Why this verdict?";
    alert(err.name === "AbortError" ? "The explanation took too long." : err.message);
  } finally {
    clearTimeout(timeoutId);
  }
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatPoints(contribution) {
  const pts = Math.abs(contribution * 100);
  const sign = contribution >= 0 ? "+" : "−";
  return `${sign}${pts < 0.1 ? pts.toFixed(2) : pts.toFixed(1)} pts`;
}

function renderExplanation(x) {
  explainContainer.classList.remove("hidden");
  explainSummary.textContent = x.summary;

  factorBar.replaceChildren();
  factorList.replaceChildren();
  for (const f of x.threat_factors) {
    const color = FACTOR_COLORS[f.key] || "var(--color-slate)";
    if (f.share) {
      const seg = el("span");
      seg.style.width = `${f.share}%`;
      seg.style.background = color;
      seg.title = `${f.label}: ${f.share.toFixed(0)}%`;
      factorBar.appendChild(seg);
    }

    const row = el("li", "factor-row");
    const head = el("div", "factor-head");
    const swatch = el("span", "factor-swatch");
    swatch.style.background = color;
    head.append(swatch, el("span", "", f.label));
    const badgeText = f.share ? `${f.share.toFixed(0)}%` : formatPoints(f.contribution);
    head.appendChild(el("span", f.contribution < 0 ? "factor-badge lowers" : "factor-badge", badgeText));
    row.appendChild(head);
    if (f.evidence && f.evidence.length) {
      row.appendChild(el("div", "factor-evidence", f.evidence.join(" · ")));
    }
    factorList.appendChild(row);
  }

  renderTokens(tokensPhish, x.top_phishing_tokens, "220, 38, 38");
  renderTokens(tokensLegit, x.top_legitimate_tokens, "16, 185, 129");

  const notes = [];
  if (x.override === "trusted_sender") {
    notes.push("Verdict lowered: trusted sender, links match the sender's domain.");
  } else if (x.override) {
    notes.push("Verdict raised by the multi-signal rule.");
  }
  if (x.truncated) notes.push("Only the first 256 tokens were analysed.");
  if (!x.converged) notes.push("Approximate: attribution did not fully converge.");
  if (x.method === "signals_only") notes.push("Fallback model: only rule signals are explained.");
  notes.push(`${(x.explain_time_ms / 1000).toFixed(1)} s`);
  explainNote.textContent = notes.join(" ");

  const canHighlight = fetchedEmailUnchanged() &&
    ["Gmail", "Outlook"].includes(lastExtracted.source) &&
    highlightPayload().length > 0;
  btnHighlight.disabled = !canHighlight;
  btnClearHighlight.disabled = !canHighlight;
  btnHighlight.title = canHighlight ? "" : "Fetch the open Gmail / Outlook email (unchanged) to highlight it";

  explainContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderTokens(container, tokens, rgb) {
  container.replaceChildren();
  if (!tokens.length) {
    container.appendChild(el("span", "token-empty", "None"));
    return;
  }
  const max = Math.max(...tokens.map(t => Math.abs(t.attribution)));
  for (const t of tokens) {
    const alpha = 0.15 + 0.6 * Math.abs(t.attribution) / (max || 1);
    const chip = el("span", "token-chip", t.occurrences > 1 ? `${t.text} ×${t.occurrences}` : t.text);
    chip.style.background = `rgba(${rgb}, ${alpha.toFixed(2)})`;
    chip.title = `${t.segment} · ${t.attribution >= 0 ? "+" : ""}${t.attribution.toFixed(3)}`;
    container.appendChild(chip);
  }
}

// Body words only: subject and URL text are not inside the message body element.
function highlightPayload() {
  if (!lastExplanation) return [];
  const pick = (list) => list.filter(t => t.segment === "body")
    .map(t => ({ text: t.text, attribution: t.attribution }));
  return [...pick(lastExplanation.top_phishing_tokens), ...pick(lastExplanation.top_legitimate_tokens)];
}

async function sendToPage(message) {
  try {
    return await chrome.tabs.sendMessage(currentActiveTab.id, message);
  } catch (err) {
    // Content script not present (tab opened before the extension loaded): inject it once, retry.
    await chrome.scripting.executeScript({ target: { tabId: currentActiveTab.id }, files: ["content.js"] });
    return await chrome.tabs.sendMessage(currentActiveTab.id, message);
  }
}

async function handleHighlight() {
  if (!lastExplanation || !currentActiveTab) return;
  try {
    const res = await sendToPage({ action: "highlightTokens", tokens: highlightPayload() });
    explainNote.textContent = res && res.found
      ? `Highlighted ${res.marked} word${res.marked === 1 ? "" : "s"} in the open email.`
      : "Could not find the open email body on the page.";
  } catch (err) {
    explainNote.textContent = "Could not highlight on this page.";
  }
}

async function handleClearHighlight() {
  if (!currentActiveTab) return;
  try {
    await sendToPage({ action: "clearHighlights" });
  } catch (err) {
    // nothing to clear
  }
}
