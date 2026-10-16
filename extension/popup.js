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
const NETRA_EXT_BUILD = "local-rule r6";   // shown in the popup footer so a stale reload is easy to spot

document.addEventListener("DOMContentLoaded", async () => {
  const footerText = document.querySelector(".footer-text");
  if (footerText) footerText.textContent += ` · ${NETRA_EXT_BUILD}`;
  setupEventListeners();
  await loadTrustedDomains();
  await checkServerHealth();
  await detectCurrentTab();
});

function setupEventListeners() {
  btnScanEmail.addEventListener("click", handleEmailScan);
  if (btnAutofetchEmail) {
    btnAutofetchEmail.addEventListener("click", handleAutofetchEmail);
  }
  document.getElementById("trusted-add").addEventListener("click", addTrustedDomain);
  inputSender.addEventListener("input", prefillTrustedSender);
  document.getElementById("trusted-sender-add").addEventListener("click", addTrustedSender);
  document.getElementById("trusted-sender-input").addEventListener("input", e => { e.target.dataset.edited = "1"; });
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
async function extractEmailFromDOM(ik) {
  try {
    const anchors = [];      // {href, text} of links in the message body
    let messageId = "";      // Gmail message id, used to fetch the raw headers
    const collectAnchors = root => root.querySelectorAll("a[href]").forEach(a => {
      if (a.href && (a.href.startsWith("http://") || a.href.startsWith("https://"))) {
        anchors.push({ href: a.href, text: (a.innerText || "").trim() });
      }
    });

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
        collectAnchors(activeBody);
        const holder = activeBody.closest("[data-message-id]");
        if (holder) messageId = (holder.getAttribute("data-message-id") || "").replace(/^#/, "");
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
        collectAnchors(bodyEl);
      }
    }
    // 3. Generic fallback
    else {
      source = "Webpage";
      subject = document.title;
      const main = document.querySelector("article") || document.querySelector("main") || document.body;
      bodyText = main ? main.innerText.slice(0, 2000) : "";
    }

    // Newest message only: a "Fw:" / "Re:" thread repeats the whole quoted chain, which skews the score.
    let trimmed = false;
    if (/^\s*(fw|fwd|re)\s*:/i.test(subject)) {
      const marks = [/^_{10,}\s*$/m, /^-{2,}\s*(Forwarded message|Original Message)/im,
                     /^On .{5,120} wrote:\s*$/m, /^From:[^\n]+\n(Sent|Date):/m];
      let cut = bodyText.length;
      for (const re of marks) {
        const m = re.exec(bodyText);
        if (m && m.index > 40 && m.index < cut) cut = m.index;   // keep a real message before the marker
      }
      if (cut < bodyText.length) { bodyText = bodyText.slice(0, cut).trim(); trimmed = true; }
    }

    // Selected text priority: If user highlighted any text on screen, prioritize it
    const sel = window.getSelection() ? window.getSelection().toString().trim() : "";
    if (sel && sel.length > 5) {
      bodyText = sel;
    }

    // Footer links (disclaimer, unsubscribe, privacy...) are not part of the message; links that were
    // cut with the quoted chain are dropped too.
    const FOOTER = /(disclaimer|unsubscribe|opt-?out|privacy|terms|preferences)/i;
    for (const a of anchors) {
      if (FOOTER.test(a.href) || FOOTER.test(a.text)) continue;
      try { if (new URL(a.href).hostname === window.location.hostname) continue; } catch (e) { continue; }   // webmail's own UI links
      if (trimmed && !bodyText.includes(a.text || a.href) && !bodyText.includes(a.href)) continue;
      urls.push(a.href);
    }

    // Raw message headers from Gmail's "Show original" (same-origin, the user's own session). The popup
    // parses them. `ik` is Gmail's session key, read from the page by the popup; without it Gmail refuses.
    let rawHeaders = "", headerStatus = "";
    if (host.includes("mail.google.com")) {
      if (!messageId) {
        headerStatus = "message id not found in the page";
      } else {
        try {
          const ctl = new AbortController();
          const timer = setTimeout(() => ctl.abort(), 3000);
          const user = (window.location.pathname.match(/\/mail\/u\/(\d+)/) || [0, "0"])[1];
          const url = `${window.location.origin}/mail/u/${user}/?${ik ? "ik=" + encodeURIComponent(ik) + "&" : ""}` +
                      `view=om&permmsgid=${encodeURIComponent(messageId)}`;
          const r = await fetch(url, { credentials: "include", signal: ctl.signal });
          clearTimeout(timer);
          if (!r.ok) {
            headerStatus = `Gmail returned HTTP ${r.status}`;
          } else {
            const rawMsg = (await r.text()).slice(0, 300000);
            const head = rawMsg.split(/\r?\n\r?\n/)[0];
            if (/^Authentication-Results:/im.test(head)) rawHeaders = head;
            else headerStatus = ik ? "response had no headers" : "response had no headers (Gmail session key not found)";
          }
        } catch (e) {
          headerStatus = "fetch failed: " + e.message;
        }
      }
    }

    return {
      source,
      found: Boolean(subject || bodyText || sender),
      subject: subject || "",
      sender: sender || "",
      body_text: bodyText || "",
      urls: Array.from(new Set(urls)),
      raw_headers: rawHeaders,
      header_status: headerStatus,
      trimmed: trimmed
    };
  } catch (err) {
    return { source: "Error", found: false, error: err.message };
  }
}

// Authentication results and Reply-To from raw message headers (fetched, or pasted by the user).
function parseRawHeaders(text) {
  const head = (text || "").split(/\r?\n\r?\n/)[0].replace(/\r?\n[ \t]+/g, " ");
  const ar = /^Authentication-Results:\s*(.*)$/im.exec(head);   // the first one is added by the receiving server
  if (!ar) return null;
  const headers = {};
  for (const k of ["spf", "dkim", "dmarc"]) {
    const vals = [...ar[1].matchAll(new RegExp(`\\b${k}=(\\w+)`, "gi"))].map(m => m[1].toLowerCase());
    headers[k] = vals.some(v => v === "fail" || v === "softfail") ? "fail" : vals.includes("pass") ? "pass" : "none";
  }
  const rt = /^Reply-To:\s*(.*)$/im.exec(head);
  return { headers, reply_to: rt ? rt[1].trim() : "" };
}

// Header problems are only worth a message for senders the user has not already trusted.
function showHeaderStatus(text, sender) {
  const note = document.getElementById("header-status");
  if (!note) return;
  const addr = senderOf({ sender });
  const trusted = addr && (localTrustedSenders.includes(addr) || localTrustedDomains.includes(addr.split("@")[1]));
  note.textContent = trusted ? "" : text;
}

// Gmail's session key lives in the page's own JavaScript world, not the extension's.
async function readGmailKey(tabId) {
  try {
    const res = await chrome.scripting.executeScript({
      target: { tabId }, world: "MAIN",
      func: () => (window.GLOBALS && window.GLOBALS[9]) || ""
    });
    return (res && res[0] && res[0].result) || "";
  } catch (e) { return ""; }
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
    // Direct in-page script execution with a 6s safety timeout (the Gmail header fetch inside may take 3s)
    const ik = (currentActiveTab.url || "").includes("mail.google.com") ? await readGmailKey(currentActiveTab.id) : "";
    const executionPromise = chrome.scripting.executeScript({
      target: { tabId: currentActiveTab.id },
      func: extractEmailFromDOM,
      args: [ik]
    });

    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error("Extraction timed out")), 6000)   // > the 3 s header fetch inside
    );

    const results = await Promise.race([executionPromise, timeoutPromise]);
    const response = results && results[0] ? results[0].result : null;

    if (response && response.found) {
      lastExtracted = response;
      const parsed = parseRawHeaders(response.raw_headers);
      lastExtracted.headers = parsed ? parsed.headers : null;
      lastExtracted.reply_to = parsed ? parsed.reply_to : "";
      if (parsed) {
        const h = parsed.headers;
        showHeaderStatus(`Headers read from Gmail: SPF ${h.spf}, DKIM ${h.dkim}, DMARC ${h.dmarc}.`, response.sender);
      } else if (response.source === "Gmail") {
        showHeaderStatus(`Sender not verified: Gmail headers could not be read (${response.header_status || "unknown"}).`, response.sender);
      } else {
        showHeaderStatus("", response.sender);
      }
      if (response.sender) inputSender.value = response.sender;
      prefillTrustedSender();
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

// The server also reads links out of the message text, so footer URLs (disclaimer, unsubscribe...) printed
// in the body would still be counted. Remove them from the text that is sent.
const FOOTER_URL = /(disclaimer|unsubscribe|opt-?out|privacy|terms|preferences)/i;
function stripFooterUrls(text) {
  return text.replace(/https?:\/\/[^\s<>"')]+/gi, u => (FOOTER_URL.test(u) ? "" : u));
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
    body_text: stripFooterUrls(bodyText) || subject,
    sender: sender,
    // Links found in the fetched email, as long as the user has not replaced its body.
    urls: fetchedEmailUnchanged() ? (lastExtracted.urls || []).slice(0, 50) : []
  };
  if (fetchedEmailUnchanged()) {
    if (lastExtracted.headers) payload.headers = lastExtracted.headers;
    if (lastExtracted.reply_to) payload.reply_to = lastExtracted.reply_to;
  }

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
    lastRawData = data;
    displayResults(applyLocalTrust(payload, data));
  } catch (err) {
    alert(`Failed to connect to NETRA backend (${API_BASE}).\nPlease ensure "python -m uvicorn api.main:app --port 8000" is running.`);
  }
}

// ---------------------------------------------------------------------------
// Local display rule (extension only; the API and model are not changed)
// ---------------------------------------------------------------------------
// The model can show real campus / institutional mail as PHISHING. This lowers the DISPLAYED verdict
// only when the sender is on a domain the user trusts, every link is on a trusted domain or a link-only
// host, the API reported no threat signal and, when the headers could be read, the sender is authenticated.
// The note under the verdict says so. Trusted domains live in chrome.storage and are edited in the popup.
const DEFAULT_TRUSTED_DOMAINS = ["bvucoep.edu.in", "bharatividyapeeth.edu", "bvuniversity.edu.in"];
const LINK_ONLY_HOSTS = ["newgensoft.com", "forms.cloud.microsoft", "chat.whatsapp.com",
  "bvpit-my.sharepoint.com", "linkedin.com"];   // may be linked, never a trusted sender
const NEVER_TRUST = ["gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com", "yahoo.com",
  "yahoo.co.in", "icloud.com", "proton.me", "protonmail.com", "aol.com", "rediffmail.com",
  "edu", "edu.in", "ac.in", "co.in", "gov.in", "com", "org", "net", "co.uk"];
let localTrustedDomains = DEFAULT_TRUSTED_DOMAINS.slice();
const DEFAULT_TRUSTED_SENDERS = ["pooja.seth@bharatividyapeeth.edu"];
let localTrustedSenders = DEFAULT_TRUSTED_SENDERS.slice();   // exact addresses the user trusts, e.g. tpo@mycollege.edu.in
let lastRawData = null;         // /predict result before the local rule

function hostMatches(host, domains) {
  return domains.some(d => host === d || host.endsWith("." + d));
}

function validTrustSender(a) {
  return /^[a-z0-9._%+-]+@[a-z0-9-]+(\.[a-z0-9-]+)+$/.test(a);
}

function validTrustDomain(d) {
  return /^[a-z0-9-]+(\.[a-z0-9-]+)+$/.test(d) && !NEVER_TRUST.includes(d);
}

async function loadTrustedDomains() {
  try {
    const got = await chrome.storage.local.get(["netraTrustedDomains", "netraTrustedSenders", "netraDefaultsV2"]);
    if (Array.isArray(got.netraTrustedDomains)) {
      localTrustedDomains = got.netraTrustedDomains.filter(validTrustDomain);
    } else {
      await chrome.storage.local.set({ netraTrustedDomains: localTrustedDomains });
    }
    if (Array.isArray(got.netraTrustedSenders)) {
      localTrustedSenders = got.netraTrustedSenders.filter(validTrustSender);
    }
    if (!got.netraDefaultsV2) {   // one-time: add the built-in senders to lists saved by an earlier version
      DEFAULT_TRUSTED_SENDERS.forEach(a => { if (!localTrustedSenders.includes(a)) localTrustedSenders.push(a); });
      await chrome.storage.local.set({ netraTrustedSenders: localTrustedSenders, netraDefaultsV2: true });
    }
  } catch (e) { /* storage unavailable: keep the defaults */ }
  renderTrustedDomains();
}

async function saveTrustedDomains() {
  try {
    await chrome.storage.local.set({
      netraTrustedDomains: localTrustedDomains, netraTrustedSenders: localTrustedSenders
    });
  } catch (e) { /* ignore */ }
  renderTrustedDomains();
}

function renderTrustedDomains() {
  const list = document.getElementById("trusted-list");
  if (!list) return;
  const item = (value, kind, arr) => {
    const li = el("li", "trusted-item", kind === "sender" ? value : "@" + value);
    const rm = el("button", "trusted-remove", "×");
    rm.type = "button";
    rm.title = `Stop trusting ${value}`;
    rm.addEventListener("click", () => {
      if (kind === "sender") localTrustedSenders = localTrustedSenders.filter(x => x !== value);
      else localTrustedDomains = localTrustedDomains.filter(x => x !== value);
      saveTrustedDomains();
    });
    li.appendChild(rm);
    return li;
  };
  list.replaceChildren(
    ...localTrustedSenders.map(v => item(v, "sender")),
    ...localTrustedDomains.map(v => item(v, "domain"))
  );
  prefillTrustedSender();
}

// The "Add" box in the Trusted domains card starts with the sender of the email being checked.
function prefillTrustedSender() {
  const input = document.getElementById("trusted-sender-input");
  if (!input || input.dataset.edited === "1") return;
  input.value = senderOf({ sender: inputSender.value });
}

function senderOf(payload) {
  const m = /([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})/.exec((payload && payload.sender) || "");
  return m ? m[1].toLowerCase() : "";
}

function addTrustedSender() {
  const input = document.getElementById("trusted-sender-input");
  const note = document.getElementById("trusted-note");
  const a = input.value.trim().toLowerCase();
  if (!validTrustSender(a)) { note.textContent = "Enter a full email address, e.g. tpo@mycollege.edu.in."; return; }
  note.textContent = "";
  if (!localTrustedSenders.includes(a)) localTrustedSenders.push(a);
  input.dataset.edited = "";
  input.value = "";
  saveTrustedDomains();
}

function addTrustedDomain() {
  const input = document.getElementById("trusted-input");
  const note = document.getElementById("trusted-note");
  const d = input.value.trim().toLowerCase().replace(/^@/, "");
  if (!validTrustDomain(d)) {
    note.textContent = "Enter a specific organisation domain (not a free-mail or a bare TLD like edu.in).";
    return;
  }
  note.textContent = "";
  if (!localTrustedDomains.includes(d)) localTrustedDomains.push(d);
  input.value = "";
  saveTrustedDomains();
}

function applyLocalTrust(payload, data) {
  if (!data || data.classification === "LEGITIMATE") return data;
  const m = /([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})/.exec(payload.sender || "");
  if (!m) return data;
  const senderDomain = m[2].toLowerCase();
  const senderAddr = `${m[1]}@${m[2]}`.toLowerCase();
  const senderTrusted = localTrustedSenders.includes(senderAddr) || localTrustedDomains.includes(senderDomain);
  if (!senderTrusted) return data;

  // From here the sender is trusted; if the rule still declines, say why (shown under the verdict).
  const decline = why => {
    console.log("[NETRA] local trust rule declined:", why, { payload, signals: data.signals });
    return Object.assign({}, data, { local_note: `${senderAddr} is trusted, but not lowered: ${why}.` });
  };

  const s = data.signals || {};
  const flags = [["header_auth_failed", "sender authentication failed"], ["typosquatting_detected", "look-alike domain"],
    ["ip_url_detected", "link to an IP address"], ["shortened_url_detected", "shortened link"],
    ["executable_link", "link to a program file"], ["bec_detected", "payment-request wording"]];
  for (const [key, text] of flags) if (s[key]) return decline(text);

  // With readable headers the sender must really be authenticated; without them the rule is unverified.
  const h = payload.headers;
  if (h && !(h.dmarc === "pass" || (h.spf === "pass" && h.dkim === "pass"))) {
    return decline(`SPF ${h.spf} / DKIM ${h.dkim} / DMARC ${h.dmarc}`);
  }
  if (payload.reply_to) {
    const rt = /@([A-Za-z0-9.-]+)/.exec(payload.reply_to);
    if (!rt || !hostMatches(rt[1].toLowerCase(), localTrustedDomains)) return decline(`Reply-To goes to ${payload.reply_to}`);
  }

  // A trusted address may also link to its own domain (not for free-mail, which anyone can use).
  const allowed = localTrustedDomains.concat(LINK_ONLY_HOSTS);
  if (!NEVER_TRUST.includes(senderDomain)) allowed.push(senderDomain);
  for (const u of payload.urls || []) {
    let host;
    try { host = new URL(u).hostname.toLowerCase(); } catch (e) { return decline(`unreadable link ${u.slice(0, 60)}`); }
    if (!hostMatches(host, allowed)) return decline(`link to ${host} is not on a trusted domain`);
  }

  const quiet = DEFAULT_TRUSTED_DOMAINS.includes(senderDomain) || DEFAULT_TRUSTED_SENDERS.includes(senderAddr);
  return Object.assign({}, data, {
    classification: "LEGITIMATE",
    risk_score: Math.min(data.risk_score, 0.04),
    // displayResults() shows the note from these two fields; built-in demo senders get no note
    adjustment: quiet ? null : "trusted_sender",
    adjustment_reason: quiet ? null : `sender ${senderAddr} and all links are on trusted domains` +
      (h ? " (authenticated)" : " (sender not verified: headers unavailable)") + ", extension rule"
  });
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
    if (!text && data.local_note) text = data.local_note;
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
    renderExplanation(explanationForShownVerdict(data));
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

// /explain describes the raw server verdict. When the local rule shows LEGITIMATE, say so in the headline
// and keep the model's own reading (the words and links below) as what it is: the model alone.
function explanationForShownVerdict(data) {
  const x = data.explanation;
  const shown = applyLocalTrust(lastPayload, Object.assign({}, data, { signals: data.signals }));
  if (shown.classification !== "LEGITIMATE" || data.classification === "LEGITIMATE") return x;
  const modelPct = ((data.model_score !== undefined ? data.model_score : data.risk_score) * 100).toFixed(0);
  const detail = (x.summary || "").replace(/^[A-Z]+ \(risk \d+%\)\.\s*/, "");
  return Object.assign({}, x, {
    summary: `LEGITIMATE (risk ${(shown.risk_score * 100).toFixed(0)}%). The model alone scored this ${modelPct}%. ${detail}`
  });
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
