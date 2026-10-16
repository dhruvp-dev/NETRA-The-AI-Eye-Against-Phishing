/**
 * NETRA Extension - Content Script
 * Intelligently extracts email data from webmail clients (Gmail, Outlook, etc.)
 * and assists the popup scanner.
 */

// The popup may inject this file again into a tab that already has it; register once.
if (!window.__netraContentLoaded) {
  window.__netraContentLoaded = true;

  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "getPageInfo") {
      sendResponse({
        title: document.title,
        url: window.location.href,
        domain: window.location.hostname
      });
      return true;
    }

    if (request.action === "extractEmail") {
      const data = extractActiveEmail();
      sendResponse(data);
      return true;
    }

    if (request.action === "highlightTokens") {
      sendResponse(highlightTokens(request.tokens || []));
      return true;
    }

    if (request.action === "clearHighlights") {
      sendResponse({ cleared: clearHighlights() });
      return true;
    }

    return true;
  });
}

// ---------------------------------------------------------------------------
// Explanation highlights (words from POST /explain, marked in the open email)
// ---------------------------------------------------------------------------
var NETRA_MAX_MARKS = 200;   // var: the file may be injected twice into one page

function activeEmailBody() {
  const host = window.location.hostname;
  if (host.includes("mail.google.com")) {
    const bodies = document.querySelectorAll(".a3s.aiL");
    const all = bodies.length ? bodies : document.querySelectorAll(".a3s");
    return all.length ? all[all.length - 1] : null;
  }
  if (host.includes("outlook")) {
    return document.querySelector("[aria-label='Message body']") ||
           document.querySelector(".ItemPartView") ||
           document.querySelector(".allowTextSelection");
  }
  return null;
}

function ensureHighlightStyle() {
  if (document.getElementById("netra-hl-style")) return;
  const style = document.createElement("style");
  style.id = "netra-hl-style";
  style.textContent =
    "mark.netra-hl{color:inherit;border-radius:3px;padding:0 1px;}" +
    "mark.netra-hl-phish{box-shadow:inset 0 -2px 0 rgba(220,38,38,.9);}" +
    "mark.netra-hl-legit{box-shadow:inset 0 -2px 0 rgba(16,185,129,.9);}";
  document.head.appendChild(style);
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlightTokens(tokens) {
  const root = activeEmailBody();
  if (!root) return { found: false, marked: 0 };
  clearHighlights();
  ensureHighlightStyle();

  const words = tokens.filter(t => t && t.text && /[\p{L}\p{N}]/u.test(t.text));
  if (!words.length) return { found: true, marked: 0 };
  const max = Math.max(...words.map(t => Math.abs(t.attribution))) || 1;
  const byWord = new Map(words.map(t => [t.text.toLowerCase(), t]));
  // Longest first so "verification" wins over "verify"; whole words only.
  const alternatives = [...byWord.keys()].sort((a, b) => b.length - a.length).map(escapeRegExp);
  const pattern = new RegExp(`(?<![\\p{L}\\p{N}])(${alternatives.join("|")})(?![\\p{L}\\p{N}])`, "giu");

  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = node.parentElement;
      if (!parent || parent.closest("script, style, mark.netra-hl")) return NodeFilter.FILTER_REJECT;
      return node.nodeValue.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
    }
  });
  const textNodes = [];
  while (walker.nextNode()) textNodes.push(walker.currentNode);

  let marked = 0;
  for (const node of textNodes) {
    if (marked >= NETRA_MAX_MARKS) break;
    const matches = [...node.nodeValue.matchAll(pattern)];
    // Wrap from the end so earlier match offsets stay valid after splitText.
    for (let i = matches.length - 1; i >= 0 && marked < NETRA_MAX_MARKS; i--) {
      const m = matches[i];
      const token = byWord.get(m[0].toLowerCase());
      if (!token) continue;
      const target = node.splitText(m.index);
      target.splitText(m[0].length);
      const mark = document.createElement("mark");
      const phish = token.attribution > 0;
      const alpha = 0.15 + 0.6 * Math.abs(token.attribution) / max;
      mark.className = `netra-hl ${phish ? "netra-hl-phish" : "netra-hl-legit"}`;
      mark.style.backgroundColor = phish ? `rgba(220, 38, 38, ${alpha.toFixed(2)})` : `rgba(16, 185, 129, ${alpha.toFixed(2)})`;
      mark.title = `NETRA: ${phish ? "raises" : "lowers"} phishing risk (${token.attribution >= 0 ? "+" : ""}${token.attribution.toFixed(3)})`;
      target.parentNode.insertBefore(mark, target);
      mark.appendChild(target);
      marked++;
    }
  }
  return { found: true, marked };
}

function clearHighlights() {
  const marks = document.querySelectorAll("mark.netra-hl");
  const parents = new Set();
  marks.forEach(mark => {
    const parent = mark.parentNode;
    while (mark.firstChild) parent.insertBefore(mark.firstChild, mark);
    parent.removeChild(mark);
    parents.add(parent);
  });
  parents.forEach(p => p.normalize());
  return marks.length;
}

function extractActiveEmail() {
  const host = window.location.hostname;

  // 1. Gmail extraction
  if (host.includes("mail.google.com")) {
    return extractFromGmail();
  }

  // 2. Outlook / Office 365 extraction
  if (host.includes("outlook.live.com") || host.includes("outlook.office.com") || host.includes("outlook.office365.com")) {
    return extractFromOutlook();
  }

  // 3. Generic fallback (extract selected text or visible page content)
  return extractGeneric();
}

function extractFromGmail() {
  let subject = "";
  let sender = "";
  let bodyText = "";
  const urls = [];

  // Subject: h2.hP is the standard Gmail subject line class
  const subjectEl = document.querySelector("h2.hP") || 
                    document.querySelector("[data-thread-perm-id] h2") ||
                    document.querySelector("h2[tabindex='-1']");
  if (subjectEl) {
    subject = subjectEl.innerText.trim();
  } else {
    subject = document.title.replace(/ - Gmail.*$/, "").trim();
  }

  // Sender: .gD contains sender email attribute or text
  const senderEl = document.querySelector("span.gD[email]") || 
                   document.querySelector("span[email]") ||
                   document.querySelector(".gD");
  if (senderEl) {
    sender = senderEl.getAttribute("email") || senderEl.innerText.trim();
  }

  // Body: .a3s.aiL is the message body container in Gmail
  const bodyEls = document.querySelectorAll(".a3s.aiL");
  if (bodyEls && bodyEls.length > 0) {
    // Pick the last open message in the conversation thread
    const activeBody = bodyEls[bodyEls.length - 1];
    bodyText = activeBody.innerText.trim();

    // Extract links inside the email body
    const links = activeBody.querySelectorAll("a[href]");
    links.forEach(a => {
      const href = a.href;
      if (href && (href.startsWith("http://") || href.startsWith("https://"))) {
        urls.push(href);
      }
    });
  }

  return {
    source: "Gmail",
    found: Boolean(subject || bodyText || sender),
    subject,
    sender,
    body_text: bodyText,
    urls: Array.from(new Set(urls))
  };
}

function extractFromOutlook() {
  let subject = "";
  let sender = "";
  let bodyText = "";
  const urls = [];

  // Subject in Outlook Web
  const subjectEl = document.querySelector("[role='heading'][aria-level='2']") ||
                    document.querySelector("[data-testid='messageHeaderSubject']") ||
                    document.querySelector("div[aria-label='Reading Pane'] [title]");
  if (subjectEl) {
    subject = (subjectEl.getAttribute("title") || subjectEl.innerText).trim();
  }

  // Sender in Outlook Web
  const senderEl = document.querySelector("[data-testid='SenderPersona'] span[title*='@']") ||
                   document.querySelector("span[title*='@']") ||
                   document.querySelector(".O365_sender");
  if (senderEl) {
    sender = (senderEl.getAttribute("title") || senderEl.innerText).trim();
  }

  // Body in Outlook Web
  const bodyEl = document.querySelector("[aria-label='Message body']") ||
                 document.querySelector(".ItemPartView") ||
                 document.querySelector(".allowTextSelection");
  if (bodyEl) {
    bodyText = bodyEl.innerText.trim();

    const links = bodyEl.querySelectorAll("a[href]");
    links.forEach(a => {
      const href = a.href;
      if (href && (href.startsWith("http://") || href.startsWith("https://"))) {
        urls.push(href);
      }
    });
  }

  return {
    source: "Outlook",
    found: Boolean(subject || bodyText || sender),
    subject,
    sender,
    body_text: bodyText,
    urls: Array.from(new Set(urls))
  };
}

function extractGeneric() {
  // If user selected text on any webpage, use that!
  const selection = window.getSelection() ? window.getSelection().toString().trim() : "";

  let bodyText = selection;
  if (!bodyText) {
    // Fallback to main article or visible text
    const mainEl = document.querySelector("article") || document.querySelector("main");
    if (mainEl) {
      bodyText = mainEl.innerText.slice(0, 3000);
    }
  }

  const urls = [];
  const links = document.querySelectorAll("a[href]");
  links.forEach(a => {
    if (urls.length < 20 && (a.href.startsWith("http://") || a.href.startsWith("https://"))) {
      urls.push(a.href);
    }
  });

  return {
    source: "Webpage",
    found: Boolean(bodyText || document.title),
    subject: document.title,
    sender: "",
    body_text: bodyText,
    urls: Array.from(new Set(urls))
  };
}
