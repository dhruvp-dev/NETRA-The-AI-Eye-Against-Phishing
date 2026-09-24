/**
 * NETRA Extension - Content Script
 * Intelligently extracts email data from webmail clients (Gmail, Outlook, etc.)
 * and assists the popup scanner.
 */

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

  return true;
});

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
