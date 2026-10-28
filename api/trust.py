"""
NETRA - Sender trust and link alignment (post-model de-escalation)
==================================================================
The shipped Tier-1 checkpoint learned a corpus shortcut: formal, brand-style mail with links
scores as phishing (bugs/tier1_model_bugs.md, B1). Until it is retrained on representative
legitimate mail, this rule lowers the verdict when the evidence outside the text says the
email is what it claims to be:

    trusted sender  : a known brand domain (curated list below, a stand-in for a reputation feed)
                      or authenticated (DMARC pass, or SPF and DKIM pass) with no auth failure
    aligned links   : every link goes to the sender's own organisation, another known brand,
                      or a neutral email-service host; no look-alike, shortener, IP, high-risk TLD
                      or user-content host; Reply-To in the sender's organisation

A spoofed brand email whose links all go to the real brand cannot send the reader to a fake
page, so lowering the verdict there is low-risk. Authentication alone proves origin, not
intent (an attacker can authenticate their own domain), so an authenticated but unknown
sender is lowered by one level only (PHISHING -> SUSPICIOUS).

Usage:
    from api.trust import assess
    t = assess(sender, reply_to, headers, urls)
    t.level   # "known" | "verified" | None
"""

from dataclasses import dataclass, field
from email.utils import parseaddr
from typing import Dict, Iterable, List, Optional, Set
from urllib.parse import urlparse

from ml.features.header_features import normalize_auth
from ml.features.url_features import (BRAND_REAL_DOMAINS, _has_phishing_tld, _is_ip_address,
                                      _split_host, check_typosquatting)

# ---------------------------------------------------------------------------
# Curated domain lists
# ---------------------------------------------------------------------------
# Organisation -> registrable domains it owns. Starts from the typosquatting brand table and adds
# common transactional / notification senders. Extend it, or replace it with a reputation feed.
KNOWN_ORGS: Dict[str, Set[str]] = {org: set(domains) for org, domains in BRAND_REAL_DOMAINS.items()}
for _org, _domains in {
    "amazon": {"amazon.in", "amazon.co.uk", "amazon.de", "amazon.ca", "amazon.com.au", "a.co",
               "media-amazon.com", "ssl-images-amazon.com", "primevideo.com", "audible.com",
               "amazonpay.in"},
    "google": {"youtube.com", "gstatic.com", "withgoogle.com", "g.co", "google.co.in", "youtu.be"},
    "microsoft": {"bing.com", "skype.com", "xbox.com", "msn.com", "azure.com", "windows.com",
                  "microsoftstore.com"},
    "apple": {"me.com", "itunes.com", "apple.news"},
    "facebook": {"facebookmail.com", "meta.com", "messenger.com", "whatsapp.com"},
    "linkedin": {"licdn.com", "lnkd.in"},
    "twitter": {"t.co", "twimg.com"},
    "netflix": {"nflxext.com", "nflximg.net", "nflxso.net"},
    "spotify": {"spotify.com", "scdn.co", "spotifycdn.com"},
    "paypal": {"paypal.me", "paypalobjects.com"},
    "stripe": {"stripe.com", "stripe.network"},
    "chase": {"jpmorganchase.com", "jpmorgan.com"},
    "bankofamerica": {"bofa.com", "ml.com"},
    "citi": {"citi.com", "citibank.com"},
    "capitalone": {"capitalone.com"},
    "amex": {"americanexpress.com", "aexp.com"},
    "discover": {"discover.com"},
    "usbank": {"usbank.com"},
    "hsbc": {"hsbc.com", "hsbc.co.uk", "hsbc.co.in"},
    "barclays": {"barclays.co.uk", "barclays.com"},
    "sbi": {"sbi.co.in", "onlinesbi.sbi"},
    "hdfc": {"hdfcbank.com"},
    "icici": {"icicibank.com"},
    "axis": {"axisbank.com"},
    "target": {"target.com"},
    "walmart": {"walmart.com"},
    "ebay": {"ebay.com", "ebay.co.uk", "ebaystatic.com"},
    "etsy": {"etsy.com", "etsystatic.com"},
    "bestbuy": {"bestbuy.com"},
    "costco": {"costco.com"},
    "homedepot": {"homedepot.com"},
    "ikea": {"ikea.com"},
    "flipkart": {"flipkart.com"},
    "myntra": {"myntra.com"},
    "aliexpress": {"aliexpress.com"},
    "uber": {"uber.com", "ubereats.com"},
    "lyft": {"lyft.com"},
    "doordash": {"doordash.com"},
    "grubhub": {"grubhub.com"},
    "instacart": {"instacart.com"},
    "swiggy": {"swiggy.com", "swiggy.in"},
    "zomato": {"zomato.com"},
    "airbnb": {"airbnb.com"},
    "booking": {"booking.com"},
    "expedia": {"expedia.com"},
    "usps": {"usps.com"},
    "royalmail": {"royalmail.com"},
    "gitlab": {"gitlab.com"},
    "atlassian": {"atlassian.com"},
    "slack": {"slack.com", "slack-edge.com"},
    "zoom": {"zoom.us", "zoom.com"},
    "box": {"box.com"},
    "dropbox": {"dropboxmail.com"},
    "notion": {"notion.so"},
    "figma": {"figma.com"},
    "adobe": {"adobe.com"},
    "salesforce": {"salesforce.com"},
    "openai": {"openai.com"},
    "anthropic": {"anthropic.com", "claude.ai"},
    "docusign": {"docusign.com", "docusign.net"},
    "intuit": {"intuit.com", "quickbooks.com"},
    "mailchimp": {"mailchimp.com"},
    "canva": {"canva.com"},
    "duolingo": {"duolingo.com"},
    "twitch": {"twitch.tv"},
    "reddit": {"reddit.com", "redditmail.com"},
    "pinterest": {"pinterest.com"},
    "tiktok": {"tiktok.com"},
    "medium": {"medium.com"},
    "substack": {"substack.com"},
    "quora": {"quora.com"},
    "coursera": {"coursera.org"},
    "udemy": {"udemy.com"},
    "edx": {"edx.org"},
    "khanacademy": {"khanacademy.org"},
    "nytimes": {"nytimes.com"},
    "washingtonpost": {"washingtonpost.com"},
    "bbc": {"bbc.co.uk", "bbc.com"},
    "cnn": {"cnn.com"},
    "guardian": {"theguardian.com"},
    "healthline": {"healthline.com"},
    "hulu": {"hulu.com"},
    "disney": {"disneyplus.com", "disney.com"},
    "max": {"max.com", "hbomax.com"},
    "verizon": {"verizon.com"},
    "att": {"att.com"},
    "tmobile": {"t-mobile.com"},
    "airtel": {"airtel.in"},
    "jio": {"jio.com"},
}.items():
    KNOWN_ORGS.setdefault(_org, set()).update(_domains)

DOMAIN_TO_ORG: Dict[str, str] = {d: org for org, ds in KNOWN_ORGS.items() for d in ds}

# Anyone can get an address here, so a freemail sender is never a "known brand".
FREEMAIL_DOMAINS = {
    "gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com", "msn.com", "yahoo.com",
    "yahoo.co.in", "ymail.com", "aol.com", "icloud.com", "me.com", "proton.me", "protonmail.com",
    "gmx.com", "gmx.de", "mail.com", "zoho.com", "yandex.com", "rediffmail.com",
}

# Email-service click-tracking / unsubscribe / CDN hosts used by legitimate bulk senders.
NEUTRAL_LINK_DOMAINS = {
    "list-manage.com", "mailchi.mp", "mcusercontent.com", "sendgrid.net", "mandrillapp.com",
    "mailgun.org", "sparkpostmail.com", "rs6.net", "constantcontact.com", "hubspotlinks.com",
    "hubspotemail.net", "exacttarget.com", "exct.net", "mktomail.com", "klaviyomail.com",
    "createsend.com", "cmail19.com", "cmail20.com", "amazonses.com", "sailthru.com", "responsys.net",
    "cloudfront.net", "akamaihd.net",
}

# Hosts where anyone can publish a page. They break alignment even when the registrable domain
# belongs to a known organisation (phishing pages on Google Sites / Forms, Firebase, GitHub Pages...).
USER_CONTENT_HOST_SUFFIXES = (
    "docs.google.com", "sites.google.com", "drive.google.com", "forms.gle", "storage.googleapis.com",
    "googleusercontent.com", "firebaseapp.com", "web.app", "github.io", "githubusercontent.com",
    "dropbox.com", "dropboxusercontent.com", "1drv.ms", "onedrive.live.com", "sharepoint.com",
    "blogspot.com", "wixsite.com", "weebly.com", "notion.site", "myshopify.com", "glitch.me",
    "netlify.app", "vercel.app", "pages.dev", "r2.dev", "ipfs.io", "s3.amazonaws.com",
)

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly", "cutt.ly", "rebrand.ly", "buff.ly",
    "shorturl.at", "rb.gy", "t.ly", "tiny.cc", "bl.ink", "s.id", "v.gd", "lnkd.in", "youtu.be",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def address_domain(address: Optional[str]) -> str:
    _, addr = parseaddr(address or "")
    return addr.rsplit("@", 1)[1].lower().strip(".") if "@" in addr else ""


def registrable(host: str) -> str:
    return _split_host(host.lower().strip("."))[1] if host else ""


def url_host(url: str) -> str:
    try:
        return (urlparse(url if "://" in url else f"http://{url}").hostname or "").lower()
    except ValueError:
        return ""


def is_shortener(host: str) -> bool:
    return registrable(host) in URL_SHORTENERS or host in URL_SHORTENERS


def is_user_content(host: str) -> bool:
    return any(host == s or host.endswith("." + s) for s in USER_CONTENT_HOST_SUFFIXES)


def org_domains(domain: str) -> Set[str]:
    """Domains of the organisation that owns `domain` (just {domain} for an unknown one)."""
    reg = registrable(domain)
    org = DOMAIN_TO_ORG.get(reg)
    return set(KNOWN_ORGS[org]) if org else {reg}


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------
@dataclass
class TrustAssessment:
    sender_domain: str = ""
    known_sender: bool = False
    verified: bool = False
    aligned: bool = False
    reasons: List[str] = field(default_factory=list)    # why it is trusted (for explanations)
    blockers: List[str] = field(default_factory=list)   # why it is not

    @property
    def level(self) -> Optional[str]:
        """'known' (known brand, links aligned), 'verified' (authenticated, links aligned) or None."""
        if not self.aligned:
            return None
        if self.known_sender:
            return "known"
        if self.verified:
            return "verified"
        return None


def assess(sender: Optional[str], reply_to: Optional[str], headers: Optional[dict],
           urls: Iterable[str]) -> TrustAssessment:
    t = TrustAssessment()
    domain = address_domain(sender)
    t.sender_domain = registrable(domain) if domain else ""
    if not t.sender_domain:
        t.blockers.append("no sender address")
        return t

    hdrs = headers or {}
    auth = {k: normalize_auth(hdrs.get(k)) for k in ("spf", "dkim", "dmarc")}
    if any(v == "fail" for v in auth.values()):
        t.blockers.append("sender authentication failed")
        return t
    if check_typosquatting(domain)[0]:
        t.blockers.append("look-alike sender domain")
        return t

    freemail = t.sender_domain in FREEMAIL_DOMAINS
    t.known_sender = not freemail and t.sender_domain in DOMAIN_TO_ORG
    t.verified = not freemail and (auth["dmarc"] == "pass" or (auth["spf"] == "pass" and auth["dkim"] == "pass"))
    if freemail:
        t.blockers.append("free-mail sender")
    if t.known_sender:
        t.reasons.append(f"known sender ({t.sender_domain})")
    if t.verified:
        passed = [k.upper() for k, v in auth.items() if v == "pass"]
        t.reasons.append(f"authenticated ({', '.join(passed)} passed)")

    own = org_domains(t.sender_domain)
    rt_domain = address_domain(reply_to)
    if rt_domain and registrable(rt_domain) not in own:
        t.blockers.append(f"Reply-To goes to {registrable(rt_domain)}")

    for u in urls:
        host = url_host(u)
        reg = registrable(host)
        if not host:
            t.blockers.append("unparseable link")
        elif _is_ip_address(host):
            t.blockers.append(f"link to IP address {host}")
        elif is_user_content(host):
            t.blockers.append(f"link to user-content host {host}")
        elif reg in own:
            continue
        elif is_shortener(host):
            t.blockers.append(f"shortened link ({host})")
        elif _has_phishing_tld(host) or check_typosquatting(host)[0]:
            t.blockers.append(f"suspicious link domain {host}")
        elif reg in DOMAIN_TO_ORG or reg in NEUTRAL_LINK_DOMAINS:
            continue
        else:
            t.blockers.append(f"link to unrelated domain {reg}")

    t.aligned = not t.blockers
    if t.aligned:
        t.reasons.append(f"all links point to {t.sender_domain} or known services" if urls
                         else "no links")
    return t
