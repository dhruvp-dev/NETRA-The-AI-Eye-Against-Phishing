"""
NETRA Tier-1 stress, adversarial and latency suite (script, not a unit test).

    python tests/stress_test_tier1.py                  # default p95 budget 800 ms
    python tests/stress_test_tier1.py --p95-budget-ms 500

Groups:
    core       the original 11 verdict scenarios (also used by tests/xai_sanity.py)
    legit      realistic brand / transactional / personal mail (audit B1: the model alone
               flags most of these; the trusted-sender rule must bring them back)
    phishing   realistic lures incl. BEC without links, shortener invoice, spoofed brand with one
               off-domain link and padded phishing (audit B3 / B5)

Each line shows the model score, the final score and the rule that changed it, if any.

Exit code:
    0  every expected verdict matched and p95 latency is within budget
    1  at least one verdict was missed or the latency budget was exceeded
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from api.main import PredictRequest, _extract_signals, load_models, predict, state
from ml.features.text_builder import extract_urls

AUTH_PASS = {"spf": "pass", "dkim": "pass", "dmarc": "pass"}
BAD = {"SUSPICIOUS", "PHISHING"}

TEST_CASES = [
    # 1. Clear legitimate
    {
        "name": "Legitimate E-commerce Purchase Confirmation",
        "req": PredictRequest(
            subject="Your Target.com order #84921 has been placed",
            sender="orders@target.com",
            body_text="Thank you for your order! Your items will ship soon. Total charged: $34.50. View tracking details on your account dashboard.",
            headers={"spf": "pass", "dkim": "pass", "dmarc": "pass"},
        ),
        "expected": {"LEGITIMATE"},
    },
    {
        "name": "Legitimate Corporate Calendar Invitation",
        "req": PredictRequest(
            subject="Invited: Q4 Product Strategy Review @ Thu 3pm",
            sender="alex.miller@microsoft.com",
            body_text="Hi team, Please find the agenda for the upcoming quarterly roadmap discussion. Looking forward to your thoughts and deck reviews.",
            headers={"spf": "pass", "dkim": "pass", "dmarc": "pass"},
        ),
        "expected": {"LEGITIMATE"},
    },
    # 2. Classic phishing
    {
        "name": "Classic Urgent Banking Phish (Account Suspension)",
        "req": PredictRequest(
            subject="CRITICAL: Your Wells Fargo account is suspended immediately",
            sender="security@wells-fargo-alert-auth.net",
            body_text="Unauthorized access was detected on your checking account. Click here immediately to verify your SSN and password or account will be permanently locked: http://wells-fargo-verify-login.xyz/auth",
            headers={"spf": "fail", "dkim": "fail", "dmarc": "fail"},
        ),
        "expected": {"PHISHING"},
    },
    {
        "name": "Cryptocurrency Wallet Drainer Scam",
        "req": PredictRequest(
            subject="Urgent: Claim your $5,000 ETH Airdrop within 2 hours",
            sender="airdrop@metamask-rewards-claim.org",
            body_text="Congratulations! Your wallet address was selected for the annual Ethereum Foundation distribution. Connect your secret recovery phrase now to withdraw your reward: http://metamask-claim-free-eth.com",
            headers={"spf": "none", "dkim": "none", "dmarc": "none"},
        ),
        "expected": {"PHISHING"},
    },
    # 3. Adversarial
    {
        "name": "Adversarial: Typosquatted Domain with Valid SPF",
        "req": PredictRequest(
            subject="Your Netflix Membership Renewal Problem",
            sender="billing@netf1ix.com",
            body_text="We were unable to validate your card. Please update your payment method to continue watching: https://www.netf1ix-billing.com/update",
            headers={"spf": "pass", "dkim": "pass", "dmarc": "pass"},
        ),
        "expected": BAD,
    },
    {
        "name": "Adversarial: Word Insertion & Semantic Noise Injection",
        "req": PredictRequest(
            subject="Account review memo regarding weather and schedules",
            sender="notice@secure-update-corp.com",
            body_text="The sunny sky in Paris is beautiful today. Please review your pending credentials verification here: http://secure-update-corp.com/login. Coffee break at 4 PM.",
            headers={"spf": "fail", "dkim": "fail", "dmarc": "fail"},
        ),
        "expected": BAD,
    },
    {
        "name": "Adversarial: IP Address in URL (Bypassing domain filters)",
        "req": PredictRequest(
            subject="DocuSign Document Ready for E-Signature",
            sender="documents@docusign-contracts.info",
            body_text="You have 1 document waiting for your legal signature. Sign now: http://192.168.1.100/docusign/verify",
            headers={"spf": "none", "dkim": "none", "dmarc": "fail"},
        ),
        "expected": BAD,
    },
    {
        "name": "Subtle Spear Phishing (No typical urgency buzzwords)",
        "req": PredictRequest(
            subject="Updated 2026 Employee Health & Dental Benefits",
            sender="benefits@portal-employee-update.com",
            body_text="Hello all, Following our HR review, the corporate health insurance handbook has been updated. Please log in to review your deductible choices.",
            headers={"spf": "none", "dkim": "none", "dmarc": "none"},
        ),
        "expected": BAD,
    },
    # 4. Edge cases
    {
        "name": "Edge Case: Ultra-Short Body (1 line with suspicious link)",
        "req": PredictRequest(
            subject="Scanned file",
            sender="printer@internal-scan.co",
            body_text="View scan: http://internal-scan.co/doc.exe",
            headers={"spf": "fail", "dkim": "none", "dmarc": "none"},
        ),
        "expected": BAD,
    },
    {
        "name": "Edge Case: Multilingual / Non-Latin / Emoji Injection",
        "req": PredictRequest(
            subject="Security Alert / 警告 / إشعار أمني",
            sender="support@bank-security.xyz",
            body_text="URGENT: حسابك معلق! 请验证账户 Click http://bank-security.xyz/verify to validate your identity",
            headers={"spf": "fail", "dkim": "none", "dmarc": "fail"},
        ),
        "expected": {"PHISHING"},
    },
    {
        "name": "Edge Case: Massive Body Payload (50,000 characters)",
        "req": PredictRequest(
            subject="Quarterly Financial Report Comprehensive Annex",
            sender="investor-relations@apple.com",
            body_text=("Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories. " * 800),
            headers={"spf": "pass", "dkim": "pass", "dmarc": "pass"},
        ),
        "expected": {"LEGITIMATE"},
    },
]


def _case(name, sender, subject, body, urls=(), headers=None, expected=None):
    return {"name": name, "expected": expected,
            "req": PredictRequest(subject=subject, sender=sender, body_text=body, urls=list(urls), headers=headers)}


def _legit(*a, **kw):
    return _case(*a, expected={"LEGITIMATE"}, **kw)


def _phish(*a, **kw):
    return _case(*a, expected=BAD, **kw)


# Realistic legitimate mail. About half carries no auth results, as when scanned from Gmail.
LEGIT_PROBES = [
    _legit("Amazon shipped", "shipment-tracking@amazon.com", "Your Amazon.com order #112-4567890 has shipped",
           "Hello, your package is on its way and will arrive Thursday. Track your package or view your order "
           "in Your Orders. Thank you for shopping with us.",
           ["https://www.amazon.com/gp/your-account/order-details?orderID=112-4567890"]),
    _legit("Walmart order", "help@walmart.com", "Thanks for your order",
           "We received your order #2000123-45678. Estimated delivery: Oct 21. Order total $56.12. "
           "View order details anytime in your account.",
           ["https://www.walmart.com/orders/2000123"], AUTH_PASS),
    _legit("Netflix invoice", "info@account.netflix.com", "Your Netflix payment receipt",
           "Hi Sam, thanks for your payment. We charged $15.49 to your Visa ending 4242 for your Standard plan. "
           "Your next billing date is November 17. Manage your account settings anytime.",
           ["https://www.netflix.com/YourAccount", "https://help.netflix.com/"]),
    _legit("Stripe payout", "notifications@stripe.com", "Payment of $420.00 received",
           "You received a payment of $420.00 from Acme Corp. The funds will be paid out to your bank account "
           "ending 6789 on Oct 20. View the payment in your Dashboard.",
           ["https://dashboard.stripe.com/payments/py_1abc"], AUTH_PASS),
    _legit("Spotify welcome", "no-reply@spotify.com", "Welcome to Spotify Premium",
           "Thanks for joining Premium. Enjoy ad-free music, offline listening and more. Your plan renews "
           "monthly; you can change or cancel it in your account page.",
           ["https://www.spotify.com/account/overview/", "https://open.spotify.com/"]),
    _legit("Google security alert", "no-reply@accounts.google.com", "Security alert",
           "A new sign-in to your Google Account was detected on a Windows device. If this was you, you don't "
           "need to do anything. If not, we'll help you secure your account. Check activity.",
           ["https://myaccount.google.com/notifications"], AUTH_PASS),
    _legit("LinkedIn password reset", "security-noreply@linkedin.com", "Here's the link to reset your password",
           "Hi Priya, we received a request to reset the password for your account. Use the link below to set "
           "a new password. If you didn't request this, you can ignore this email.",
           ["https://www.linkedin.com/e/v2?e=reset&token=abc123"]),
    _legit("Chase statement", "no.reply.alerts@chase.com", "Your statement is ready",
           "Your credit card statement for the account ending in 1234 is now available. Sign in to "
           "chase.com or the Chase Mobile app to view it. Minimum payment due Nov 12.",
           ["https://www.chase.com/personal/credit-cards/login"], AUTH_PASS),
    _legit("Calendar invite", "calendar-notification@google.com", "Invitation: Design sync @ Tue Oct 21, 2pm",
           "You have been invited to the following event. Design sync, Tuesday Oct 21, 2pm to 2:30pm. "
           "Join with Google Meet. Reply for yourself: Yes, No, Maybe.",
           ["https://meet.google.com/abc-defg-hij", "https://calendar.google.com/calendar/event?action=RESPOND"]),
    _legit("Healthline newsletter", "newsletters@healthline.com", "5 easy dinners for busy weeknights",
           "This week: quick sheet-pan recipes, the science of sleep, and how to build a simple stretching "
           "routine. Read more on our site. You received this because you subscribed.",
           ["https://www.healthline.com/nutrition/easy-dinners", "https://healthline.us1.list-manage.com/unsubscribe"],
           AUTH_PASS),
    _legit("Uber Eats order", "uber.us@uber.com", "Your Uber Eats order is confirmed",
           "Thanks for ordering from Thai Basil. Your order is being prepared and will arrive in 30-40 minutes. "
           "Total: $28.40. Track your order in the app.",
           ["https://www.ubereats.com/orders/xyz"]),
    _legit("Etsy order", "transaction@etsy.com", "Your Etsy order confirmation",
           "Thanks for your purchase from WoodcraftStudio. Order #3012345678, total $42.00. The seller will "
           "let you know when it ships. View your purchase on Etsy.",
           ["https://www.etsy.com/your/purchases/3012345678"], AUTH_PASS),
    _legit("GitHub notification", "notifications@github.com", "[netra] Pull request #42 merged",
           "Merged #42 into main. 3 files changed. You are receiving this because you were mentioned. "
           "Reply to this email directly or view it on GitHub.",
           ["https://github.com/example/netra/pull/42"]),
    _legit("PayPal receipt", "service@paypal.com", "Receipt for your payment to Bookshop",
           "You sent a payment of $18.99 USD to Bookshop. Transaction ID 9AB12345CD678901E. "
           "It may take a few moments for this transaction to appear in your account.",
           ["https://www.paypal.com/myaccount/transactions"], AUTH_PASS),
    _legit("Zoom recording", "no-reply@zoom.us", "Cloud recording is now available",
           "Your cloud recording of 'Weekly standup' is now available. Share the recording with "
           "participants or view it in your Zoom web portal.",
           ["https://zoom.us/recording/management"]),
    _legit("Apple receipt", "no_reply@email.apple.com", "Your receipt from Apple",
           "Apple ID: sam@example.com. iCloud+ 50 GB, monthly, $0.99. Billed to Visa 4242. "
           "If you have questions about your bill, visit Apple Support.",
           ["https://support.apple.com/billing", "https://www.apple.com/legal/"], AUTH_PASS),
    _legit("Personal mail", "maria.lopez@gmail.com", "Dinner on Saturday?",
           "Hey! Are you free on Saturday evening? We are thinking of trying the new Italian place near the "
           "park around 7. Let me know if that works for you."),
    _legit("Colleague mail", "j.chen@example-university.edu", "Draft of chapter 3",
           "Hi, I've attached my notes on the draft of chapter 3. The methods section reads well; maybe we can "
           "move the related-work part earlier. Happy to discuss on Thursday."),
    _legit("Dropbox share notice", "no-reply@dropboxmail.com", "Weekly digest",
           "Here is what happened in your shared folders this week: 4 files added to Project Alpha, "
           "2 comments on Budget.xlsx. Thanks for using Dropbox."),
    _legit("Long Amazon marketing", "store-news@amazon.com", "Deals picked for you",
           ("Discover deals on home, kitchen and electronics picked for you based on your recent browsing. "
            * 30) + "See all deals today.",
           ["https://www.amazon.com/deals", "https://www.amazon.com/gp/help/customer/display.html"]),
]

PADDING = ("Our quarterly planning meeting covered the roadmap, staffing and the office move. "
           "Thanks to everyone for the thoughtful discussion and the detailed notes. ") * 20

PHISH_PROBES = [
    _phish("PayPal spoof, one off-domain link", "service@paypal.com", "Your account has been limited",
           "We noticed unusual activity. To restore full access, confirm your information within 24 hours. "
           "Visit our help center at paypal.com or confirm now.",
           ["https://www.paypal.com/help", "https://paypal-resolution-center.com/confirm"]),
    _phish("Microsoft 365 password expiry", "it-support@office365-mailadmin.com", "Your password expires today",
           "Your Microsoft 365 password expires today. Keep your current password by verifying your account "
           "below, otherwise you will lose access to email.",
           ["https://office365-mailadmin.com/keep-password"]),
    _phish("DHL parcel fee", "noreply@dhl-parcel-redelivery.top", "Parcel on hold: customs fee unpaid",
           "Your parcel could not be delivered because a customs fee of $2.99 is unpaid. Pay now to schedule "
           "redelivery, or the parcel will be returned.",
           ["https://dhl-parcel-redelivery.top/pay"]),
    _phish("Fake invoice with bit.ly", "accounts@northwind-supplies.com", "Invoice INV-2291 overdue",
           "Hello, please find invoice INV-2291 for last month's order. The invoice is overdue; kindly settle it "
           "today to avoid late penalties. Download: https://bit.ly/inv2291-pdf"),
    _phish("CEO wire-transfer BEC, no link", "david.ross.ceo@gmail.com", "Re: wire transfer",
           "Are you at your desk? I need you to process a wire transfer of $48,500 to a new vendor today. "
           "I'm in a meeting and can't talk, so keep this confidential and email me once it's done."),
    _phish("Gift-card BEC", "principal.office@outlook.com", "Quick favor",
           "Hi, are you available? I need you to buy 5 Google Play gift cards of $100 each for staff awards. "
           "Scratch the back and send me the codes. Please keep this between us."),
    _phish("Payroll change request", "hr-payroll@acme-corp-hr.com", "Direct deposit update",
           "Hi, I changed banks and need to update my payroll details before Friday's run. Please change my "
           "direct deposit information to the new account below. This is urgent."),
    _phish("Credential harvest via Google Forms", "admin@university-it-helpdesk.com", "Mailbox quota exceeded",
           "Your mailbox has exceeded its storage limit. Validate your account within 24 hours to avoid "
           "suspension.", ["https://docs.google.com/forms/d/e/1FAIpQLSf/viewform"]),
    _phish("Apple ID locked", "support@apple-id-verification.info", "Your Apple ID has been locked",
           "Your Apple ID was locked for security reasons. Verify your identity to unlock it.",
           ["http://apple-id-verification.info/unlock"]),
    _phish("Bank OTP request", "alerts@sbi-kyc-update.com", "KYC update pending",
           "Dear customer, your KYC is pending. Your account will be suspended today. Share your OTP and "
           "login details on the link below to complete KYC.", ["https://sbi-kyc-update.com/kyc"]),
    _phish("Executable attachment link", "scanner@office-docs-share.net", "Scanned document",
           "A document was scanned and sent to you. Open the attached file to view it.",
           ["http://office-docs-share.net/files/scan_0412.exe"]),
    _phish("Netflix look-alike", "billing@netflix-account-support.com", "Payment declined",
           "We could not process your last payment. Update your billing details to avoid interruption.",
           ["https://netflix-account-support.com/billing"]),
    _phish("Padded phishing (B3)", "security@wellsfargo-online-alerts.net", "Meeting notes",
           PADDING + "Unauthorized access was detected on your checking account. Click here immediately to "
           "verify your SSN and password or account will be permanently locked: "
           "http://wells-fargo-verify-login.com/auth"),
    _phish("Crypto support scam", "support@walletconnect-help.org", "Wallet sync required",
           "Your wallet needs to be re-synchronised. Enter your recovery phrase to keep your funds safe.",
           ["https://walletconnect-help.org/sync"]),
    _phish("IP-hosted login", "it@helpdesk-mailcenter.com", "Action required",
           "Please log in to confirm your mailbox settings.", ["http://203.0.113.45/owa/login"]),
]

GROUPS = [("core", TEST_CASES), ("legit", LEGIT_PROBES), ("phishing", PHISH_PROBES)]


def run_predict(req: PredictRequest):
    return asyncio.run(predict(req))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="NETRA Tier-1 stress suite")
    parser.add_argument("--p95-budget-ms", type=float, default=800.0)
    args = parser.parse_args(argv)

    print("=" * 75)
    print("NETRA TIER-1 STRESS, ADVERSARIAL & LATENCY SUITE")
    print("=" * 75)

    load_models()
    if not state.model_loaded:
        print("FAIL: no model loaded. Place ml/models/distilbert_tier1.pt or run training first.")
        return 1
    print(f"Model   : {state.model_type} (loaded={state.model_loaded})")
    print(f"Thresholds: phishing >= {state.phishing_threshold} | "
          f"suspicious >= {state.suspicious_lower}")

    failures = []
    summary = []
    total = sum(len(cases) for _, cases in GROUPS)

    print(f"\n[Step 1/3] {total} verdict scenarios (model score -> final score)\n")
    for group, cases in GROUPS:
        print(f"--- {group} ---")
        ok_count = 0
        for idx, tc in enumerate(cases, 1):
            t0 = time.perf_counter()
            resp = run_predict(tc["req"])
            elapsed = (time.perf_counter() - t0) * 1000.0
            ok = resp.classification in tc["expected"]
            ok_count += ok
            if not ok:
                failures.append(f"[{group}] {tc['name']}: got {resp.classification}, "
                                f"expected {sorted(tc['expected'])}")
            adj = f" [{resp.adjustment}]" if resp.adjustment else ""
            print(f"[{idx:02d}/{len(cases)}] {'PASS' if ok else 'FAIL'}  {tc['name']}")
            print(f"        {resp.classification:<11} | model {resp.model_score:.4f} -> {resp.risk_score:.4f}"
                  f"{adj} | {elapsed:6.1f} ms")
        summary.append(f"{group}: {ok_count}/{len(cases)}")
        print()

    print("\n[Step 2/3] Latency benchmark (50 sequential predictions)")
    bench_req = TEST_CASES[0]["req"]
    run_predict(bench_req)                      # warm-up, not counted
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        run_predict(bench_req)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    lat = np.array(latencies)
    p50, p95 = np.percentile(lat, 50), np.percentile(lat, 95)
    print(f"  mean {lat.mean():.1f} ms | p50 {p50:.1f} ms | p95 {p95:.1f} ms | "
          f"budget p95 <= {args.p95_budget_ms:.0f} ms")
    if p95 > args.p95_budget_ms:
        failures.append(f"p95 latency {p95:.1f} ms exceeds budget {args.p95_budget_ms:.0f} ms")

    print("\n[Step 3/3] Signal checks")
    typo_req = PredictRequest(subject="Security Notice", sender="support@micros0ft.com",
                              body_text="Please update your password", headers={"spf": "fail"})
    typo_signals = _extract_signals(typo_req, extract_urls(typo_req.subject, typo_req.body_text, None))
    print(f"  typosquatting (micros0ft.com): {typo_signals['typosquatting_detected']}")
    if not typo_signals["typosquatting_detected"]:
        failures.append("micros0ft.com sender was not flagged as typosquatting")

    print("\n" + "=" * 75)
    print("Per group: " + " | ".join(summary))
    if failures:
        print(f"RESULT: FAILED ({len(failures)} problem(s))")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
