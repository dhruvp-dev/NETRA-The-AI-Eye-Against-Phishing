"""
NETRA Tier-1 stress, adversarial and latency suite (script, not a unit test).

    python tests/stress_test_tier1.py                  # default p95 budget 800 ms
    python tests/stress_test_tier1.py --p95-budget-ms 500

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
        "expected": {"SUSPICIOUS", "PHISHING"},
    },
    {
        "name": "Adversarial: Word Insertion & Semantic Noise Injection",
        "req": PredictRequest(
            subject="Account review memo regarding weather and schedules",
            sender="notice@secure-update-corp.com",
            body_text="The sunny sky in Paris is beautiful today. Please review your pending credentials verification here: http://secure-update-corp.com/login. Coffee break at 4 PM.",
            headers={"spf": "fail", "dkim": "fail", "dmarc": "fail"},
        ),
        "expected": {"SUSPICIOUS", "PHISHING"},
    },
    {
        "name": "Adversarial: IP Address in URL (Bypassing domain filters)",
        "req": PredictRequest(
            subject="DocuSign Document Ready for E-Signature",
            sender="documents@docusign-contracts.info",
            body_text="You have 1 document waiting for your legal signature. Sign now: http://192.168.1.100/docusign/verify",
            headers={"spf": "none", "dkim": "none", "dmarc": "fail"},
        ),
        "expected": {"SUSPICIOUS", "PHISHING"},
    },
    {
        "name": "Subtle Spear Phishing (No typical urgency buzzwords)",
        "req": PredictRequest(
            subject="Updated 2026 Employee Health & Dental Benefits",
            sender="benefits@portal-employee-update.com",
            body_text="Hello all, Following our HR review, the corporate health insurance handbook has been updated. Please log in to review your deductible choices.",
            headers={"spf": "none", "dkim": "none", "dmarc": "none"},
        ),
        "expected": {"SUSPICIOUS", "PHISHING"},
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
        "expected": {"SUSPICIOUS", "PHISHING"},
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

    print(f"\n[Step 1/3] {len(TEST_CASES)} verdict scenarios\n")
    for idx, tc in enumerate(TEST_CASES, 1):
        t0 = time.perf_counter()
        resp = run_predict(tc["req"])
        elapsed = (time.perf_counter() - t0) * 1000.0
        ok = resp.classification in tc["expected"]
        badge = "PASS" if ok else "FAIL"
        if not ok:
            failures.append(f"{tc['name']}: got {resp.classification}, expected {sorted(tc['expected'])}")
        print(f"[{idx:02d}/{len(TEST_CASES)}] {tc['name']}")
        print(f"        Verdict: {resp.classification:<11} | Score: {resp.risk_score:.4f} | "
              f"Latency: {elapsed:6.1f} ms | {badge}")

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
    if failures:
        print(f"RESULT: FAILED ({len(failures)} problem(s))")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
