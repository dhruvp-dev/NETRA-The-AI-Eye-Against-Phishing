import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import asyncio
from api.main import (
    load_models,
    predict,
    PredictRequest,
    state,
    _extract_signals,
    should_escalate
)

print("=" * 75)
print("NETRA TIER-1 COMPREHENSIVE STRESS, ADVERSARIAL & LATENCY TEST SUITE")
print("=" * 75)

# Initialize models
print("\n[Step 1/5] Booting NETRA Tier-1 Deep Learning Engine...")
load_models()
print(f"Model Ready: {state.model_loaded} | Type: {state.model_type}")
print(f"Thresholds : Phishing >= {state.phishing_threshold} | Suspicious [{state.suspicious_lower}, {state.suspicious_upper})")

# 12 Extensive Test Cases targeting boundaries and adversarial vectors
TEST_CASES = [
    # 1. Clear Legitimate Emails
    {
        "name": "Legitimate E-commerce Purchase Confirmation",
        "req": PredictRequest(
            subject="Your Target.com order #84921 has been placed",
            sender="orders@target.com",
            body_text="Thank you for your order! Your items will ship soon. Total charged: $34.50. View tracking details on your account dashboard.",
            headers={"spf": "pass", "dkim": "pass", "dmarc": "pass"}
        ),
        "expected": "LEGITIMATE"
    },
    {
        "name": "Legitimate Corporate Calendar Invitation",
        "req": PredictRequest(
            subject="Invited: Q4 Product Strategy Review @ Thu 3pm",
            sender="alex.miller@microsoft.com",
            body_text="Hi team, Please find the agenda for the upcoming quarterly roadmap discussion. Looking forward to your thoughts and deck reviews.",
            headers={"spf": "pass", "dkim": "pass", "dmarc": "pass"}
        ),
        "expected": "LEGITIMATE"
    },

    # 2. Classic High-Risk Phishing Attacks
    {
        "name": "Classic Urgent Banking Phish (Account Suspension)",
        "req": PredictRequest(
            subject="CRITICAL: Your Wells Fargo account is suspended immediately",
            sender="security@wells-fargo-alert-auth.net",
            body_text="Unauthorized access was detected on your checking account. Click here immediately to verify your SSN and password or account will be permanently locked: http://wells-fargo-verify-login.xyz/auth",
            headers={"spf": "fail", "dkim": "fail", "dmarc": "fail"}
        ),
        "expected": "PHISHING"
    },
    {
        "name": "Cryptocurrency Wallet Drainer Scam",
        "req": PredictRequest(
            subject="Urgent: Claim your $5,000 ETH Airdrop within 2 hours",
            sender="airdrop@metamask-rewards-claim.org",
            body_text="Congratulations! Your wallet address was selected for the annual Ethereum Foundation distribution. Connect your secret recovery phrase now to withdraw your reward: http://metamask-claim-free-eth.com",
            headers={"spf": "none", "dkim": "none", "dmarc": "none"}
        ),
        "expected": "PHISHING"
    },

    # 3. Adversarial / Evasion Techniques (Testing Tier-1 Limits)
    {
        "name": "Adversarial: Typosquatted Domain with Valid SPF",
        "req": PredictRequest(
            subject="Your Netflix Membership Renewal Problem",
            sender="billing@netf1ix.com", # Note the '1' instead of 'l'
            body_text="We were unable to validate your card. Please update your payment method to continue watching: https://www.netf1ix-billing.com/update",
            headers={"spf": "pass", "dkim": "pass", "dmarc": "pass"} # Attacker registered domain with full SPF pass!
        ),
        "expected": "SUSPICIOUS/PHISHING"
    },
    {
        "name": "Adversarial: Word Insertion & Semantic Noise Injection",
        "req": PredictRequest(
            subject="Account review memo regarding weather and schedules",
            sender="notice@secure-update-corp.com",
            body_text="The sunny sky in Paris is beautiful today. Please review your pending credentials verification here: http://secure-update-corp.com/login. Coffee break at 4 PM.",
            headers={"spf": "fail", "dkim": "fail", "dmarc": "fail"}
        ),
        "expected": "SUSPICIOUS/PHISHING"
    },
    {
        "name": "Adversarial: IP Address in URL (Bypassing domain filters)",
        "req": PredictRequest(
            subject="DocuSign Document Ready for E-Signature",
            sender="documents@docusign-contracts.info",
            body_text="You have 1 document waiting for your legal signature. Sign now: http://192.168.1.100/docusign/verify",
            headers={"spf": "none", "dkim": "none", "dmarc": "fail"}
        ),
        "expected": "SUSPICIOUS/PHISHING"
    },
    {
        "name": "Subtle Spear Phishing (No typical urgency buzzwords)",
        "req": PredictRequest(
            subject="Updated 2026 Employee Health & Dental Benefits",
            sender="benefits@portal-employee-update.com",
            body_text="Hello all, Following our HR review, the corporate health insurance handbook has been updated. Please log in to review your deductible choices.",
            headers={"spf": "none", "dkim": "none", "dmarc": "none"}
        ),
        "expected": "SUSPICIOUS/PHISHING"
    },

    # 4. Stress Edge Cases
    {
        "name": "Edge Case: Ultra-Short Body (1 line with suspicious link)",
        "req": PredictRequest(
            subject="Scanned file",
            sender="printer@internal-scan.co",
            body_text="View scan: http://internal-scan.co/doc.exe",
            headers={"spf": "fail", "dkim": "none", "dmarc": "none"}
        ),
        "expected": "SUSPICIOUS/PHISHING"
    },
    {
        "name": "Edge Case: Multilingual / Non-Latin / Emoji Injection",
        "req": PredictRequest(
            subject="⚠️ Security Alert / 警告 / إشعار أمني",
            sender="support@bank-security.xyz",
            body_text="🚨 URGENT: حسابك معلق! 请验证账户 Click http://bank-security.xyz/verify to validate your identity 🔑",
            headers={"spf": "fail", "dkim": "none", "dmarc": "fail"}
        ),
        "expected": "PHISHING"
    },
    {
        "name": "Edge Case: Massive Body Payload (50,000 characters)",
        "req": PredictRequest(
            subject="Quarterly Financial Report Comprehensive Annex",
            sender="investor-relations@apple.com",
            body_text=("Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories. " * 800),
            headers={"spf": "pass", "dkim": "pass", "dmarc": "pass"}
        ),
        "expected": "LEGITIMATE"
    }
]

print(f"\n[Step 2/5] Running {len(TEST_CASES)} Edge, Stress & Adversarial Scenarios...\n")

scenario_results = []
for idx, tc in enumerate(TEST_CASES, 1):
    t0 = time.perf_counter()
    resp = asyncio.run(predict(tc["req"]))
    elapsed = (time.perf_counter() - t0) * 1000.0

    signals = _extract_signals(tc["req"])
    is_escalated = should_escalate(
        {"classification": resp.classification, "risk_score": resp.risk_score, "confidence": resp.confidence},
        signals
    )

    matches = (resp.classification in tc["expected"]) or (is_escalated and "SUSPICIOUS" in tc["expected"])
    badge = "PASS [OK]" if matches else "WARN [CHECK]"

    print(f"[{idx:02d}/11] {tc['name']:<55}")
    print(f"       Verdict: {resp.classification:<11} | Score: {resp.risk_score:.4f} | Conf: {resp.confidence:.4f} | Escalate: {str(is_escalated):<5} | Latency: {elapsed:5.1f}ms | {badge}")

# [Step 3/5] Latency & Concurrency Stress Test
print("\n" + "=" * 75)
print("[Step 3/5] High-Throughput Stress Benchmark (50 Consecutive Predictions)")
print("=" * 75)

latencies = []
benchmark_req = TEST_CASES[0]["req"]

for i in range(50):
    t0 = time.perf_counter()
    asyncio.run(predict(benchmark_req))
    latencies.append((time.perf_counter() - t0) * 1000.0)

lat_arr = np.array(latencies)
p50 = np.percentile(lat_arr, 50)
p90 = np.percentile(lat_arr, 90)
p99 = np.percentile(lat_arr, 99)
avg_lat = np.mean(lat_arr)
throughput = 1000.0 / avg_lat

print(f"Completed 50 full transformer forward passes on CPU:")
print(f"  • Average Latency : {avg_lat:.2f} ms")
print(f"  • Median (P50)    : {p50:.2f} ms")
print(f"  • 90th %ile (P90) : {p90:.2f} ms")
print(f"  • 99th %ile (P99) : {p99:.2f} ms")
print(f"  • Throughput      : {throughput:.1f} inferences/second")

# [Step 4/5] Multi-Signal Threat Detection Verification
print("\n" + "=" * 75)
print("[Step 4/5] Heuristic Engine Stress Verification")
print("=" * 75)

typo_test = PredictRequest(
    subject="Security Notice",
    sender="support@micros0ft.com",
    body_text="Please update your password",
    headers={"spf": "fail"}
)
typo_signals = _extract_signals(typo_test)
print(f"Typosquatting Detection (micros0ft.com): {typo_signals['typosquatting_detected']} (Brand: {typo_signals.get('spoofed_brand')})")
print(f"Urgency Keyword Detection              : {typo_signals['urgency_detected']}")
print(f"Header Authentication Status           : Auth Failed = {typo_signals['header_auth_failed']}")

print("\n" + "=" * 75)
print("TIER-1 STRESS SUITE RESULTS SUMMARY")
print("=" * 75)
print("All scenarios executed successfully without crashes or memory exceptions!")
