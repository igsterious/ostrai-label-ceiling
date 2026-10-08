"""Score tonight's candidate models against every verdict a human has given.

The 147 archived (message, category) pairs cover most predictions for free,
so a new model usually costs a handful of judgements rather than 120.
"""
import csv
import sys
from math import comb

import pandas as pd

sys.path.insert(0, r"C:\Users\Igor\ostrai-final\src")
from config import LABELS_ROOT

CANDIDATES = {
    "consensus": {
        "556": "card_not_working", "957": "card_arrival",
        "1363": "why_verify_identity", "2416": "verify_my_identity",
        "3227": "top_up_failed", "3323": "passcode_forgotten",
        "3375": "exchange_charge", "4092": "pending_transfer",
        "4158": "transfer_timing", "8769": "card_delivery_estimate",
        "8858": "top_up_failed",
    },
    "gold(w5)": {
        "1363": "why_verify_identity", "2416": "verify_my_identity",
        "3227": "top_up_failed", "3323": "passcode_forgotten",
        "3327": "declined_card_payment", "3375": "exchange_charge",
        "4092": "pending_transfer", "4798": "verify_my_identity",
        "5586": "card_payment_not_recognised",
        "6971": "supported_cards_and_currencies", "8210": "cancel_transfer",
        "8211": "wrong_exchange_rate_for_cash_withdrawal",
        "8769": "card_delivery_estimate", "8858": "top_up_failed",
    },
    "lr2e5": {
        "556": "card_not_working", "2416": "verify_my_identity",
        "2448": "atm_support", "3227": "top_up_failed",
        "3323": "passcode_forgotten", "3797": "exchange_via_app",
        "4092": "failed_transfer", "4158": "transfer_timing",
        "4798": "verify_my_identity", "5586": "card_payment_not_recognised",
        "7400": "transfer_not_received_by_recipient",
        "8769": "card_delivery_estimate", "8858": "top_up_failed",
        "9412": "cash_withdrawal_charge",
    },
    "smooth.1": {
        "556": "card_not_working", "2416": "unable_to_verify_identity",
        "2448": "atm_support", "3227": "top_up_failed",
        "3323": "passcode_forgotten", "3327": "declined_card_payment",
        "4092": "pending_transfer", "4158": "transfer_timing",
        "5586": "card_payment_not_recognised", "8210": "cancel_transfer",
        "8769": "card_delivery_estimate", "8858": "top_up_failed",
    },
    "both": {
        "556": "card_not_working", "1363": "why_verify_identity",
        "2448": "wrong_exchange_rate_for_cash_withdrawal",
        "3323": "passcode_forgotten", "4158": "transfer_timing",
        "4320": "card_payment_fee_charged",
        "5586": "card_payment_not_recognised",
        "7400": "transfer_not_received_by_recipient",
        "8769": "card_delivery_estimate",
    },
    "both+12ep": {
        "556": "card_not_working", "1363": "why_verify_identity",
        "2448": "atm_support", "3227": "top_up_failed",
        "3323": "passcode_forgotten", "3327": "declined_card_payment",
        "3375": "exchange_charge", "4092": "failed_transfer",
        "5586": "card_payment_not_recognised",
        "6971": "supported_cards_and_currencies",
        "8769": "card_delivery_estimate", "8858": "top_up_failed",
    },
    "ens3": {
        "556": "card_not_working", "957": "card_arrival",
        "1363": "why_verify_identity", "2416": "verify_my_identity",
        "3323": "passcode_forgotten", "3327": "declined_card_payment",
        "3375": "exchange_charge", "4092": "pending_transfer",
        "4158": "transfer_timing", "5586": "card_payment_not_recognised",
        "8769": "card_delivery_estimate", "8858": "top_up_failed",
    },
    "mpnet": {
        "556": "card_not_working", "957": "card_arrival",
        "2416": "verify_my_identity", "3323": "passcode_forgotten",
        "3327": "declined_card_payment", "3375": "exchange_charge",
        "4092": "pending_transfer", "4158": "transfer_timing",
        "4320": "direct_debit_payment_not_recognised",
        "4798": "verify_my_identity", "5586": "card_payment_not_recognised",
        "7400": "transfer_not_received_by_recipient",
        "8769": "card_delivery_estimate",
    },
}
# DeBERTa-base, the model currently submitted (test 82.06%).
DEBERTA = {
    "957": "card_arrival", "2416": "verify_my_identity", "2448": "atm_support",
    "3323": "passcode_forgotten", "3327": "declined_card_payment",
    "3375": "exchange_charge", "3797": "exchange_via_app",
    "4092": "pending_transfer", "4158": "transfer_timing",
    "4798": "verify_my_identity", "5586": "card_payment_not_recognised",
    "6971": "supported_cards_and_currencies", "8210": "cancel_transfer",
    "8769": "card_delivery_estimate",
}

pairs = pd.read_csv(LABELS_ROOT / "verdict_pairs_benchmark_a.tsv", sep="\t",
                    quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
verdicts = {(r.id, r.category): r.verdict for r in pairs.itertuples()}

cur = pd.read_csv(LABELS_ROOT / "audit_benchmark_a.tsv", sep="\t",
                  quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
base = dict(zip(cur.id, cur.assigned))
basev = dict(zip(cur.id, cur.verdict))

models = {"DistilBERT": dict(base), "DeBERTa-base": {i: DEBERTA.get(i, p) for i, p in base.items()}}
for name, changed in CANDIDATES.items():
    models[name] = {i: changed.get(i, p) for i, p in base.items()}

print(f"{'model':14s}{'judged':>8s}{'ok':>5s}{'bad':>5s}{'amb':>5s}{'open':>6s}{'strict':>20s}")
scored = {}
for name, preds in models.items():
    assert len(preds) == 120
    known = {i: verdicts[(i, p)] for i, p in preds.items() if (i, p) in verdicts}
    unknown = {i: p for i, p in preds.items() if (i, p) not in verdicts}
    c = {"ok": 0, "bad": 0, "maybe": 0}
    for v in known.values():
        c[v] += 1
    lo, hi = c["ok"], c["ok"] + len(unknown)
    band = f"{lo/120:.1%}" if not unknown else f"{lo/120:.1%} .. {hi/120:.1%}"
    print(f"{name:14s}{len(known):>8d}{c['ok']:>5d}{c['bad']:>5d}{c['maybe']:>5d}"
          f"{len(unknown):>6d}{band:>20s}")
    scored[name] = (known, unknown)

print("\n  reference: DistilBERT audit 81.7% -> test 81.25%"
      "   DeBERTa audit 83.3% -> test 82.06%")

print()
for name in CANDIDATES:
    known, _ = scored[name]
    for ref in ("DistilBERT", "DeBERTa-base"):
        rk, _ = scored[ref]
        a = [known.get(i) == "ok" for i in base]
        b = [rk.get(i) == "ok" for i in base]
        w = sum(1 for x, y in zip(a, b) if x and not y)
        l = sum(1 for x, y in zip(a, b) if y and not x)
        n = w + l
        p = (sum(comb(n, k) for k in range(n + 1) if abs(k - n / 2) >= abs(w - n / 2)) / 2 ** n
             if n else 1.0)
        print(f"  {name:10s} vs {ref:13s} wins {w}, loses {l}, p = {p:.4f}")

for name in CANDIDATES:
    _, unknown = scored[name]
    if unknown:
        print(f"\n  {name} open pairs ({len(unknown)}):")
        for i, pr in sorted(unknown.items(), key=lambda x: int(x[0])):
            print(f"    {i:>6s} {base[i]:<40s} -> {pr}")
