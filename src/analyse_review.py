"""Turn the manual review verdicts into an error-rate estimate for the pool.

The automated QA gate measured the labeller on messages of 20-60 characters,
but 30.4% of benchmark A's pool is longer than that. This reads the human
verdicts back and answers the question the gate could not:

  - Does the labeller hold up on messages longer than anything it was tested on?
  - Is the control group (inside the tested range) consistent with the gate's
    ~98%? If not, the review process itself is suspect, not the labels.

A Wilson confidence interval accompanies every rate. With 50 samples a point
estimate alone is misleading -- 48/50 and 46/50 are not meaningfully different,
and the interval says so rather than inviting over-reading.

    python src/analyse_review.py <verdicts_dir>
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib


def wilson(successes, n, z=1.96):
    """95% confidence interval for a proportion.

    Wilson rather than the textbook normal approximation because at rates near
    1.0 the normal interval runs past 100% and understates uncertainty -- which
    is exactly the regime we are in.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load(verdict_dir):
    rows = []
    for path in sorted(pathlib.Path(verdict_dir).rglob("*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return [r for r in rows if r.get("verdict")]


def summarise(rows, benchmark):
    subset = [r for r in rows if r.get("benchmark") == benchmark]
    out = {"benchmark": benchmark, "n": len(subset), "strata": {}}
    for stratum in ("long", "typical"):
        s = [r for r in subset if r.get("stratum") == stratum]
        n = len(s)
        ok = sum(1 for r in s if r["verdict"] == "ok")
        bad = sum(1 for r in s if r["verdict"] == "bad")
        maybe = sum(1 for r in s if r["verdict"] == "maybe")
        lo, hi = wilson(ok, n)
        # An ambiguous item is not a labelling error, but it is not a clean hit
        # either. Reporting both bounds avoids having to pick a convention:
        # strict counts only "correct"; lenient credits ambiguous ones too.
        lo2, hi2 = wilson(ok + maybe, n)
        out["strata"][stratum] = {
            "n": n, "ok": ok, "bad": bad, "maybe": maybe,
            "strict": ok / n if n else None, "strict_ci": (lo, hi),
            "lenient": (ok + maybe) / n if n else None, "lenient_ci": (lo2, hi2),
        }
    all_n = len(subset)
    all_ok = sum(1 for r in subset if r["verdict"] == "ok")
    out["overall"] = {"n": all_n, "ok": all_ok,
                      "strict": all_ok / all_n if all_n else None,
                      "strict_ci": wilson(all_ok, all_n)}
    return out


def main():
    ap = argparse.ArgumentParser(description="Analyse manual review verdicts")
    ap.add_argument("verdict_dir")
    args = ap.parse_args()

    rows = load(args.verdict_dir)
    print(f"loaded {len(rows)} verdicts\n")

    for benchmark in ("benchmark_a", "benchmark_b"):
        s = summarise(rows, benchmark)
        if not s["n"]:
            continue
        print("=" * 74)
        print(f"{benchmark}  --  {s['n']} messages reviewed by hand")
        print("=" * 74)
        print(f"{'group':<34s} {'n':>3s} {'ok':>3s} {'bad':>4s} {'amb':>4s} "
              f"{'correct':>8s} {'95% CI':>16s}")
        names = {"long": "longer than gate tested", "typical": "within tested range"}
        for stratum in ("long", "typical"):
            d = s["strata"][stratum]
            if not d["n"]:
                continue
            lo, hi = d["strict_ci"]
            print(f"  {names[stratum]:<32s} {d['n']:>3d} {d['ok']:>3d} {d['bad']:>4d} "
                  f"{d['maybe']:>4d} {d['strict']:>7.1%} {f'{lo:.1%} - {hi:.1%}':>16s}")

        lo, hi = s["overall"]["strict_ci"]
        print(f"  {'combined':<32s} {s['overall']['n']:>3d} {s['overall']['ok']:>3d} "
              f"{'':>4s} {'':>4s} {s['overall']['strict']:>7.1%} {f'{lo:.1%} - {hi:.1%}':>16s}")

        lg, tp = s["strata"]["long"], s["strata"]["typical"]
        if lg["n"] and tp["n"]:
            gap = tp["strict"] - lg["strict"]
            # Fisher's exact test, NOT "do the confidence intervals overlap".
            # That heuristic is one-directional and misleads here: non-overlap
            # implies a difference, but overlap does NOT imply no difference.
            # The calibration round overlapped at 100% vs 70% and still came
            # out significant at p=0.02.
            from scipy.stats import fisher_exact

            _, p = fisher_exact([[tp["ok"], tp["n"] - tp["ok"]],
                                 [lg["ok"], lg["n"] - lg["ok"]]])
            print(f"\n  gap (typical - long): {gap:+.1%}")
            print(f"  Fisher exact p = {p:.4f} -> "
                  f"{'a real difference' if p < 0.05 else 'not distinguishable at this sample size'}")
            print(f"  lenient (counting ambiguous as acceptable): "
                  f"long {lg['lenient']:.1%}, typical {tp['lenient']:.1%}")
        print()

    wrong = [r for r in rows if r["verdict"] == "bad"]
    if wrong:
        print("=" * 74)
        print(f"EVERY MESSAGE MARKED WRONG ({len(wrong)})")
        print("=" * 74)
        for r in wrong:
            print(f"  [{r['benchmark'][-1].upper()}/{r.get('stratum','?'):<7s}] "
                  f"id={r.get('item_id')}  assigned: {r.get('assigned')}")
            if r.get("note"):
                print(f"      note: {r['note']}")

    amb = [r for r in rows if r["verdict"] == "maybe"]
    if amb:
        print(f"\n{len(amb)} marked ambiguous:")
        for r in amb:
            print(f"  [{r['benchmark'][-1].upper()}/{r.get('stratum','?'):<7s}] "
                  f"id={r.get('item_id')}  assigned: {r.get('assigned')}")


if __name__ == "__main__":
    main()
