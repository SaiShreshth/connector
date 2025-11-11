"""
analysis.py
------------
A/B testing analysis tool for comparing multiple variants such as
embedding_matcher, keyword_matcher, and baseline.

Usage:
    python -m app.core.ab_testing.analysis
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Path to event log file
EVENTS_FILE = os.path.join("app", "core", "ab_testing", "ab_events.csv")


def load_events():
    if not os.path.exists(EVENTS_FILE):
        print(f"[WARN] No events file found: {EVENTS_FILE}")
        return pd.DataFrame()

    df = pd.read_csv(EVENTS_FILE)
    if "result_label" not in df.columns or "variant" not in df.columns:
        print("[ERROR] Missing required columns in ab_events.csv")
        return pd.DataFrame()
    return df


def compute_variant_stats(df):
    """Compute per-variant success rates, latency, and costs."""
    stats = []
    for variant, group in df.groupby("variant"):
        total = len(group)
        success = (group["result_label"].str.lower() == "success").sum()
        success_rate = success / total if total else 0.0
        cost = group["cost_usd"].sum() if "cost_usd" in group else 0.0
        p50_latency = group["latency_ms"].median() if "latency_ms" in group else 0
        p95_latency = group["latency_ms"].quantile(0.95) if "latency_ms" in group else 0

        stats.append({
            "variant": variant,
            "total_requests": total,
            "success": success,
            "success_rate": round(success_rate, 4),
            "p50_latency_ms": round(p50_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "total_cost_usd": round(cost, 6)
        })
    return pd.DataFrame(stats)


def omnibus_chi_square(df):
    """Run omnibus chi-square test across all variants."""
    contingency = []
    variants = []
    for variant, group in df.groupby("variant"):
        success = (group["result_label"].str.lower() == "success").sum()
        fail = len(group) - success
        contingency.append([success, fail])
        variants.append(variant)

    if len(contingency) < 2:
        return {"note": "Need at least 2 variants for chi-square test."}

    chi2, p, dof, expected = chi2_contingency(contingency)
    return {
        "chi2": round(chi2, 4),
        "p_value": round(p, 4),
        "dof": dof,
        "expected": np.round(expected, 3).tolist(),
        "variants": variants
    }


def pairwise_tests(df):
    """Run pairwise Fisher exact tests between variants."""
    results = []
    variants = df["variant"].unique().tolist()

    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            v1, v2 = variants[i], variants[j]
            g1 = df[df["variant"] == v1]
            g2 = df[df["variant"] == v2]

            s1 = (g1["result_label"].str.lower() == "success").sum()
            f1 = len(g1) - s1
            s2 = (g2["result_label"].str.lower() == "success").sum()
            f2 = len(g2) - s2

            # Fisher’s exact (more reliable for small samples)
            _, p_raw = fisher_exact([[s1, f1], [s2, f2]])
            p_bonf = min(p_raw * 3, 1.0)

            results.append({
                "pair": f"{v1} vs {v2}",
                "rate_v1": f"{s1}/{len(g1)} ({s1/len(g1):.3f})",
                "rate_v2": f"{s2}/{len(g2)} ({s2/len(g2):.3f})",
                "diff": round((s1/len(g1)) - (s2/len(g2)), 4),
                "p_raw": round(p_raw, 4),
                "p_bonf": round(p_bonf, 4)
            })
    return pd.DataFrame(results)


def main():
    df = load_events()
    if df.empty:
        print("No events found. Run some simulations first.")
        return

    print("\n=== A/B Test Summary ===\n")

    stats = compute_variant_stats(df)
    for _, row in stats.iterrows():
        print(f" - {row['variant']}: {row['success']}/{row['total_requests']} successes, "
              f"success_rate={row['success_rate']:.4f}")

    print("\nOmnibus chi-square test (all variants):")
    result = omnibus_chi_square(df)
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\nPairwise comparisons (Bonferroni-corrected p-values):")
    pw = pairwise_tests(df)
    for _, row in pw.iterrows():
        print(f" - {row['pair']}: {row['rate_v1']}  vs  {row['rate_v2']}")
        print(f"   Δ={row['diff']}, p_raw={row['p_raw']}, p_bonf={row['p_bonf']}")
        print("   Note: Fisher's exact used due to small counts.")

    print("\nInterpretation hints:")
    print(" - Omnibus p_value < 0.05 → at least one variant differs significantly.")
    print(" - Pairwise p_bonf < 0.05 → significant difference between two variants.")
    print(" - With small data, collect more samples before making a decision.")

    # --- 🏆 Winner calculation section ---
    try:
        success_rates = stats.set_index("variant")["success_rate"].to_dict()
        if success_rates:
            best_rate = max(success_rates.values())
            winners = [v for v, r in success_rates.items() if r == best_rate]
            if len(winners) == 1:
                print(f"\n🏆 Winner: {winners[0]} (highest success rate {best_rate * 100:.1f}%)")
            else:
                ties = ", ".join(winners)
                print(f"\n🤝 Tie between: {ties} (equal success rate {best_rate * 100:.1f}%)")
        else:
            print("\n⚠️ No success_rate data available to determine a winner.")
    except Exception as e:
        print(f"\n⚠️ Could not determine winner: {e}")


if __name__ == "__main__":
    main()
