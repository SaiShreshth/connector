
=== A/B Test Summary ===

 - baseline: 1/2 successes, success_rate=0.5000
 - keyword_matcher: 2/5 successes, success_rate=0.4000
 - variant_a: 2/2 successes, success_rate=1.0000

Omnibus chi-square test (all variants):
  chi2: 2.115
  p_value: 0.3473
  dof: 2
  expected: [[1.111, 0.889], [2.778, 2.222], [1.111, 0.889]]
  variants: ['baseline', 'keyword_matcher', 'variant_a']

Pairwise comparisons (Bonferroni-corrected p-values):
 - variant_a vs baseline: 2/2 (1.000)  vs  1/2 (0.500)
   Δ=0.5, p_raw=1.0, p_bonf=1.0
   Note: Fisher's exact used due to small counts.
 - variant_a vs keyword_matcher: 2/2 (1.000)  vs  2/5 (0.400)
   Δ=0.6, p_raw=0.4286, p_bonf=1.0
   Note: Fisher's exact used due to small counts.
 - baseline vs keyword_matcher: 1/2 (0.500)  vs  2/5 (0.400)
   Δ=0.1, p_raw=1.0, p_bonf=1.0
   Note: Fisher's exact used due to small counts.

Interpretation hints:
 - Omnibus p_value < 0.05 → at least one variant differs significantly.
 - Pairwise p_bonf < 0.05 → significant difference between two variants.
 - With small data, collect more samples before making a decision.

🏆 Winner: variant_a (highest success rate 100.0%)
