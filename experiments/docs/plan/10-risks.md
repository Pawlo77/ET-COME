# Risks, Mitigations & Regime of Applicability

## Risks and mitigations

### ν★ surrogate

Validate Spearman **ρ > 0.75** on ≥ 8/10 pilots before treating OT placement as a core claim. **Soft band ρ ∈ [0.65, 0.75]:** dual-report OT vs. SMOTE-within-A per [02-method.md](02-method.md). **Hard stop:** ρ < 0.50 on any pilot, or ρ < 0.65 on > 2 pilots — fall back to admissible-set SMOTE; do not title the method as transport-driven. The admissibility mask from Module A already excludes many high-overlap nodes. The formula treats P̂ and R̂ as locally constant; that approximation is **empirically** validated, not theoretical.

### Module C at severe imbalance
Study acceptance rate vs. IR on pilots. If < 20% at IR > 50: loosen α to 0.2, then 0.25; disable Module C only if still < 20% after both relaxations. Report final α per stratum.

### Admissibility collapse
Monitor |A| during pilots. If |A| < 2k on > 50% of datasets in a stratum: loosen q_e to 0.5 or q_a to 0.6, re-run pilots.

**Fallback:** If |A| < 2k on a specific dataset after tuning, use Borderline-SMOTE (Han et al., ICIC 2005) on full minority set (original, non-synthetic points only). Log event: (dataset_id, iteration_t, |A|_size, q_e, q_a, borderline_smote_n_generated).

### ENN over-cleaning at setup

On datasets where the minority class has high overlap with the majority, ENN may remove > 50% of minority points. The current safety floor (abort ENN if n_minority_cleaned < 20) handles the extreme case but does not systematically characterise when ENN is harmful. Module C's conformal quantile reliability degrades when the quantile is estimated from a noisy, uncleaned set after ENN exclusion.

**Systematic protocol:** On every dataset in the tuning corpus and full benchmark, log (n_minority_original, n_minority_post_enn, pct_removed). For each dataset where pct_removed > 30%: run the full pipeline both with and without ENN; record (AUPRC_with_enn, AUPRC_without_enn). If AUPRC_without_enn > AUPRC_with_enn + 1%, exclude ENN for that dataset and document the exclusion. Report ENN exclusion rate across the benchmark as a diagnostic statistic in the paper's appendix.

**Module C interaction:** When ENN is excluded due to over-cleaning, Module C's quantile is computed on the noisy, uncleaned minority set. In this regime, tighten α to 0.15 (stricter admission) to compensate for reduced quantile reliability. Log which datasets trigger this α override.

### Sinkhorn stability
Use log-domain Sinkhorn (Schmitzer, *SIAM J. Sci. Comput.* 41(3), 2019, arXiv:1610.06519) with ε ≥ 0.05. Validate on: (1) 2D synthetic example with known OT solution, (2) 5 real pilots confirming monotone loss decrease. If divergence: retry with higher ε; if still diverging, switch to L-BFGS solver.

### Majority-dominated k-NN (extreme IR)

When the HNSW graph is built on **all** training points, k-NN neighbourhoods of minority targets can be majority-heavy — distorts OT costs. Mitigation and minority-only graph variant: [02-method.md](02-method.md) Setup.

### Convergence cycling
Max 20 iterations is the hard backstop. Validate behaviour of ‖π^(t) − π^(t-1)‖₁ on 10 pilots (2 per IR stratum); monotone decrease on ≥ 8 of 10 is the go/no-go criterion for proceeding to Phase 9 (Optuna at scale). The paper does not claim convergence; it claims bounded termination and empirical stability on the benchmark datasets.

### Ensemble diversity safeguard
Track mean pairwise tree prediction correlation on OOB minority points per iteration. If correlation increases > 0.10 from iteration 0 baseline on ≥ 3 pilots, reduce step_size and increase feature subsampling randomness.

### Synthesis collision (multi-class)
When minority classes are close in feature space, synthetic points for class A may be rejected by secondary filter. **Proactive check:** Measure pairwise Bhattacharyya distance (Bhattacharyya, *Bull. Calcutta Math. Soc.* 35, 1943) between minority class pairs. If distance < 0.3, flag as high-overlap. If any class has < 10% acceptance rate, disable synthesis for that class and use class-weighted RF.

### Class imbalance pattern heterogeneity (multi-class)
Report results broken by number of minority classes (2 vs. 3+ vs. 5+).

### Module G — admissible set shrinkage (primary risk)
The adaptive γ* criterion with δ_A = 0.05 directly bounds this effect. **Proactive check (Phase 11):** Plot |A(D \ R_γ)| / |A(D)| vs. γ for γ ∈ [0, 0.60] to characterise the shrinkage curve.

### Module G — boundary overshoot at iterative synthesis
d_i^A constraint mitigates this: removed points are far from A, synthesis targets A. If Module D is active, extend safety zone by using d_i^A > 1.5 × θ_d.

### Module G — approximate HNSW recall
HNSW recall < 1.0 (typically 0.95–0.99; Malkov & Yashunin, 2020, Table 1). **Mitigation:** Use ef_search = 200 for safety-check query only → recall > 0.99.

### Module G — interaction with Module B's OOB F1 estimate
Removed points have high confidence → contribute ~0 false positives. P̂ = TP/(TP+FP) and R̂ = TP/(TP+FN) are both unchanged. Validate empirically: compare s_j rankings before/after Module G (require Spearman ρ > 0.95).

---

### Equal tree budget vs. RF baselines (reviewer risk)

At fixed tree count, **ExtraTrees** typically exhibit **lower tree correlation** than **RF**. ET-COME's pipeline uses ExtraTrees by design; SMOTE+RF baselines do not — see **Criterion 0** and correlation diagnostics in [08-evaluation.md](08-evaluation.md).

---

## Recommended Regime of Applicability

| Dimension | Range | Rationale |
|-----------|-------|-----------|
| Imbalance ratio | 1.5 ≤ IR ≤ 1000 | OBS benchmark coverage; extrapolation beyond 1000 not studied |
| Training set size | n ≥ 100 | OOB estimates require sufficient tree replicates |
| Feature dimensionality | d ≤ 500 | HNSW cost scales linearly; pilot validation up to d≈200 |
| Minority class size | n_minority ≥ 20 | Module C quantile estimation needs sufficient samples |
| Data type | Tabular, mixed numeric/categorical | Assumes Euclidean feature space |

**Empirical grounding (required in appendix):** Report **min / median / max** of (n, d, IR) across the **N** OBS datasets. The regime table is valid only insofar as published benchmarks occupy that hull — do not claim universal applicability beyond observed ranges.

### Not recommended for

- **Extreme imbalance (IR >> 1000):** Module C acceptance rate may collapse
- **Very small datasets (n < 100):** OOB infrastructure unstable
- **Very high-dimensional (d > 500):** HNSW becomes bottleneck
- **Non-tabular data:** Requires substantial adaptation
- **Binary-only when multi-class needed:** Use native multi-class extension

---

## When to Pivot or Redesign (Early-Stopping Rules)

During pilot validation (**Phases 5–7**), if **any** of the following occur on ≥ 5 of 10 pilot datasets within a stratum:

| # | Condition | Action |
|---|-----------|--------|
| 1 | Admissibility collapse: \|A\| < 10 on > 50% of datasets | Loosen q_e to 0.5; if still collapsed, document as regime limitation |
| 2 | Module C acceptance rate < 10% at any IR stratum | Disable Module C for that stratum; run A+B only |
| 3 | Sinkhorn not monotone or diverges on any pilot | Increase ε to ≥ 0.2; if still diverges, switch solver |
| 4 | Convergence: iterations > 20 or oscillate on > 30% of pilots | Lower ε_π (increase tolerance) |
| 5 | ET-COME underperforms SMOTE+RF on > 50% of pilots | **STOP.** Investigate root cause via ablation before grid search |
| 6 | Module G: γ* < 0.05 on > 50% of pilots | Disable Module G in default pipeline |

**If any early-stopping criterion is triggered, record the decision, cause, and outcome in a "Diagnosis Log" for transparency.**
