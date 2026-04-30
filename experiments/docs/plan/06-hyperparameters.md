# Hyperparameters

## Parameters and provisional starting points

| Parameter | Module | Provisional default | First-principles rationale |
|-----------|--------|---------------------|---------------------------|
| q_e | A | 0.7 | Top 30% epistemic — targets clear learnable uncertainty |
| q_a | A | 0.5 | Bottom 50% aleatoric — excludes noisy overlap regions |
| k | B | 15 | HNSW neighbourhood; determines graph topology and OT support |
| ε | B | 0.1 | Sinkhorn regularisation; log-domain solver needed below 0.05 |
| τ | B | 1.0 | Softmax temperature for ν★; 1.0 = no distortion as baseline |
| α | C | 0.1 | OOB screening level; 90% minority consistency |
| γ_max | G | 0.50 | Upper bound for adaptive removal; actual γ* ≤ γ_max via bisection |
| δ_A | G | 0.05 | Admissibility preservation tolerance; allows ≤5% shrinkage of A |
| step_size | E | 5 | Trees added per iteration; inherited from v1 |
| max_iter | E | 20 | Hard backstop — convergence criterion usually stops earlier |

These are starting points only. The grid search below determines the final reported defaults.

---

## Why IR-stratified grid search is required

Optimal parameter values interact with imbalance ratio in non-trivial ways. At low IR, a tight admissibility mask (high q_e, low q_a) may exclude too many minority points. At extreme IR (> 100), a loose α admits too many implausible synthetic points because the score quantile is estimated from very few real minority examples. A single global default optimised on mixed-IR datasets will be suboptimal. The grid search must be run separately within each IR stratum.

---

## Tuning corpus

Use **20 KEEL datasets held out from the OBS primary benchmark**, chosen to balance representation across IR strata:

| Stratum | IR range | Datasets |
|---------|----------|----------|
| Low | [1.5, 10) | 5 datasets |
| Medium | [10, 30) | 5 datasets |
| High | [30, 100) | 5 datasets |
| Extreme | ≥ 100 | 5 datasets |

These 20 datasets are pre-registered before any search begins. They must be **disjoint** from the OBS primary benchmark list and from the ablation holdout.

---

## Search grid

| Parameter | Values | Motivation & Citation |
|-----------|--------|-----------|
| q_e | {0.3, 0.4, 0.5, 0.6, 0.7} | Spans from permissive (70% of minority admissible) to strict (30% admissible). Values below 0.3 make admissibility meaningless; values above 0.7 risk admissibility collapse at low IR. The epistemic/aleatoric decomposition via mutual information follows Depeweg et al. (2018) and Smith & Gal (2018, arXiv:1803.08533). |
| q_a | {0.2, 0.3, 0.4, 0.5} | Controls overlap noise tolerance. q_a = 0.2 excludes only the noisiest 20%; q_a = 0.5 excludes the noisiest half. Values above 0.5 conflict with q_e, collapsing A. Grounded in Kendall & Gal (NeurIPS 2017, arXiv:1703.04977); Gawlikowski et al. (2023, arXiv:2107.03342). |
| k | {10, 15, 20} | HNSW neighbourhood size. Malkov & Yashunin (2018, arXiv:1603.09320) show recall > 0.95 for ef_construction ≥ 2×M; our k ∈ [10, 20] aligns with recommended ef_search range. k=15 matches t-SNE perplexity-equivalent (van der Maaten & Hinton, JMLR 2008). |
| ε | {0.05, 0.1, 0.2} | Cuturi (NeurIPS 2013, arXiv:1306.0895) introduced ε-regularised OT; Peyré & Cuturi (FnTML 2019, arXiv:1803.00567, §4.2) recommend ε ∈ [1/n, 1]. Schmitzer (SISC 2019, arXiv:1610.06519) shows log-domain stabilisation required below ε ≈ 0.05. |
| τ | {0.5, 1.0, 2.0} | Temperature formalised by Hinton et al. (NIPS 2014, arXiv:1503.02531). Guo et al. (ICML 2017, arXiv:1706.04599) show τ ∈ [0.5, 2.5] is calibration-relevant. |
| α | {0.1, 0.15, 0.2} | Follows conformal principles (Vovk et al., 2005). Romano et al. (NeurIPS 2020) establish α ∈ [0.05, 0.20] as standard miscoverage range. α < 0.1 too strict for small n_minority (Lei et al., JASA 2018). |
| γ_max | {0.15, 0.30, 0.50} | Literature on RUS (Liu et al., IEEE TSMC 2009; Drummond & Holte, 2003) shows removing up to 50% of majority is common without catastrophic loss. |
| δ_A | {0.03, 0.05, 0.10} | Novel criterion. 3% < detection threshold of TPE; 10% is where OT plan sensitivity becomes measurable (Peyré & Cuturi §4.4). |
| step_size | {3, 5, 10} | Breiman (2001) shows OOB error stabilises after ~50 trees. Geurts et al. (2006) demonstrate ExtraTrees converge faster, justifying step_size up to 10. |

max_iter is not grid-searched (safety backstop). θ_d is adaptive (median-based) and not grid-searched.

**Total search space:** 5×4×3×3×3×3×3×3×3 = **43,740 unique configurations**.

---

## Tree / ensemble hyperparameters — why they are fixed, not searched

**v1 searched** 108 tree combinations × n_estimators ∈ {100, 500}. **v2 does not search tree hyperparameters** for three reasons:

1. **ET-COME self-corrects via OOB feedback.** Tree hyperparameters affect convergence *speed* but not the *equilibrium*. This is analogous to Breiman's bias-variance decomposition (*Machine Learning* 45(1), 2001, §4). Searching conflates tree quality with ET-COME module quality.

2. **Equal-budget fairness.** Probst et al. (*WIREs Data Mining* 9(3), 2019) show tuning RF hyperparameters provides 1–3% accuracy gains. If ET-COME optimises tree internals while baselines use defaults, the comparison is unfair. Fix at sklearn defaults (Geurts et al., *Machine Learning* 63(1), 2006): max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='gini', max_features='sqrt'.

3. **Computational budget.** Adding 108 tree combinations would yield ~4.7M configurations. Bergstra & Bengio (JMLR 2012, §5) show d_eff ≈ 2–3 for trees, largely orthogonal to ET-COME parameters.

**Post-hoc validation:** Fix module parameters at optimum, vary (max_depth, max_features) over {(None, 'sqrt'), (15, 'sqrt'), (None, None)} on 5 pilots. If > 2% AUPRC improvement, add to benchmark. Follows Bischl et al. (*WIREs Data Mining*, 2023, arXiv:2107.05847) two-stage protocol.

---

## Experiment count budget

| Pilot / phase | Experiment type | Datasets | Configs/trials | CV folds | Seeds | Fits/trial | Model fits |
|-------|----------------|----------|---------------|----------|-------|-----------|------------|
| Phase 5 | Module A/C/G pilot validation | 5 | 1 | 5 | 1 | 5 | 25 |
| Phase 6 | Spearman ρ + Sinkhorn validation | 5 | 1 | 5 | 1 | 5 | 25 |
| Phase 7 | Full loop convergence + leakage audit | 10 | 2 | 5 | 1 | 5 | 50 |
| Phase 9 | Optuna search (4 strata × 300 trials) | 20 | 300/stratum | 5 | 1 | 25 | **30,000** |
| Phase 9 | Post-hoc tree validation | 5 | 3 | 5 | 1 | 5 | 75 |
| Phase 11 | Ablation (all configs in [07-ablation.md](07-ablation.md)) | 20 | ≥11 | 5 | 5 | 25 | ~5,500 |
| Phase 12 | Full benchmark — ET-COME | **N** | 1 | 5 | 5 | 25 | 25N |
| Phase 12 | Full benchmark — baselines (×8) | **N** | 8 | 5 | 5 | 25 | 200N |
| Phase 12 | KEEL binary supplement | ~30 | 9 | 5 | 5 | 25 | 6,750 |
| Phase 13 | Multi-class Optuna (4 strata × 300) | 15 | 300/stratum | 5 | 1 | 20 | **24,000** |
| Phase 13 | Multi-class benchmark | 15 | 9 | 5 | 5 | 25 | 3,375 |
| | | | | | | **Total** | **scales with N** (~84k when N≈73) |

**Key points:**
- Search space (43,740) ≠ model fits (total scales with **N**)
- Optuna dominates when Phase 9 runs at full trial count
- **Realistic planning multiplier:** Add **1.5–2×** to tabulated fits for pilot restarts, Module G bisection refits on benchmark scale, ENN @ >30% removal sensitivity pairs, and failed-gate reruns — treat ~84k core fits as **~125k–168k** engineering fits unless proven otherwise.
- At ~30 seconds/fit and N≈70: order **~700** CPU-hours for Optuna-heavy phases; **double** if median fit ≈ 60–120 s; at **120 s/fit** upper envelope approaches **~5,600 CPU-hours** for optimistic-count workloads — use contingency ladder **before** committing calendar time.

**Hardware specification and compute validation required (Phase 1 — before committing to this budget):**
1. Specify exact hardware (CPU model, core count, RAM) and confirm access. Document in the pre-registration record.
2. Benchmark fit time on the **5 largest OBS datasets** (by n × d product) at T₀ = 50 trees and default module settings. Record median **m** seconds per fit.
3. **Contingency ladder (mandatory — pick one before Phase 9):**
   - If **m ≤ 60 s:** proceed with planned 300 trials/stratum (or adjust per Optuna parallelism caveat below).
   - If **60 s < m ≤ 120 s:** **either** (i) halve Optuna trials per stratum (150) with documented sensitivity check on 20-dataset corpus, **or** (ii) extend calendar timeline ~2×; **or** (iii) obtain cluster allocation — specify which branch in OSF amendment **before** burning budget.
   - If **m > 120 s:** **first** pursue cluster/cloud within **48 hours** of Phase 1 close.
   - **Branch 4 — cluster denied / unavailable within 48 h:** **mandatory:** reduce **N** from ≥60 to **40**, reduce Optuna to **150 trials/stratum**, **re-run power analysis** in OSF amendment **within Phase 1**; if **85% power** at δ=0.02 cannot be maintained at N=40 given pilot **σ̂**, register **δ_min = 0.03** or increase **N** via extra open datasets — **do not** proceed with N=60 at degraded per-fit time without an amended registered plan.
4. **Optuna parallelism caveat:** TPE's surrogate model requires sequential observations to update the density estimate. Asynchronous parallel TPE (n_jobs > 1) reduces sequential observation quality proportionally to the degree of parallelism. When using ≥ 4 parallel workers, increase trials from 300 to 360 per stratum to compensate for reduced TPE efficiency. Report whether parallel or sequential TPE was used, the Optuna version, and n_jobs in the paper's experimental setup.

**Phase 6 profiling gate:** Before locking Phase 6 duration, time **one full Sinkhorn solve** on the **largest OBS pilot** (max n×d). Confirm OT overhead remains in line with "< ~5% per-iteration wall-clock" claim in [05-complexity.md](05-complexity.md) or revise wording.

---

## Search protocol

**Optuna with TPE sampler** (Akiba et al., KDD 2019, arXiv:1907.10902), sampling 300 configurations per stratum (0.7% coverage), reducing binary HP search to 300 × 4 strata × 5 datasets × 5 folds = **30,000 model fits** — a 145× reduction vs. exhaustive.

### Why TPE over alternatives

- **TPE** (Bergstra et al., NeurIPS 2011): models P(x|y < y★) and P(x|y ≥ y★) with KDE, concentrating trials in promising regions. Handles mixed continuous/discrete and conditional dependencies natively.
- **Random search** (Bergstra & Bengio, JMLR 2012): requires O(1/δ²) trials for δ-close-to-optimal; TPE converges ~3× faster empirically.
- **GP BO** (e.g., SMAC, Hutter et al., LION 2011): scales as O(n³) in trials; impractical above ~200 trials. TPE scales linearly.

### Why 300 trials/stratum

- Bergstra et al. (2011) show TPE achieves < 1% optimality gap within 200 trials for 10-D spaces
- Our effective dimensionality is ~4–5 (per Bergstra & Bengio 2012)
- 300 = 200 (sufficient) + 100 (safety margin). Matches Optuna benchmarks (Akiba et al., 2019, §4.2)
- Watanabe (2023, arXiv:2305.17595) confirms multi-fidelity could reduce further, but we use full-fidelity

The objective is mean AUPRC across all datasets in the stratum, evaluated with 5-fold stratified CV per dataset.

**Fold-level leakage guard:** all preprocessing and resampling are fit on the training fold only.

Run four independent searches (one per stratum). Compare optima: if they differ by > one grid step, report IR-conditional defaults.

---

## Parameter sensitivity report

After Optuna search, produce a **one-way sensitivity plot** for each parameter: fix all others at stratum optimum, vary the target across its full range, report mean AUPRC ± std.

The (k, ε) interaction deserves a **two-way plot** — a sparser graph (low k) may need more regularisation (high ε) for Sinkhorn stability.

---

## What if first-principles defaults are already near-optimal?

This is the preferred outcome: it validates the design rationale. Report explicitly — "provisional defaults fall within one grid step of Optuna optimum on 17 of 20 tuning datasets." If the opposite is true, update defaults and explain which parameters are most sensitive.
