# Evaluation

## Benchmarks (open-source only)

**Scope:** All datasets must have **public downloads** under licenses permitting redistribution and benchmarking (e.g. KEEL, UCI via permissive terms). **No registration-gated or proprietary-only suites** — removes dependency on unpublished or restricted corpora.

**Binary — primary:** Pre-registered **Open Binary Suite (OBS)** assembled from **KEEL imbalance collection** and compatible **public UCI/OpenML** binary tasks (exact list and **N** locked at OSF/AsPredicted **before Phase 1 closes — dataset manifest gate**). Target **N ≥ 60** datasets spanning IR strata.

**Binary — supplementary:** Additional KEEL / public datasets for comparability with prior KEEL-era papers (disjoint from tuning corpus).

**Multi-class:** KEEL multi-class imbalance suite (≥ 15 datasets; see [03-multiclass.md](03-multiclass.md)). Report binary and multi-class results in separate tables.

**Before Phase 2:** Pre-register exact dataset lists (URLs + hashes), 20-dataset ablation holdout, and 20-dataset HP tuning corpus at AsPredicted or OSF.

---

## Metrics

### Binary

| Metric | Role |
|--------|------|
| AUPRC (Davis & Goadrich, ICML 2006) | Primary — standard for severe imbalance on tabular benchmarks |
| Macro-F1 | Directly optimised by ν★; standard in ensemble comparisons |
| G-mean (Kubat & Matwin, ICML Workshop 1997) | KEEL-era comparability |
| AUROC | Completeness; less sensitive to IR |
| Wall-clock at equal tree budget | Efficiency — reported at equal total tree count |

### Multi-class

| Metric | Role |
|--------|------|
| Macro-AUPRC (Davis & Goadrich, ICML 2006) | Primary — average AUPRC over all classes |
| Macro-F1 | Standard multi-class metric |
| Per-class AUPRC | Breakdown showing which classes benefit most |
| G-mean macro (Kubat & Matwin, ICML Workshop 1997) | KEEL multi-class comparability |

---

## Baselines

### Tier 0 — lower bounds that must be beaten

| Method | Why it matters | Tuning Strategy |
|--------|---|---|
| RF without resampling | True floor | n_estimators=100, max_depth=None |
| LightGBM `is_unbalance=True` | Frequently beats naive oversampling | Tune n_estimators ∈ {50, 100, 150} to match tree budget |
| Class-weighted RF | Same argument; native multi-class | `class_weight='balanced'`; n_estimators=100 |
| **ExtraTrees `class_weight='balanced'`** | **ET-COME's own base learner with class reweighting — required diagnostic: oversampling pipeline must beat this** | `bootstrap=True, oob_score=True`; n_estimators=100; identical configuration to ET-COME's E⁰ |

### Tier 1 — preprocessing oversamplers

Binary: SMOTE (Chawla et al., *JAIR* 16, 2002), Borderline-SMOTE (Han et al., ICIC 2005), ADASYN (He et al., IEEE WCCI 2008), SMOTE+ENN (Batista et al., *SIGKDD Explorations* 6(1), 2004), KWSMOTE (Li et al. 2025; if unavailable, document exclusion).

Multi-class: imblearn (Lemaître et al., *JMLR* 18(17), 2017) handles SMOTE/Borderline-SMOTE/ADASYN natively. All tuned on same 20-dataset tuning corpus before benchmark. Run RF with n_estimators matched to ET-COME total tree count.

### Tier 2 — ensemble methods

EasyEnsemble (Liu et al., *IEEE Trans. SMC* 39(2), 2009), BalancedRandomForest (Chen et al., UC Berkeley TR, 2004), RUSBoost (Seiffert et al., *IEEE Trans. SMC-A* 40(1), 2010). Implementation deadline: Phase 3, week 1.

**Tuning strategy:** (1) Implement with standard hyperparameters first, (2) Tune on 20-dataset corpus if time permits, (3) Report tuned vs. default in supplementary.

### Tier 3 — recent strong methods (contingent)

Only include if implementation completed by Phase 5 end:
- MESA (NeurIPS 2020): arXiv:2010.08349
- ART (arXiv:2509.00955, 2025)
- TabPFN v2 (Nature 2024): only if n < 10K
- DGOT (IEEE TKDE 2025)
- TabDDPM (ICML 2023, arXiv:2209.15421): restrict to n < 10,000

**Deterministic inclusion rule:** Include only if: (a) public implementation exists, (b) license permits benchmarking, (c) method runs on ≥ 3 pilot datasets without failure. If any condition fails, freeze exclusion before Phase 6.

**Claim governance when Tier 3 is missing:**
- Allowed: superiority vs. Tier 0–2 baselines under equal budget
- Disallowed: "state-of-the-art across all recent methods"
- Required: disclosure table in appendix (method, missing reason, date checked, URL)

### Tier 4 — ET-COME ablations

See [07-ablation.md](07-ablation.md). Run on same 20-dataset holdout.

---

## Equal tree budget and base-learner fairness

All methods use **equal total tree count**:
- RF: n_estimators = 100
- ET-COME: T₀ + (L × step_size) = 100
- EasyEnsemble: total trees across sub-models = 100
- LightGBM: n_estimators tuned to ≈ 100

**Structural caveat (report honestly):** ET-COME uses **bootstrapped ExtraTrees**; Tier 1–2 baselines often use **RF** at equal tree count — ExtraTrees have lower tree-to-tree correlation than RF at fixed **n_estimators**, so diversity per tree may differ. **Mitigations:** (1) **Criterion 0** compares against **class-weighted ExtraTrees** (same base learner); (2) **Main paper (not appendix-only):** report **mean pairwise OOB prediction correlation** on minority points for ET-COME vs. SMOTE+RF vs. EasyEnsemble — readers must see this diagnostic prominently; (3) optional sensitivity: match methods by **correlation budget** (post-hoc) on a subset if reviewers request.

**Proof-of-concept pilots (internal credibility — Gate A):** Run **one untuned** ET-COME vs. SMOTE+RF on **3 KEEL datasets** (low / medium / high IR strata). Not for publication — **go/no-go for Gate A:** if ET-COME does not beat SMOTE+RF on **≥ 2 of 3**, pause for scope review **before Phase 3** ([11-roadmap.md](11-roadmap.md) Gate A). Same runs feed **termination split** (dual vs backstop) for **title governance** pilots ([08-evaluation.md](08-evaluation.md) §Title and abstract governance).

**Reporting:** Show (method, total_trees, mean_AUPRC, std_AUPRC). Verify all methods have total_trees ∈ [95, 105].

**Wall-clock measurement:** Per-dataset training+prediction runtime at equal tree budget, excluding HP tuning. Run on same machine, report CPU/RAM. Benchmarks with `debug_mode=False`.

---

## Statistical tests

Bayesian signed-rank test (Benavoli, Corani, Demšar, Zaffalon, *JMLR* 18(1), 2017) as primary — posterior probability of superiority, not a binary p-value. McNemar test (Dietterich, *Neural Computation* 10(7), 1998) per-dataset for pairwise comparisons. Win/draw/loss counts with ±0.5% AUPRC practical equivalence region (ROPE concept from Benavoli et al., 2017, §3.2). Outer 5-fold CV repeated 5× with different seeds. All methods use identical fold assignments.

---

## Power analysis (pre-registered numbers)

**Primary:** Effect size δ = **0.02** AUPRC (absolute); pooled standard deviation **σ** estimated from **Phase 5–7 pilots** (heterogeneous OBS-style folds — **do not** treat σ as fixed before pilots). α = 0.05; Bayesian signed-rank framing per Benavoli et al. (2017).

**σ decision rule (locked at OSF before Phase 5 — avoids post-hoc power):** After pilots, compute pooled **σ̂** across dataset-level ΔAUPRC. **If σ̂ ≤ 0.07:** keep registered **N** and δ = 0.02 if power ≥ 85%. **If σ̂ > 0.07:** **either** (i) **increase N** (solve numerically for N such that power ≥ 85% at δ = 0.02 given σ̂ — document formula/simulation in OSF) **or** (ii) register **revised minimum detectable effect δ_min = 0.03** at original **N**. Do not silently claim 85% power at δ = 0.02 when σ̂ implies otherwise.

**Approximate calibration:** At σ = 0.05, **N ≈ 70** datasets with **25** repeats often yields high replication for δ = 0.02 — verify by simulation. If true advantage is **1% AUPRC**, win-rate Criteria **1–3** dominate interpretation.

**Requirement:** OSF embeds **conditional branches** (σ̂ ≤ 0.07 vs. > 0.07) **before Phase 5**. Final numbers updated once after pilots — **before Phase 12** full OBS — with amendment log.

---

## Stratify by imbalance ratio

Report results stratified by IR ∈ [1.5, 10), [10, 30), [30, ∞). Pre-register **IR-monotonicity** test (Criterion 4 below).

---

## Minimum bar for publication

All **12 numbered criteria** must be met simultaneously. Not any subset.

**Threshold rationale (non-circular).** Asymmetric win-rate thresholds are justified **only** by **anticipated effect sizes relative to baseline strength**, not by ET-COME's internal design story:
- **LightGBM:** Strong tuned GBDT — historically competitive on tabular tasks; expected ET-COME advantage **smaller** than vs. vanilla SMOTE+RF → threshold **60%** win rate if pre-registered effect-size prior supports it.
- **EasyEnsemble:** Strong ensemble diversity — **55%** reflects expected harder head-to-head.
- **SMOTE+RF:** Weaker mechanistic baseline — **70%** reflects expectation of larger gap **if** oversampling logic matters; this is **not** "because ET-COME supersedes SMOTE" (that would beg the question). Pre-register numeric priors in OSF before results.

**OSF obligation:** Lock **point estimates or intervals** for expected win rates vs. each tier (documented as subjective priors + literature, not data-derived) at the same time as the thresholds — satisfies reviewer demand that asymmetry is not post-hoc tuning.

### Contribution accounting across fallback regimes (reviewer objection — "what is the method after fallbacks?")

Define **ET-COME_min**: the conservative stack actually run when ν★ validation fails and Module B uses **SMOTE-within-A only**, Module C may be **off** at extreme IR, and other gates match [02-method.md](02-method.md). Define **ET-COME_full**: all modules active with OT passing ρ > 0.75 validation.

**Required reporting:**
1. **Versus geometric residual:** Compare **ET-COME_min** to **Baseline-A + Borderline-SMOTE** (same folds) — establishes value above "epistemic-mask Borderline-SMOTE" floor.
2. **Novelty delta:** Report **Δ = metric(ET-COME_full) − metric(ET-COME_min)** (and per-module ablations in [07-ablation.md](07-ablation.md)) so OT / screening / G are not credited without incremental evidence.
3. **Per-stratum module table:** Which of A–G are active per IR stratum (already required in [02-method.md](02-method.md) fallback statement).

**Minimum publishable claim across all stacks:** Module **A** — OOB-based admissibility targeting validated against **Borderline-SMOTE** and **Baseline-A-simple** (ϕ_total-only). Higher layers are **additive** conditional on their gates.

**Abstract alignment:** The abstract must summarize **per-stratum** behaviour when modules differ — not only best-case OT-active runs.

### Prerequisite diagnostic (Criterion 0 — publication framing)

0. **Class-weighted ExtraTrees:** P(ET-COME-ABCG > ExtraTrees `class_weight='balanced'`) > 0.75 on ≥ **65%** of **N** OBS datasets — raised from a bare majority so the method is not “useful” while losing to reweighting ~half the time.

**Decision tree (must appear in paper supplement):**
- If Criterion 0 **passes** → proceed with standard oversampling narrative when Criteria 1–12 pass.
- If Criterion 0 **fails** **and** **≥ 2** of Criteria **1–3** fail → manuscript pivots to **negative result / conditional value** framing: *under what IR / data regimes does synthesis beat class-weighted ExtraTrees?* Oversampling-only novelty claims are withdrawn.
- If Criterion 0 fails **but** **≤ 1** of Criteria 1–3 fails → **mixed result**: headline ET-COME vs. weak baselines only; dedicated subsection on class-weighted ET.

### Binary results (OBS)

Let **N** denote pre-registered open binary benchmark count.

1. **LightGBM:** P(ET-COME > LightGBM) > 0.75 on ≥ **60%** of **N** datasets
2. **EasyEnsemble:** P(ET-COME > EasyEnsemble) > 0.75 on ≥ **55%** of **N** datasets
3. **SMOTE:** P(ET-COME > SMOTE+RF) > 0.75 on ≥ **70%** of **N** datasets at equal tree count

4. **IR-monotonicity:** Fit **linear regression** of per-dataset **ΔAUPRC** (ET-COME minus **best Tier 0–1 baseline per dataset**) on **log(IR)** using **all N** OBS datasets (one point per dataset). **Require simultaneously:**
   - **(a)** Slope **> 0** with **p < 0.05**
   - **(b)** **95% confidence interval lower bound on slope > 0**
   - **(c)** **R² > 0.15** (weak-but-non-trivial explained variance — **not** token 5%)
   - **(d)** Stratum-mean ΔAUPRC **> 0** in **each** of the three IR strata **[1.5,10), [10,30), [30,∞)**

   If slope > 0 but **R² < 0.15**, report **"weak monotone association — IR-driven design validation is not established."**

   **Why reviewers won’t dismiss this as a token rule.** A threshold like **R² > 0.05** is easy to satisfy with noise and one influential point — it reads as “technically satisfied” but not defensible at NeurIPS/ICML. The bundle **(b) + (c)** requires a **positive slope CI** *and* **R² > 0.15**, which targets ~**15%** explained variance in cross-dataset ΔAUPRC from log(IR) — weak by social-science standards but **meaningfully stricter** than 5%. **Optional appendix robustness:** Spearman **ρ** between **log(IR)** and **ΔAUPRC** with permutation **p < 0.05** (no claim if regression and rank tests disagree — report the tension).

5. **Efficiency:** Wall-clock ≤ 1.2× baseline RF on ≥ 75% of **N** datasets
6. **Module independence:** Both Baseline-AB and Baseline-AC beat SMOTE+RF on ≥ 50% of ablation holdout (P > 0.7)
7. **Module G:** **No thresholds here** — use **[02-method.md](02-method.md) §Module G — Decision rule for inclusion (Phase 11 ablation)** only. **Outcome 1** → full Module G exposition in main paper; **Outcome 2** → appendix-focused G with ≤ 1 paragraph main text; **Outcome 3** → G excluded from default pipeline narrative.

### Multi-class results (KEEL)

8. **RF comparison:** P(ET-COME > class-weighted RF) > 0.75 on ≥ 55% of multi-class datasets
9. **SMOTE comparison:** P(ET-COME > SMOTE multi-class) > 0.75 on ≥ 65% of multi-class datasets
10. **Extension consistency:** No multi-class dataset performs > 2% worse than equivalent-IR binary result

### General requirements

11. **Bounded termination (not "convergence"):** Mean iterations ≤ 15 on ≥ 90% of **N** datasets. **Reporting obligation:** for each dataset, log termination cause (**dual criterion** vs. **max_iter backstop** vs. early oscillation stop). If aggregate iterations ≤ 15 but **most** stops are backstop at iterations **15–19**, describe results as **bounded iteration budget**, not equilibrium convergence — see [02-method.md](02-method.md) Module E.

12. **OOB leakage sensitivity:** v2 AUPRC changes < 1% under corrupted-validation-label stress test; v1 degrades ≥ 3%

### Title and abstract governance (branding — OSF-registered)

**Default until proven:** Pre-register the **fallback title** (no “Equilibrium”) as the **working manuscript title** through Phase 7 pilots. **Promote** to **Equilibrium …** in the title **only after** either **(i)** ≥ **70%** of **10** Phase 7 pilot datasets terminate via **dual criterion**, **or** **(ii)** ≥ **70%** of full **OBS** — document which gate triggered. That avoids a branding mismatch where the PDF says “Equilibrium” while most runs hit **max_iter** ([02-method.md](02-method.md) Module E).

- **Final publication rule:** **"Equilibrium"** in the title only if ≥ **70%** of **N** OBS datasets terminate **primarily** via the **dual convergence criterion** (Module E), **not** via **max_iter** as the effective stop; otherwise keep fallback title or omit “Equilibrium”.
- **Abstract** must summarize **per-stratum** module activation when OT/C/G differ — not only the best-case stack.

---

## Failure protocol

- **Single criterion fails (1 of 12):** Investigate root cause. Criterion (4) failure = required pivot. Others may be acceptable.
- **Partial IR-monotonicity:** If regression slope positive but **R² < 0.15** or CI includes 0, document weak evidence per Criterion 4.
- **Multiple criteria fail (≥ 2, especially criterion 4):** Fundamental issue. Options: redesign, negative result paper, or pivot.
- **Contingencies:** Tier 3 unavailable → proceed with Tier 0–2. < 15 multi-class datasets → use 10–12.

---

## Leakage audit

**Protocol:** On 10 KEEL pilots (2 per IR stratum):
1. Run both v1 (uses validation in loop) and v2 (OOB-only) with identical 5-fold CV splits
2. Record test-fold AUPRC under clean labels
3. Rerun v1 with 30% of minority validation labels randomly flipped
4. v2 should show < 1% change (never uses validation fold)

**Expected:** v1 degrades ≥ 3%; v2 changes < 1%. If v2 degrades ≥ 2%, investigate hidden validation dependency.

---

## Logical consistency checks

Prevent gaming via hyperparameter tuning:

1. **Criterion (4) gates interpretation of (1)–(3):** If IR-monotonicity **does not meet** **R² > 0.15** (or slope **95% CI** includes 0), temper causal claims about IR-driven advantage
2. **Baseline fairness:** All methods tuned on same 20-dataset corpus
3. **Equal budget audit:** Verify total_trees ∈ [95, 105] per method per dataset
4. **Criterion (6) independence:** Baseline-AB and Baseline-AC trained independently
5. **IR-stratified consistency:** Report per-stratum results in supplementary
6. **No moving goalposts:** Criterion 0 and all 12 numbered criteria locked at AsPredicted **before Phase 5 pilot experiments**
