# Core Method

## Setup (runs once)

Clean the training data with ENN (Wilson, *IEEE Trans. SMC* 2(3), 1972). Fit a StandardScaler on the cleaned data only (not on removed outliers, not on validation or test). Scale all features — required because the HNSW graph (Malkov & Yashunin, *IEEE TPAMI* 42(4), 2020, arXiv:1603.09320) uses L2 distance and unscaled features corrupt neighbourhood structure. Build one HNSW graph on the **full training set** (majority + minority). This graph never changes.

**Majority-dominated k-NN diagnostic (extreme IR):** For each admissible minority node j, inspect its k HNSW neighbours; report **fraction that are majority-class**. Stratify by IR. If **> 50%** majority neighbours in the **[30, ∞)** stratum, **also** build an auxiliary HNSW graph on **minority-class points only** (same scaled features), recompute restricted k-NN for OT cost on that graph, and compare transport stability vs. full-data graph on a pilot subset — document whether minority-only topology is adopted per stratum in the Method Development Log.

**ENN safety floor:** After ENN cleaning, if n_minority_cleaned < 20 for any class, abort ENN for that class and use the original (uncleaned) minority set. Log this event as (dataset_id, class_label, n_minority_original, n_minority_post_enn, reason: 'ENN-collapse'). If n_minority_cleaned < 0.5 × n_minority_original on any class, log a warning regardless of whether the floor was triggered — aggressive cleaning at this scale degrades Module C's conformal quantile estimates.

**Data integrity guard (before ENN):** Apply deterministic preprocessing inside each training fold only: (1) missing-value imputation (median numeric, most-frequent categorical), (2) categorical encoding with train-fold-fitted encoders, (3) finite-value checks (replace inf with NaN before imputation). Abort run and log dataset ID if non-finite values remain after preprocessing.

Fit the initial ensemble E⁰ (ExtraTreesClassifier; Geurts et al., *Machine Learning* 63(1), 2006) on the cleaned data. E⁰ plays two roles throughout the loop: it provides the baseline OOB predictions (Breiman, *Machine Learning* 45(1), 2001) for Module A, and it acts as the fixed screener in Module C. Using E⁰ as a fixed screener (rather than the growing ensemble) gives Module C a stable, interpretable admission criterion across all iterations — E⁰ never trained on any synthetic point, so its OOB scores for real minority points are always clean.

**ExtraTrees bootstrap requirement:** `sklearn.ensemble.ExtraTreesClassifier` defaults to `bootstrap=False`, which produces no OOB estimates. Set `bootstrap=True, oob_score=True` explicitly — this is a hard requirement for the entire OOB infrastructure. With bootstrapping enabled, ExtraTrees retain their randomised-threshold property (the source of their speed and diversity advantage over RF; Geurts et al., 2006, §2.2) while providing OOB estimates. **Calibration validation (Phase 5 — Module A/C/G pilots):** On 3 pilot datasets, compare OOB Brier scores between bootstrapped ExtraTrees and an equivalent RandomForestClassifier. If bootstrapped ExtraTrees OOB Brier score is > 0.05 higher than RF on any pilot, switch to `RandomForestClassifier` and document the change.

---

## Module A — Epistemic admissibility

For each training point, collect the OOB probability estimates from the trees in the current ensemble that did not see it (Breiman, 2001). Decompose using the mutual-information identity (Smith & Gal, UAI 2018, arXiv:1803.08533; see also Depeweg et al., ICML 2018 for Bayesian NN formulation; Gawlikowski et al., *Artificial Intelligence Review* 56, 2023, arXiv:2107.03342 for survey):

```
Total entropy   H(p̄)
Aleatoric       (1/T) Σ_t H(p_t)      — irreducible noise; high when label is
                                         genuinely ambiguous regardless of T
Epistemic       H(p̄) − (1/T) Σ_t H(p_t)  — learnable uncertainty; high when
                                             trees disagree for fixable reasons
```

This decomposition follows from the mutual information I[y; θ | x] = H[y|x] − E_θ[H[y|x, θ]], which equals total predictive entropy minus expected per-model entropy (Houlsby et al., arXiv:1112.5745, 2011 — BALD; Kendall & Gal, NeurIPS 2017, arXiv:1703.04977). In Bayesian neural networks, diversity among sampled θ approximates posterior uncertainty. **That interpretation does not carry over:** ExtraTrees diversity is driven by bootstrap resampling and random split thresholds — algorithmic randomness with no probabilistic equivalence to posterior sampling. The mutual-information *algebra* still yields two nonnegative terms that partition total entropy, but φ_learn here measures **inter-tree disagreement attributable to ensemble construction**, not epistemic uncertainty in a inferential-statistics sense; φ_noise measures **average within-tree predictive entropy**. **Paper framing:** describe this explicitly as an "OOB-based partitioning of predictive entropy inspired by Kendall & Gal (2017), used as a synthesis-targeting heuristic — not as Bayesian epistemic/aleatoric decomposition." Empirical validation is mandatory: [07-ablation.md](07-ablation.md) **Baseline-A-simple** (ϕ_total-only mask) and the **synthetic sanity check** (supplementary): 2D ground-truth regions where learnable vs. irreducible difficulty are known — report whether φ_learn/φ_noise ordering matches design intent.

**Empirical discriminator ablation (required).** Reviewers may ask whether ϕ_learn/φ_noise adds information beyond simpler OOB summaries (e.g. margin **p̄_j(1 − p̄_j)** or total entropy **H(p̄)** only). Baseline-A-simple isolates that question; if it matches Baseline-A on AUPRC, claims must retreat to total-entropy or margin-based targeting without decomposition narrative.

A graph node is admissible if its epistemic component φ_learn is above the q_e-th percentile and its aleatoric component φ_noise is below the q_a-th percentile, both computed over minority training points only. Call this set A.

**Hyperparameters:** (q_e, q_a) tuned via Optuna per-stratum; search space: q_e ∈ {0.3, 0.4, 0.5, 0.6, 0.7}, q_a ∈ {0.2, 0.3, 0.4, 0.5} (ensures 15–70% of minority nodes are candidate admissible).

**Constraints:** Enforce |A| ≥ 2k (else reject (q_e, q_a) proposal); log all rejected proposals per dataset for diagnostic debugging.

### Synthetic ground-truth check (~half-day) — **go/no-go before Phase 5**

**Schedule:** Complete **before Phase 5** implementation (target: **end of Phase 2** alongside venue/osf framing — see [11-roadmap.md](11-roadmap.md)). This is **not** optional polish: it is the only direct test that φ_learn/φ_noise produce **spatially distinct** structure vs. trivial monotone transforms of disagreement.

Construct a **2D synthetic** binary dataset with known geometry (e.g. two Gaussian minority clusters — one near the decision boundary with fixable ensemble disagreement; one in **high overlap** with irreducible noise). Colour-map **φ_learn** and **φ_noise** on a grid. **Expected qualitative pattern:** elevated φ_learn in the first cluster; elevated φ_noise in the second.

**Go/no-go:** If spatial discrimination **fails** (maps nearly collinear with H(p̄) or arbitrary), **or** **Baseline-A-simple** later matches **Baseline-A** on pilots — **pre-specified paper framing:** retreat to **“total-OOB-entropy (or margin) admissibility mask”** without epistemic/aleatoric **interpretation**; Module A remains an ensemble-derived mask, but **terminology** drops “epistemic/aleatoric decomposition” in title/abstract. OSF registers which branch is active **before** Phase 5.

Include as supplementary Figure 1.

---

## Module B — Risk-targeted transport

### Scoring each admissible node

**Intuition (read first):** The score rewards placing a synthetic minority point where it would most increase F1 if treated as a true positive at fixed global P/R: highest when the ensemble is **most uncertain** about minority class at j (**p̄_j ≈ 0.5**) **and** current F1 is low — i.e. where an extra true positive moves F1 the most. Formal derivation follows.

F1 = 2PR/(P+R). The partial derivative with respect to placing one true-positive synthetic point at node j, treating P̂ and R̂ as locally constant:

```
s_j = ∂F̂1_OOB / ∂n_j  ≈  2 · p̄_j · (1 − p̄_j) / (P̂ + R̂)²
```

This is highest when p̄_j ≈ 0.5 (near the decision boundary) and when current F1 is low. The Module A admissibility restriction already excludes high-overlap regions (high φ_noise nodes), so a reviewer concern that "p̄_j = 0.5 might be in a majority-dense region" is handled: those nodes have high aleatoric uncertainty and are excluded from A before scoring.

The locally-constant P̂, R̂ approximation is a heuristic: it treats global precision and recall as fixed while varying n_j, which is only valid when the synthesis budget is small relative to the total dataset. We treat s_j as a *ranking function* rather than a true gradient estimate. Its rank agreement with actual OOB gain is validated empirically before any experiment proceeds.

**Validate before committing.** On 10 pilot datasets (2 per IR stratum), rank admissible nodes by s_j and compare to true leave-one-out OOB F1 gain. **Primary gate:** Spearman ρ > **0.75** on ≥ **8 of 10** pilots. ρ = 0.65 explains only ~42% rank-variance in squared correlation terms — too weak for a core OT claim; the stricter threshold aligns surrogate quality with stated novelty.

**Soft warning (ρ ∈ [0.65, 0.75] on a pilot or stratum):** Treat as degrading evidence for OT-specific claims. **Reporting obligation:** publish primary results **both** with OT placement (Module B as tuned) **and** with admissible-set SMOTE (uniform or k-NN within A, no transport coupling) on the same folds, and quantify Δ(AUPRC) attributable to OT in that stratum. Optionally switch that stratum to `s_j^simple = p̄_j · (1 − p̄_j)` after investigating P̂/R̂ variance as below; re-validate until ρ^simple > 0.75 or accept SMOTE-within-A fallback.

**Hard stop (ρ < 0.50 on any pilot, OR ρ < 0.65 on > 2 of 10 pilots):** Do not proceed with risk-targeted OT as a headline mechanism. Fall back to **synthesis restricted to A via SMOTE** (Chawla et al., *JAIR* 16, 2002; k-NN pairs within A only). See **Fallback method statement** below. Log strata and pre-register the active configuration.

---

### Fallback method statement (publish verbatim in paper when any gate triggers)

**Floor (read first):** The **minimum publishable contribution** common to **all** fallback stacks is the **OOB-based admissibility mask (Module A)** — a validated targeting mechanism vs. geometric proxies such as Borderline-SMOTE alone. OT placement, screening, and inertia undersampling are **additive** when their gates pass.

In configurations where pilot validation is not met, ET-COME **degrades gracefully**: if the ν★ surrogate fails validation (ρ below thresholds above), Module B reduces to **uniform or SMOTE synthesis within the admissible set A** — denoted **ET-COME-A⊕SMOTE** (OT inactive). If Module C acceptance rate drops below 20% at extreme IR, Module C may be disabled per stratum. **All published tables and the abstract must state which modules are active per IR stratum** — do not imply a single unified pipeline if OT/C are off in extreme-IR rows. Title/abstract claims about **equilibrium transport** follow [08-evaluation.md](08-evaluation.md) **title governance** and Module B ρ validation; otherwise use **OOB-targeted synthesis with optional transport** (pre-register **fallback title** string at OSF).

### Setting the target and temperature

ν★ = softmax(s/τ) restricted to A, where τ (softmax temperature) controls concentration of the transport mass (Hinton et al., NIPS 2014 Workshop, arXiv:1503.02531 — temperature scaling originally for distillation; here repurposed for target sharpness). Higher τ makes ν★ more uniform; lower τ concentrates mass on top-scoring admissible nodes.

**Hyperparameter:** τ ∈ {0.5, 1.0, 2.0} tuned via Optuna per-stratum (default τ = 1.0 if preliminary validation on pilots shows insensitivity).

**Validation before tuning:** On 10 pilots (2 per IR stratum), verify that halving/doubling τ changes the transport plan by < 20% in L1 norm; if sensitive, constrain τ ∈ {0.8, 1.0, 1.2}.

### Transport

Solve entropy-regularised OT (Cuturi, NeurIPS 2013, arXiv:1306.0895; Peyré & Cuturi, *FnTML* 11(5–6), 2019, arXiv:1803.00567) from the empirical minority μ̂ to ν★:

```
min_{π ∈ Π(μ̂, ν★)}  ⟨C, π⟩ + ε · KL(π ‖ μ̂ ⊗ ν★)
```

**Cost matrix — efficient formulation.** C_ij is the L2 distance between minority source node i and admissible target node j, restricted to the k nearest neighbors of j in the HNSW graph (k = 15 default; tuned in Phase 9 Optuna grid search). All other pairs are set to ∞ and excluded from the Sinkhorn solve. This gives a sparse cost matrix with at most n_minority × k entries, making the Sinkhorn solve trivially fast. **Hyperparameter:** k ∈ {10, 15, 20} tuned via Optuna per-stratum.

**Why k-NN instead of 2-hop:** Direct k-NN is standard in computational geometry and is parameter-free once k is fixed. Early drafts used "2-hop graph distance" which introduced hidden hyperparameter (graph hop limit) without principled justification. Using explicit k-NN ensures (1) fixed cost per source (k entries), (2) fairness across datasets of varying size, (3) consistency with nearest-neighbor principles. The choice k=15 is motivated by HNSW literature defaults; tuning confirms robustness.

### Warm-starting dual potentials

Carry dual potentials u, v from iteration t−1 to t, initialized with zeros for iteration 0. When A changes: zero out potentials for nodes that left A; initialise new nodes at zero. The plan π is compared only over A^(t) ∩ A^(t-1) for the stopping criterion.

**Entropy regularization:** Use log-domain Sinkhorn (Schmitzer, *SIAM J. Sci. Comput.* 41(3), 2019, arXiv:1610.06519) with regularization ε (tuned hyperparameter) to ensure numerical stability. **Hyperparameter:** ε ∈ {0.05, 0.1, 0.2} tuned via Optuna per-stratum.

**Pre-benchmark validation (Phase 6 — Module B):** On 10 pilots (2 per IR stratum), verify: (1) Sinkhorn loss decreases monotonically, (2) no NaN or divergence for any (ε, τ, k) combination, (3) warm-starting reduces inner iterations by ≥ 30% vs. cold-start.

### Why OT instead of sampling from ν★ directly

Direct sampling from ν★ scatters synthetic points across admissible nodes with no connection to where real minority points are. OT provides a coupling: each synthetic point is anchored to a real minority source, keeping synthesis on the minority manifold.

---

## Module C — OOB consistency screening

E⁰ assigns every minority training point x_i an OOB nonconformity score: s_i = 1 − E⁰_{−i}(x_i)[minority]. Accept a synthetic candidate x̃ only if:

```
E⁰(x̃)[minority]  ≥  1 − q̂_{1−α}({s_i : y_i = minority})
```

This rejects synthetic points that E⁰ — which never trained on any synthetic data — says are inconsistent with the minority class at level α. The nonconformity score formulation follows the transductive conformal framework (Vovk et al., *Algorithmic Learning in a Random World*, Springer, 2005, Ch. 2).

**What this module does not claim.** This is OOB-consistency screening, not a formal conformal prediction set. The Jackknife+ coverage guarantee (Romano, Patterson, Candès, NeurIPS 2020) applies to test-time prediction on i.i.d. held-out data. Using OOB scores to filter training data violates both assumptions, so no finite-sample coverage guarantee is claimed. The criterion is practical: synthetic points that pass the screen look like minority points to an ensemble that has only seen the original training data. Validate empirically: coverage calibration plot (nominal α vs. actual minority coverage on test folds) on ≥ 5 datasets.

**Synthetic validity guard:** After generation and before acceptance, clip each numeric feature to the min/max range of the source minority training fold and reject candidates that violate hard categorical constraints (if present). Log rejection counts under reason `out_of_domain`.

**Acceptance rate at severe imbalance.** When n_minority < 20, the score quantile is estimated from too few points to be reliable. Study acceptance rate vs. IR on pilots. If median acceptance rate drops below 20% for IR > 50, either loosen α to 0.2 or disable Module C for that stratum. Log which stratum triggers this.

---

## Module D — Entropic position sampling (optional, decision-gated)

Sample synthetic point positions from a Gaussian kernel centred on the local k-NN neighbourhood of the transport plan's matched nodes, rather than placing points exactly at graph nodes. Breaks the on-grid constraint without a trained generator.

**Decision rule for inclusion:** Evaluated in Phase 11 ablation on 20-dataset holdout.

| Outcome | Action |
|---------|--------|
| Mean AUPRC improvement ≥ 2% on ≥ 60% of datasets | Include in default pipeline as ET-COME-full |
| Improvement 1–2% on ≥ 60% of datasets | Report both ET-COME and ET-COME+D |
| Improvement < 1% | Omit from default; note in supplementary appendix |

Do not include Module D in minimum bar evaluation unless criterion (1) is met.

---

## Module G — Adaptive inertia-based majority undersampling

Remove majority points that are informationally inert — carrying no learning signal, not anchoring the decision boundary, and spatially disconnected from synthesis targets. Applied **once before iteration 0** (never re-evaluated iteratively, preventing positive feedback collapse). This approach is motivated by the empirical finding that random undersampling often matches or outperforms sophisticated oversampling methods (Drummond & Holte, ICML 2003 Workshop; Liu et al., *IEEE Trans. SMC* 39(2), 2009 — EasyEnsemble), suggesting that much majority mass is informationally redundant. Module G makes this insight precise by scoring redundancy via the OOB uncertainty decomposition.

### Inertia scoring

For each majority point x_i, compute:

```
Inertia(x_i) = (1 − φ_learn(x_i)) · (1 − φ_noise(x_i)) · 𝟙[d_i^A > θ_d]
```

where:
- φ_learn(x_i), φ_noise(x_i) are the same OOB entropy partition as Module A (see above — heuristic, not Bayesian epistemic labels)
- d_i^A = min_{j ∈ A} ‖x_i − x_j‖₂ = distance from x_i to the nearest admissible minority node in A
- θ_d = adaptive distance threshold (see below)

**Dependence of φ_learn and φ_noise (required disclosure).** For each point, **φ_learn + φ_noise = H(p̄)** (total entropy of mean OOB probabilities). The two terms are **not independent**: for fixed H(p̄), raising one lowers the other. For majority points with **near-zero total entropy** (trees unanimously predict majority), both φ_learn and φ_noise are near zero, so **(1 − φ_learn)(1 − φ_noise) ≈ 1** regardless of how entropy is split — the multiplicative decomposition adds negligible discrimination in that regime. There, **𝟙[d_i^A > θ_d]** is the operative filter (pure-distance behaviour). This is expected: inertia scoring **degenerates to distance-based filtering** for unambiguous majority points, which is computationally cheap and consistent with design goals.

A majority point is informationally inert when trees agree on it (low disagreement term), per-tree entropy is low (clear assignment), and it is far from synthesis targets (far from A). Such points contribute near-zero Gini impurity reduction (Breiman, 2001, §3; Geurts et al., 2006, §2.1) at any split in subsequent trees.

### Why distance to A, not distance to the boundary

The boundary ∂B^(t) shifts across iterations as synthetic minority mass is added. A majority point that is "far from the boundary now" may not remain so. But A is exactly where synthesis targets (Module B sends mass to A), so d_i^A > θ_d guarantees that the boundary cannot reach x_i via synthesis-induced drift — the OT coupling from Module B never transports mass near x_i. This makes the removal decision **future-proof** across all iterations without requiring boundary re-estimation.

### Adaptive removal fraction

The removal fraction γ* is not fixed but determined by an admissibility-preservation criterion:

```
γ* = max{ γ : |A(D_train \ R_γ)| / |A(D_train)| ≥ 1 − δ_A }
```

where R_γ is the set of the top-γ fraction of majority points ranked by Inertia, and |A(·)| is the admissible set size computed on the given training data. δ_A = 0.05 (default) means we allow at most 5% shrinkage of the admissible set due to undersampling.

### Theoretical justification for the adaptive criterion

The concern is that removing majority mass indirectly changes the OOB probability estimates for nearby minority points via two pathways:

1. **Split threshold shift:** Trees trained without the removed majority points may place splits differently, altering p_{t,j} for minority point x_j → changing φ_learn(x_j) → potentially changing A.
2. **Bootstrap composition:** With fewer majority points, each bootstrap replicate contains proportionally more minority points, slightly inflating minority-class OOB probabilities.

The adaptive criterion directly measures the net effect: refit E⁰ on D_train \ R_γ, recompute A, and verify |A| is preserved. The binary search over γ terminates in O(log(1/Δγ)) refits. In practice, since the search starts from γ = γ_max and decreases, a single check at γ_max followed by one bisection step suffices (2–3 E⁰ refits, each on a smaller dataset, so cheaper than the original E⁰ fit).

### Greedy ranking as heuristic

We use greedy ranking by Inertia as a computationally tractable heuristic for identifying the removable set. No formal optimality claim is made: the independence assumption (that removing x_i and x_j have independent effects on |A|) fails when x_i and x_j share HNSW neighbours and thus jointly influence OOB entropy estimates for nearby minority points — which is common in high-density majority regions.

The adaptive γ* criterion (not the greedy ranking) is the operative safety mechanism. It directly measures the net effect of removing R_γ on |A| and enforces the δ_A bound regardless of ranking quality. If the greedy heuristic misidentifies inert points, the adaptive criterion corrects by reducing γ*. The guarantee is: *whatever set is removed satisfies |A(D \ R_{γ*})| / |A(D)| ≥ 1 − δ_A*. This holds independent of the quality of the ranking heuristic.

### Safety constraints

**Hard safety constraint.** Never remove x_i if x_i ∈ kNN_k(x_j) for any x_j with y_j = minority. This ensures no minority point loses a majority neighbor, preserving the local geometry that drives Module A's entropy decomposition exactly.

**Distance threshold θ_d.** Set adaptively as the median of {d_i^A : y_i = majority, x_i ∉ ∪_j kNN_k(x_j) for minority x_j}. This ensures only the "far half" of eligible majority points can have non-zero Inertia.

### OOB infrastructure preservation bound

For P̂ (precision) in Module B's scoring formula to remain reliable, the OOB-evaluated majority count per tree must satisfy n_OOB_maj ≥ 100. Given bootstrap OOB rate ≈ 0.37 (the fraction of out-of-bag samples per tree converges to 1 − 1/e ≈ 0.368 as n → ∞; Breiman, 2001, §3.1):

```
n_maj_after = n_maj · (1 − γ*)
n_OOB_maj_per_tree ≈ 0.37 · n_maj_after ≥ 100
⟹ γ* ≤ 1 − 270 / n_maj
```

For n_maj = 4500 (IR=10, n=5000), this gives γ* ≤ 0.94 — the OOB bound is never binding in practice. For n_maj = 270 (n=300, IR=10), γ* ≤ 0.0 — undersampling is disabled on tiny datasets. This bound is enforced as a hard cap alongside the adaptive criterion.

### Decision rule for inclusion (Phase 11 ablation)

**Single source of truth** for Module G thresholds and paper placement — [08-evaluation.md](08-evaluation.md) **Criterion 7** points here only (no duplicate numbers).

On the **20-dataset ablation holdout** (same folds): define **ΔAUPRC = AUPRC(ET-COME-ABCG) − AUPRC(ET-COME-ABC)** per dataset. Compare **wall-clock training+prediction** for ABCG vs ABC on the same machine; **Outcome 1** requires **≥ 10%** reduction vs ABC; **Outcome 2** requires **≥ 20%** reduction vs ABC.

| ID | Outcome | Default pipeline | Module G in main paper vs appendix |
|----|---------|------------------|-------------------------------------|
| **1** | **ΔAUPRC ≥ +0.01** on ≥ **60%** of holdout datasets **and** wall-clock ≤ **0.90 ×** ABC (i.e. ≥ **10%** faster than ABC) | Include **G** in default **ET-COME-ABCG** | **Full** main-text (Module G subsections as written above) |
| **2** | Mean absolute **ΔAUPRC** < **0.01** (neutral) **and** wall-clock ≤ **0.80 ×** ABC (≥ **20%** faster) | Include **G** in default (efficiency-motivated stack) | **Appendix-first**: ≤ **1** paragraph on G in main body; details in supplementary |
| **3** | **ΔAUPRC** < **−0.01** on **> 30%** of holdout datasets | **Exclude G** from default pipeline; optional supplementary negative result | **Do not** foreground Module G in main text |

**Interpretation:** Outcome **1** is the **dual-signal** row (predictive **and** efficiency). Outcome **2** is **efficiency-only** — keep Module G theory **appendix-heavy**, consistent with [08-evaluation.md](08-evaluation.md) Criterion 7.

### Hyperparameters

| Parameter | Default | Search range | Rationale |
|-----------|---------|-------------|-----------|
| γ_max | 0.50 | {0.15, 0.30, 0.50} | Upper bound for adaptive search; actual γ* ≤ γ_max |
| δ_A | 0.05 | {0.03, 0.05, 0.10} | Admissibility preservation tolerance |
| θ_d | median(eligible d_i^A) | adaptive | Ensures half of eligible points are candidates |

### E⁰ identity after Module G bisection

The bisection search refits E⁰ two to three times on progressively smaller datasets. After bisection concludes, all downstream modules (A, C, and E) use **E⁰_final**: the ExtraTrees ensemble fitted on D_train \ R_{γ*}. This is distinct from E⁰_original fitted on full D_train. This is intentional: E⁰_final's OOB predictions reflect the reduced training set that will be used for synthesis, ensuring internal consistency.

**Validation:** Log (E⁰_original OOB F1, E⁰_final OOB F1) per dataset. If E⁰_final OOB F1 degrades by > 2% absolute compared to E⁰_original, the γ* removal level is too aggressive — tighten δ_A to 0.03 and re-run bisection.

---

## Module E — Equilibrium stopping

The training loop uses **incremental tree addition**, not full refit. Each iteration adds `step_size` new trees trained on D_clean ∪ X̃^(t). OOB predictions are maintained as a running average: old trees' OOB estimates are already computed and unchanged; only the new trees' OOB estimates are added to the pool (following the OOB mechanism of Breiman, 2001, §3). This is the same mechanism as v1's `step_size` parameter and is the key reason the method does not carry a large computational overhead.

Stop when both hold:

1. `‖π^(t) − π^(t−1)‖₁ < ε_π` — measured only on A^(t) ∩ A^(t-1), normalised
2. `|W^(t) − W^(t−1)| < ε_W` — W^(t) = mean_{j: y_j = minority} std_t(p_t(x_j)), i.e. the mean across minority training points of the standard deviation of OOB predicted minority-class probabilities across the T trees that excluded each point; a measure of ensemble disagreement on the minority class (decreases as the ensemble converges)

### Convergence (no formal guarantee; empirically validated before full scale)

ET-COME has no formal convergence guarantee. Breiman (2001) Theorem 1.2 establishes that RF generalisation error converges a.s. as the number of trees increases — but only for a *fixed training distribution*. ET-COME's training distribution changes every iteration as synthetic points are added. The theorem does not apply to this setting and is not cited as support for convergence.

The heuristic intuition is: as the ensemble improves on the current training set, high-epistemic-uncertainty regions become smaller, reducing A, reducing ν★'s entropy, and causing π to stabilise. This may fail if synthetic points introduce new high-uncertainty regions (distribution shift) or if the ensemble overfits to synthetic mass (cycling). The hard max_iter = 20 backstop ensures termination regardless.

**Empirical validation:** Plot ‖π^(t) − π^(t−1)‖₁ on 10 pilot datasets (2 per IR stratum) before running full experiments. **Decision rule:** If monotone decrease on ≥ 8 of 10 pilots, proceed. If oscillation on ≥ 3 pilots, investigate whether step_size or Module G interactions are causing cycling before proceeding. The paper will state explicitly: "ET-COME terminates by the max_iter backstop or the dual convergence criterion; no convergence guarantee is claimed."

**Reporting vs. minimum-bar Criterion 11:** For every dataset, log whether termination was due to **(i)** dual criterion satisfied, **(ii)** max_iter backstop, or **(iii)** oscillation early-stop. If mean iterations ≤ 15 primarily because iterations cluster at **15–19** (backstop-driven), report as **"bounded termination"** not **"convergence"** — do not market equilibrium convergence unless (i) dominates.

### Defaults

| Parameter | Value |
|-----------|-------|
| ε_π | 0.01 |
| ε_W | 0.005 |
| step_size | 5 |
| max_iter | 20 |
