# Multi-class Extension

## Scope gate before Phase 13 (mandatory one-pager)

Do **not** start the multi-class codebase rewrite until this checklist is answered in writing (OSF amendment or internal memo):

1. **Binary Phase 12 results:** Do Criteria 1–7 clearly exceed minimum bars, or barely pass? Link aggregate tables.
2. **Contribution story:** Is multi-class a **required** core claim for the target venue, or a **natural extension** better suited to a follow-up paper?
3. **Schedule realism — prerequisite:** At **Phase 12 completion**, produce an explicit **binary technical-debt audit**: list every **hard-coded binary assumption** (label encoding, shape (n,2) OOB, single Sinkhorn, metrics, tests). Only then answer whether Phase 13 can deliver ≥15 datasets, per-class Sinkhorn, HP search, and integration in **5 weeks + buffer**.
4. **Decision rule:** If binary results are **borderline**, default recommendation is **binary-only main paper** + multi-class appendix or separate submission — a focused strong story beats a broad weak one.

**Target venue:** Declare **before Phase 5** ([11-roadmap.md](11-roadmap.md)) — venue determines whether multi-class is load-bearing.

## Design principle: native multi-class, not one-vs-rest decomposition

OvR decomposition is the obvious path — run ET-COME as C binary problems and merge the synthetic sets. It is wrong for this method for three concrete reasons.

**Incoherent entropy signals.** OvR decomposes the entropy independently per binary sub-problem. A training point that is aleatoric-noisy because classes B and C overlap will look epistemic-learnable to the B-vs-rest classifier (the rest class absorbs C's uncertainty). Module A's admissibility decision is then wrong: it synthesises into a region that is genuinely irreducible, not learnable. The correct entropy must be computed over the full C-class OOB probability vector — only then does the aleatoric term correctly captures irreducible confusion between all C classes.

**C separate ensembles vs. one.** OvR requires C full Random Forest models, each with its own OOB infrastructure. ET-COME uses one C-class RF. Module A, Module C, and Module E all consume the same OOB probability matrix — there is no additional training cost to handle C classes vs. 2.

**No cross-class contamination control.** OvR cannot apply the secondary filter (reject x̃ if argmax E⁰(x̃) ≠ c) because each binary sub-classifier does not know about the other classes. In the native design, the single ensemble's argmax check is free and catches synthesis collision before it corrupts the training set.

---

## Concrete architecture

One C-class ExtraTreesClassifier (or RF) is trained on the full training set. At each iteration its OOB output is a matrix of shape (n_train, C). Module A operates row-wise on this matrix. Module B runs C_minority independent Sinkhorn solves, one per minority class, using the same graph. Module C applies the threshold check and argmax check to each candidate using the same ensemble. One merged synthetic set is added per iteration. The training loop is identical to binary, parameterised by C_minority instead of 1.

---

## Which classes are treated as minority

Class c is minority if IR_c = n_majority / n_c > 1.5 (default; user-adjustable). Majority class is the largest class by count. If two classes have similar sizes with no clear majority, use max(n_c) / n_c for each class. Classes with IR_c ≤ 1.5 are treated as majority and are never synthesised.

---

## Module A — multi-class admissibility

The C-class OOB probability vector p̄_j ∈ Δ^C is the natural output of a multi-class RF (Breiman, 2001). The entropy decomposition is unchanged — Shannon entropy over C classes instead of 2 (the mutual information identity I[y; θ | x] = H[y|x] − E_θ[H[y|x,θ]] generalises straightforwardly to C classes; see Gawlikowski et al., 2023, §3.1 for the multi-class formulation):

```
H(p̄_j)         = −Σ_c p̄_j[c] log p̄_j[c]    (total entropy)
(1/T) Σ_t H(p_t_j)                             (aleatoric)
Epistemic       = H(p̄_j) − (1/T) Σ_t H(p_t_j)
```

Admissibility is computed **per minority class**. Node j is admissible for synthesis into class c (A_c) if: it belongs to class c AND φ_learn(j) > q_e-th percentile AND φ_noise(j) < q_a-th percentile, both percentiles computed over class-c training points only. A node belonging to class c is not admissible for class c' ≠ c — synthesis respects class membership.

---

## Module B — per-class independent transport

Run C_minority independent OT problems, one per minority class:

```
For each minority class c:
  μ̂_c  = empirical distribution over class-c training points
  ν★_c  = softmax(s/τ) restricted to A_c
  π_c   = Sinkhorn( μ̂_c → ν★_c, cost = k-NN L2 with same k as binary )
```

Synthesis budget for class c: bring n_c up to mean(n_majority, n_c) — halfway to the majority count. This avoids gross over-synthesis when one class is extremely large. User can override with `sampling_strategy` analogous to imblearn's convention (Lemaître et al., *JMLR* 18(17), 2017).

The C_minority transport problems are independent (no coupling across classes). This is computationally favourable but ignores class–class interactions — see [10-risks.md](10-risks.md). Warm-started dual potentials are maintained per class (u_c, v_c) and never shared across classes.

---

## Module C — per-class OOB screening

Apply screening independently for each minority class c. E⁰ assigns a C-class OOB probability vector to each training point. For class c:

```
s_i^c  = 1 − E⁰_{−i}(x_i)[c]    for all x_i with y_i = c
Accept x̃ for class c if  E⁰(x̃)[c] ≥ 1 − q̂_{1−α}({s_i^c})
```

Secondary filter: also reject x̃ if argmax E⁰(x̃) ≠ c — the ensemble's top-1 prediction for the candidate is a different class. This catches synthesis collision without requiring any additional OOB calls.

---

## Module E — multi-class stopping

Stopping is on the worst-case class:

```
max_c ‖π_c^(t) − π_c^(t−1)‖₁ < ε_π    (on A_c^(t) ∩ A_c^(t−1), normalised)
max_c |W_c^(t) − W_c^(t−1)| < ε_W
```

Using max rather than mean makes convergence conservative — the loop does not stop while any minority class is still shifting.

---

## Module G — multi-class majority undersampling

In the multi-class setting, "majority" refers to all classes with IR_c ≤ 1.5. Module G operates on the union of majority-class points. The adaptations are:

- **Distance to admissible set:** d_i^A = min_{c ∈ minority_classes} min_{j ∈ A_c} ‖x_i − x_j‖₂
- **Safety constraint:** Never remove x_i if x_i ∈ kNN_k(x_j) for any minority-class point x_j
- **Adaptive criterion (per-class preservation):** γ* = max{ γ : min_c |A_c(D \ R_γ)| / |A_c(D)| ≥ 1 − δ_A }
- **Inertia scoring unchanged:** The entropy decomposition uses the full C-class OOB probability vector

---

## Synthesis order

Synthesise all minority classes in a single pass per iteration (not sequentially). Sequential synthesis would bias the OOB estimates seen by later classes because earlier synthetic points have already been added to the training set. A single merged augmentation X̃^(t) = ∪_c X̃_c^(t) is added before the next OOB update.

---

## IR definition for multi-class grid search

For multi-class datasets the relevant IR is computed as **mean pairwise IR** = mean(n_majority / n_c) over minority classes c. This single number determines which IR stratum a dataset falls into for the grid search.

**Edge case handling:** Datasets with extreme heterogeneity (e.g., [1000, 100, 10, 1] vs. [500, 500, 1, 1], both mean IR ≈ 50 but very different geometry) should also report **stratum-specific results broken by (min_IR, max_IR) range** in supplementary appendix.

---

## Experiment infrastructure changes required

The current codebase is binary at every layer. The following changes are needed before any multi-class experiment can run.

### MultiClassDatasetManager

New class (extends or mirrors `BinaryDatasetManager`):
- Remove `len(unique_values) != 2` validation and `np.where(target == 1, 1, 0)` normalisation
- `_prepare_datasets` must compute per-class counts and mean IR
- Add the multi-class dataset registry below

### Multi-class dataset list (≥ 15 datasets, stratified by mean IR)

| Dataset | UCI ID / Source | Classes | Mean IR stratum | Verification |
|---------|--------|---------|---|---|
| Wine | UCI 109 | 3 | Low | ✓ Public |
| Thyroid (ann) | UCI 102 | 3 | Low | ✓ Public |
| Dermatology | UCI 33 | 6 | Low | ✓ Public |
| Car Evaluation | UCI 19 | 4 | Low | ✓ Public |
| Contraceptive | UCI 30 | 3 | Low | ✓ Public |
| Glass | UCI 42 | 6 | Medium | ✓ Public |
| Yeast | UCI 110 | 10 | Medium | ✓ Public |
| Page Blocks | UCI 78 | 5 | Medium | ✓ Public |
| Abalone (binned) | UCI 1 | 3 | Medium | ✓ Public |
| Ecoli | UCI 39 | 8 | High | ✓ Public |
| Shuttle | KEEL | 7 | High | ⚠ Verify availability |
| Lymphography | UCI 63 | 4 | High | ✓ Public |
| Balance Scale | UCI 12 | 3 | High | ✓ Public |
| Vowel | KEEL | 11 | Extreme | ⚠ Verify availability |
| Optdigits | KEEL | 10 | Extreme | ⚠ Verify availability |

**Contingency:** If KEEL datasets unavailable, replace with UCI equivalents (Monk problems, Iris, Segmentation). Document replacements.

**Overlap control:** KEEL datasets used for multi-class benchmark must be disjoint from tuning corpus and ablation holdout.

### MultiClassEtComeClassifier

- `minority_class` → `minority_classes: List[Any]`
- `fit()` OOB output shape (n_train, C); all downstream modules consume this
- Module B loop: `for c in minority_classes: solve Sinkhorn(μ̂_c, ν★_c)`
- Stopping: `max_c` over convergence deltas

### Scoring changes

- All metrics need `average='macro'` for multi-class
- `average_precision`: compute macro-AUPRC manually as mean over per-class AUPRC
- `geometric_mean_score` (G-mean; Kubat & Matwin, ICML Workshop 1997) from imblearn (Lemaître et al., *JMLR* 18(17), 2017) accepts `average='multiclass'`
- `roc_auc` with `multi_class='ovr'` and `average='macro'`
