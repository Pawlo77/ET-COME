# Ablation Design

The ablation must show each module's independent value — not just a sequential ladder where later modules can mask earlier failures.

## Execution-order naming (paper-facing)

Letters **A–G** follow historical module IDs, not pipeline order. **Recommended paper structure** — chronological execution:

| Order | Module ID | Role |
|-------|-----------|------|
| 0 (once, setup) | **G** | Inertia undersampling on raw training data — optional; produces `E⁰_final` |
| 1 | **A** | Admissibility mask from OOB entropy partition |
| 2 | **B** | Risk-targeted OT placement (or SMOTE-within-A fallback) |
| 3 | **C** | OOB consistency screening |
| 4 (loop) | **E** | Equilibrium stopping |
| optional | **D** | Entropic position sampling |
| parallel | **F** | Debug / tooling only — not part of the published algorithm |

**Diagram (required in main paper):** one figure with flow **E⁰ → OOB matrix → Module A → Module B → Module C** (shared matrix), with Module E closing the loop — rebuts the "three glued methods" objection visually.

## Configurations

| Configuration | Components | What it isolates |
|--------------|-----------|-----------------|
| **Baseline-A-simple** | **Total OOB entropy `φ_total = H(p̄)` mask only** (no φ_learn/φ_noise split) + Borderline-SMOTE from masked nodes | Whether entropy **partitioning** beats undifferentiated OOB uncertainty |
| Baseline-A | φ_learn/φ_noise mask + Borderline-SMOTE (Han et al., ICIC 2005) from admissible nodes | Value of principled admissibility (full heuristic) alone |
| Baseline-OT-random | Random admissibility (uniform) + B transport + C screening | Value of Module A's filtering vs. random selection |
| Baseline-AC | A + C screening, no transport (SMOTE from admissible nodes, then screened) | Value of screening without transport |
| Baseline-AB | A + B transport, no screening | Value of transport without screening |
| **Baseline-Sequential** | **Independent ensembles:** CGMOS-style admissibility → OTOS-style placement → SMOGAN-style screen — **no shared OOB matrix**. **Fair primary variant:** **E_A has T₀ trees** (match ET-COME’s initial E⁰, default **T₀ = 50**); **E_B** and **E_C** use the **remainder** of the comparison budget so total trees **≥ 100** (typically **50 + 25 + 25** or **50 + 30 + 30** — pre-register split). This removes the confound where a 33-tree E_A faced ET-COME’s 50-tree Module A signal. | Architectural coupling vs. sequential composition **at matched stage-A ensemble size** |
| **Baseline-Sequential-equal-total** | Same pipeline as Sequential but **33 / 33 / 34** trees (total = 100 only) — **secondary diagnostic** for ceiling effects; **not** the primary fairness test | Shows contribution of extra trees vs. coupling when budgets are strictly equal |
| ET-COME-ABC | A + B + C (shared OOB) | Core method (oversampling only) |
| ET-COME-ABCG | A + B + C + G | Core method + adaptive undersampling; **stable reporting name** |
| ET-COME-full | A + B + C + D + G (conditional) | With entropic positions + undersampling |
| Baseline-G-only | G + RF (undersampling only, no synthesis) | Isolated value of inertia undersampling |
| Baseline-ABC-no-G | A + B + C (without G) | Confirms G adds value beyond oversampling alone |

**Baseline-Sequential protocol (novelty control):** On the **same 10 pilot datasets** as ν★ validation (2 per IR stratum): (1) Train **E_A** with **n_estimators = T₀** (same as ET-COME E⁰) → admissibility mask A′ from **E_A's OOB only**; (2) Train **E_B** → OT/placement using E_B on admissible nodes; (3) Train **E_C** for screening (real data only). **Primary comparison:** Total trees may exceed 100 so that **Module A’s ensemble width matches ET-COME** — document total FLOPs/tree count in tables. **Secondary:** Run **Baseline-Sequential-equal-total** (33/33/34) to quantify how much lift comes from **extra trees** alone.

**Interpretation:** Coupling is supported if ET-COME-ABC meaningfully beats **Baseline-Sequential** (fair primary). If the gap vs. Sequential is **≤ ~2% AUPRC**, treat coupling as **weak / incidental** unless pre-registered robustness checks pass. Report **both** Sequential variants in the ablation table.

**OSF / reviewer one-liner (mandatory):** The **primary** novelty comparison uses **E_A = T₀** so admissibility sees the **same OOB ensemble width** as ET-COME’s Module A; **Baseline-Sequential-equal-total** is **secondary** (extra-tree ceiling check only). Pre-register that sentence verbatim — closes the “33 vs 50 trees” objection before review.

---

## ET-COME-full definition

ET-COME-full is conditionally defined:
- If Module D meets inclusion criterion (≥ 2% AUPRC on ≥ 60% of datasets): ET-COME-full = A+B+C+D+G
- Otherwise: ET-COME-full collapses to ET-COME-ABCG

The stable, unconditional reporting name for the default pipeline is **ET-COME-ABCG**. Result tables must always include an ET-COME-ABCG row; ET-COME-full appears only if Module D's inclusion criterion is met.

---

## Design rationale

Running Baseline-AB, Baseline-AC, Baseline-OT-random as three separate conditions (not derived from each other) is the critical design choice:
- **Baseline-A-simple vs. Baseline-A** tests whether φ_learn/φ_noise decomposition adds measurable lift over **φ_total-only** masking — required for any decomposition narrative
- **Baseline-OT-random** cleanly isolates Module A by holding the synthesis method (OT) constant while replacing epistemic filtering with random selection
- **Baseline-AB** and **Baseline-AC** then isolate B and C independently
- **Baseline-Sequential** addresses the reviewer counterfactual — **with E_A = T₀ trees** to match ET-COME’s admissibility signal strength
- **ET-COME-ABCG vs. ET-COME-ABC** isolates Module G's contribution
- **Baseline-G-only** tests whether inertia undersampling has independent value even without oversampling

All configurations run on the same fixed **20-dataset holdout**, chosen before any ablation begins.

---

## Module G ablation-specific analysis

Report for each configuration:
1. Wall-clock time
2. Effective training set size
3. Actual γ* achieved per dataset

If ET-COME-ABCG ≈ ET-COME-ABC in AUPRC but significantly faster → Module G's value is efficiency (report speedup factor).

If ET-COME-ABCG > ET-COME-ABC in AUPRC → Module G has predictive value (investigate whether reduced majority noise improves OOB signal quality).
