# Module F — Debugging & Visualization

**Purpose:** Enable iterative diagnosis of where ET-COME oversamples, why the method accepted/rejected synthetic points, and how the transport plan evolved across iterations. This is **not part of the core package** — it lives in a separate `gui/` subdirectory.

**Core principle:** Opt-in "god-debugging mode" saves rich artifacts at each iteration without burdening the main ET-COME classifier. Users toggle debug mode via `ET_COME(debug_mode=True, debug_dir="./et_come_debug")`.

---

## Artifacts Saved Per Iteration

At each iteration t, if `debug_mode=True`, save to `debug_dir/iteration_{t}/`:

### Admissibility (Module A)
- `admissible_set.parquet`: (point_id, φ_learn, φ_noise, epistemic_percentile, aleatoric_percentile, is_admissible)
- `entropy_decomp_stats.json`: {total_entropy_mean, epistemic_mean, aleatoric_mean, admissible_count, admissible_pct}
- `admissible_features.parquet`: Full feature vectors for all admissible points

### Transport Plan (Module B)
- `transport_plan.npy`: Sparse OT coupling matrix π^(t) (CSR format)
- `scoring_and_target.parquet`: (point_id, s_j_score, softmax_target_prob, is_selected_for_synthesis)
- `sinkhorn_convergence.json`: {iterations, final_loss, monotone_decrease, diverged}
- `cost_matrix_stats.json`: {mean_cost, max_cost, sparsity, nnz}

### Synthetic Candidates (Modules B + C)
- `synthetic_candidates_pre_screening.parquet`: All candidates before Module C screening
- `synthetic_accepted.parquet`: Points accepted (features + reason + OT coupling)
- `synthetic_rejected.parquet`: Points rejected (features + rejection reason)
- `screening_stats.json`: {candidates_generated, accepted, rejected, acceptance_rate, reasons}

### Ensemble & Convergence (Module E)
- `ensemble_oob_matrix.npy`: OOB probability matrix shape (n_train, 2 or C)
- `convergence_metrics.json`: {pi_diff_norm, width_diff, thresholds, converged}
- `minority_distribution_before_after.json`

### Module G Artifacts (saved once, before iteration 0)
- `module_g_inertia_scores.parquet`: (point_id, phi_learn, phi_noise, d_to_A, in_minority_knn, inertia_score, removed)
- `module_g_search.json`: {gamma_max, gamma_star, delta_A, A_size_before, A_size_after, n_removed, bisection_steps}
- `module_g_shrinkage_curve.json`: [{gamma, A_ratio}, ...]

### Summary
- `iteration_summary.json`: {dataset_id, iteration, n_admissible, n_synthetic_generated, n_synthetic_accepted, acceptance_rate, convergence_delta, stopped}

---

## Streamlit Interactive Dashboard

**File location:** `gui/streamlit_app.py` (standalone app, not imported by ET-COME core)

```bash
streamlit run gui/streamlit_app.py -- --debug_dir /path/to/et_come_debug --dataset_name mydata
```

### Tabs

1. **Iteration Slider:** Select iteration 0 → max_iter; dashboard updates all tabs dynamically

2. **Admissibility Tab:**
   - 2D/3D scatter: admissible vs. non-admissible points colored by epistemic/aleatoric
   - Histograms: φ_learn and φ_noise distributions with percentile thresholds
   - Table: top 10 most epistemic / most aleatoric points

3. **Transport Plan Tab:**
   - Network graph: nodes = admissible points, edges = OT couplings weighted by π[i,j]
   - Sinkhorn convergence plot

4. **Synthesis & Screening Tab:**
   - Scatter: candidates (pre-screening) vs. accepted vs. rejected in feature space
   - Color by rejection reason
   - Acceptance rate gauge

5. **Ensemble Evolution Tab:**
   - Heatmap: OOB probability matrix
   - Line plot: mean OOB predicted probability for minority class vs. iteration
   - Line plot: F1-score vs. iteration

6. **Convergence & Diagnostics Tab:**
   - ‖π^(t) − π^(t−1)‖₁ vs. iteration (should be monotone decreasing)
   - Width difference vs. iteration
   - Alert box for abnormal behavior

7. **Comparison Across Iterations Tab:**
   - Side-by-side visualisation of same metric at iteration t vs. t+1

---

## Storage & Performance

- **Disk usage:** ~10–50 MB per iteration. For 20 iterations on n=5000: 200–1000 MB.
- **Lazy loading:** App loads only selected iteration on demand.
- **Debug levels:**
  - `debug_level=0`: No artifacts (production mode)
  - `debug_level=1`: Summary JSON only
  - `debug_level=2`: Full parquet/npy artifacts

---

## Integration with Core Package

```python
from et_come import EtCome
clf = EtCome(debug_mode=True, debug_dir="./debug", debug_level=2)
clf.fit(X_train, y_train)
# Artifacts saved to ./debug/iteration_0, ./debug/iteration_1, ...
```

**GUI package:** Separate `gui/` subdirectory with own `setup.py` and requirements (streamlit, pandas, plotly, scikit-learn).

```bash
pip install -e gui/
streamlit run gui/streamlit_app.py -- --debug_dir ./debug --dataset_name mydata
```

**Zero core overhead:** If `debug_mode=False` (default), no artifacts saved, no performance penalty. Debug mode adds ~5–10% overhead (serialization only).

---

## Use Cases

1. **Debugging method collapse:** Inspect why admissible set is empty or acceptance rate is 0%
2. **Understanding synthesis regions:** Visualize which regions were targeted
3. **Validating design assumptions:** Confirm ‖π^(t) − π^(t−1)‖₁ is monotone
4. **Comparing parameter settings:** Run with different (q_e, q_a), inspect debug_dirs
5. **Publication figures:** Export visualizations for paper appendix
