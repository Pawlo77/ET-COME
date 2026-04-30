Additional Notes
================

Implementation Details
----------------------

**Class Imbalance Handling:**

The ET-COME method addresses class imbalance through three integrated modules:

1. **Module A (Epistemic Admissibility)**: Identifies learnable uncertainty regions using information-theoretic decomposition
2. **Module B (Risk-Targeted Transport)**: Uses optimal transport to move synthetic mass toward regions where the ensemble is genuinely confused
3. **Module C (Conformal Screening)**: Filters synthetic points using OOB-based conformal prediction thresholds

**Algorithm Steps:**

1. Clean training data with ENN, build HNSW graph, train initial ensemble E⁰
2. For each iteration:
   - Decompose OOB entropy into epistemic and aleatoric components (Module A)
   - Score admissible minority nodes by F1 gradient (Module B)
   - Solve entropy-regularized optimal transport problem (Module B)
   - Screen candidates against E⁰ OOB consistency threshold (Module C)
   - Add accepted synthetic points to training set
   - Increment ensemble with new trees on augmented data
3. Stop when transport plan and OOB interval width both converge

**Performance Considerations:**

- Training time scales linearly with number of iterations
- Memory usage depends on oversampling ratio and dataset size
- Recommendation: Start with 3-5 iterations, tune based on validation performance

**Hyperparameter Tuning:**

Key hyperparameters to tune:
- ``iterations``: Number of refinement iterations (default: 5)
- ``n_neighbors``: Neighborhood size for SMOTE (default: 5)
- ``sampling_strategy``: Oversampling ratio (default: 'auto')
- ``classifier``: Base classifier and its parameters

Validation and Evaluation
--------------------------

**Metrics:**

Due to class imbalance, standard accuracy is not recommended. Instead, use:

- F1-score
- Balanced Accuracy
- Precision-Recall AUC
- G-Mean

**Cross-Validation:**

Always use stratified k-fold cross-validation to maintain class distribution in folds.

.. warning::
   Do not use standard accuracy as the primary evaluation metric on imbalanced datasets.

Memory and Runtime
------------------

**Memory Usage:**
- Oversampling ratio significantly impacts memory
- Consider dataset size and available memory when setting ``sampling_strategy``
- Option to use ``verbose=False`` to reduce intermediate data storage

**Runtime Optimization:**
- Use ``n_jobs=-1`` with scikit-learn classifiers for parallel processing
- Consider smaller datasets for hyperparameter tuning before full experiments
- Pre-allocate arrays where possible in custom implementations

Limitations
-----------

1. **Binary Classification**: Currently optimized for binary classification
2. **Feature Types**: Works best with numerical features
3. **Small Datasets**: May require more careful hyperparameter tuning
4. **Categorical Features**: Recommend encoding before use

Future Work
-----------

- Multiclass classification support
- Automated hyperparameter tuning
- Production-ready optimization
- GPU acceleration for large datasets
- Support for categorical features with native encoding

Troubleshooting
---------------

**Issue: Poor performance compared to baseline**

- Check that cross-validation is stratified
- Verify that appropriate metrics are used (not accuracy)
- Try increasing number of iterations
- Ensure base classifier is appropriate for your data

**Issue: Training time very long**

- Reduce number of iterations
- Use smaller validation set
- Consider reducing dataset size for tuning
- Use parallel processing (``n_jobs=-1``)

**Issue: Memory errors**

- Reduce oversampling ratio
- Use smaller batch sizes
- Consider sampling the training data
- Monitor memory usage during iterations
