# References

## Imbalanced learning & benchmarks
- CLIMB benchmark, arXiv:2505.17451, NeurIPS 2025 — **optional cite** if OBS benchmark includes overlapping datasets; **this plan uses open-source suites only** (see [08-evaluation.md](08-evaluation.md)).
- "Do we need rebalancing strategies?", arXiv:2402.03819, 2024.
- Improving GBDT on imbalanced data, arXiv:2407.14381, *Neurocomputing*, 2025.
- ART: Adaptive Resampling-based Training, arXiv:2509.00955, 2025.
- TabPFN v2, *Nature*, 2024.
- Liu et al., MESA, NeurIPS 2020, arXiv:2010.08349.
- Liu et al., EasyEnsemble, *IEEE Trans. Systems, Man, and Cybernetics* 39(2), 2009.
- Chen et al., BalancedRandomForest, UC Berkeley TR, 2004.
- IICOE, *Tsinghua Science and Technology*, SciOpen, 2024.
- Li et al., KWSMOTE, arXiv:2504.09147, 2025.
- Kotelnikov et al., TabDDPM, ICML 2023, arXiv:2209.15421.
- Chen et al., A survey on imbalanced learning, *Artificial Intelligence Review*, 2024.
- Drummond & Holte, C4.5, Class Imbalance, and Cost Sensitivity: Why Under-Sampling Beats Over-Sampling, ICML 2003 Workshop on Learning from Imbalanced Datasets II.
- Chawla, Bowyer, Hall, Kegelmeyer, SMOTE: Synthetic Minority Over-sampling Technique, *Journal of Artificial Intelligence Research* 16:321–357, 2002.
- Han, Wang, Mao, Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning, in *Proceedings of ICIC*, LNCS 3644:878–887, 2005.
- He, Bai, Garcia, Li, ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning, in *Proceedings of IEEE WCCI / IJCNN*, pp. 1322–1328, 2008.
- Batista, Prati, Monard, A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data, *SIGKDD Explorations Newsletter* 6(1):20–29, 2004.
- Seiffert, Khoshgoftaar, Van Hulse, Napolitano, RUSBoost: A Hybrid Approach to Alleviating Class Imbalance, *IEEE Transactions on Systems, Man, and Cybernetics — Part A* 40(1):185–197, 2010.
- Barua, Islam, Yao, Murase, MWMOTE — Majority Weighted Minority Oversampling Technique for Imbalanced Data Set Learning, *IEEE Transactions on Knowledge and Data Engineering* 26(2):405–425, 2014.
- Douzas & Bacao, Self-Organizing Map Oversampling (SOMO) for imbalanced data set learning, *Expert Systems with Applications* 82:40–52, 2017.
- Nguyen, Cooper, Kamei, Borderline over-sampling for imbalanced data classification, in *Proceedings of the International Conference on Machine Learning and Cybernetics (ICMLC)*, 2011.
- Zhang, Ma, Gan, Jiang, Agam, CGMOS: Certainty Guided Minority OverSampling, in *ACM CIKM*, pp. 1623–1631, 2016. arXiv:1607.06525.
- D'souza, M., Sarawagi, S., Synthetic Tabular Data Generation for Imbalanced Classification: The Surprising Effectiveness of an Overlap Class, in *AAAI*, 2025. arXiv:2412.15657.
- Koziarski, Krawczyk, Woźniak, Radial-Based Oversampling for Noisy Imbalanced Data Classification, *Neurocomputing* 343:19–33, 2019.
- Koziarski, Woźniak, MC-RBO: Radial-Based Oversampling for Multiclass Imbalanced Data Classification, in *IJCAI*, 2019.
- (SMOGAN) Synthetic Minority Oversampling with GAN Refinement, arXiv:2504.21152, 2025.
- Yan, Tan, Xu, Cao, Ng, Min, Wu, Oversampling for Imbalanced Data via Optimal Transport, in *AAAI*, pp. 5605–5612, 2019. (Already listed in OT section — cross-reference)

## Conformal prediction & uncertainty-guided synthesis
- Conformalised Data Synthesis, arXiv:2312.08999, *Machine Learning*, 2025.
- Filtering with Confidence, arXiv:2509.21479, 2025.
- UQDIR, *Expert Systems with Applications*, 2024.
- LEO-CVAE, arXiv:2509.25334, 2025.
- Vovk, Gammerman, Shafer, *Algorithmic Learning in a Random World*, Springer, 2005.
- Romano, Patterson, Candès, Conformalized Quantile Regression / Jackknife+, NeurIPS 2020.
- Lei et al., Distribution-Free Predictive Inference for Regression, *JASA* 113(523), 2018.

## Optimal transport
- P²OT, arXiv:2401.09266, ICLR 2024.
- Yan et al., Oversampling for Imbalanced Data via OT, AAAI 2019.
- Guo et al., Learning to Re-Weight with OT, NeurIPS 2022.
- Ren et al., DGOT, *IEEE TKDE*, 2025.
- Cuturi, Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances, NeurIPS 2013, arXiv:1306.0895.
- Peyré & Cuturi, Computational Optimal Transport, *Foundations and Trends in Machine Learning* 11(5–6), 2019, arXiv:1803.00567.
- Schmitzer, Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems, *SIAM J. Scientific Computing* 41(3), 2019, arXiv:1610.06519.
- De Bortoli et al., Diffusion Schrödinger Bridge, NeurIPS 2021.

## Uncertainty quantification & ensemble methods
- Shaker & Hüllermeier, Aleatoric and Epistemic Uncertainty with Random Forests, in *DS 2020 (Discovery Science)*, LNCS, 2020. arXiv:2001.00893. [Establishes formal epistemic/aleatoric decomposition for RF; foundational framework for Module A.]
- Kendall & Gal, What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?, NeurIPS 2017, arXiv:1703.04977.
- Smith & Gal, Understanding Measures of Uncertainty for Adversarial Example Detection, UAI 2018, arXiv:1803.08533.
- Gawlikowski et al., A Survey of Uncertainty in Deep Neural Networks, *Artificial Intelligence Review* 56, 2023, arXiv:2107.03342.
- Depeweg et al., Decomposition of Uncertainty in Bayesian Deep Learning for Efficient and Risk-sensitive Learning, ICML 2018.
- Houlsby, Huszár, Ghahramani, Lengyel, Bayesian Active Learning for Classification and Preference Learning (BALD), arXiv:1112.5745, 2011.
- Breiman, Random Forests, *Machine Learning* 45(1):5–32, 2001.
- Geurts, Ernst, Wehenkel, Extremely Randomized Trees, *Machine Learning* 63(1):3–42, 2006.
- Probst, Wright, Boulesteix, Hyperparameters and Tuning Strategies for Random Forest, *WIREs Data Mining and Knowledge Discovery* 9(3), 2019.

## Distance measures
- Bhattacharyya, On a Measure of Divergence between Two Statistical Populations Defined by Their Probability Distributions, *Bulletin of the Calcutta Mathematical Society* 35:99–109, 1943.

## Approximate nearest neighbours
- Malkov & Yashunin, Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs, *IEEE TPAMI* 42(4), 2020, arXiv:1603.09320.
- van der Maaten & Hinton, Visualizing Data using t-SNE, *JMLR* 9(Nov):2579–2605, 2008.

## Software & toolboxes
- Lemaître, Nogueira, Aridas, Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning, *Journal of Machine Learning Research* 18(17):1–5, 2017.

## Hyperparameter optimisation
- Akiba, Sano, Yanase, Ohta, Koyama, Optuna: A Next-generation Hyperparameter Optimization Framework, KDD 2019, arXiv:1907.10902.
- Bergstra, Bardenet, Bengio, Kégl, Algorithms for Hyper-Parameter Optimization, NeurIPS 2011.
- Bergstra & Bengio, Random Search for Hyper-Parameter Optimization, *JMLR* 13(10):281–305, 2012.
- Bergstra, Yamins, Cox, Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures, ICML 2013, arXiv:1209.5111.
- Hutter, Hoos, Leyton-Brown, Sequential Model-Based Optimization for General Algorithm Configuration (SMAC), LION 2011.
- Bischl et al., Hyperparameter Optimization: Foundations, Algorithms, Best Practices and Open Challenges, *WIREs Data Mining and Knowledge Discovery* 13(2), 2023, arXiv:2107.05847.
- Watanabe, Python Wrapper for Simulating Multi-Fidelity Optimization on HPO Benchmarks, arXiv:2305.17595, 2023.

## Softmax temperature & calibration
- Hinton, Vinyals, Dean, Distilling the Knowledge in a Neural Network, NIPS 2014 Deep Learning Workshop, arXiv:1503.02531.
- Guo, Pleiss, Sun, Weinberger, On Calibration of Modern Neural Networks, ICML 2017, arXiv:1706.04599.

## Evaluation metrics
- Davis & Goadrich, The Relationship Between Precision-Recall and ROC Curves, in *Proceedings of ICML*, pp. 233–240, 2006.
- Kubat & Matwin, Addressing the Curse of Imbalanced Training Sets: One-Sided Selection, in *Proceedings of the 14th International Conference on Machine Learning (ICML) Workshop on Learning from Imbalanced Data Sets*, pp. 179–186, 1997.

## Statistical testing
- Benavoli, Corani, Demšar, Zaffalon, Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis, *JMLR* 18(1), 2017.
- Dietterich, Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms, *Neural Computation* 10(7):1895–1923, 1998.

## Data cleaning
- Wilson, Asymptotic Properties of Nearest Neighbor Rules Using Edited Data, *IEEE Trans. Systems, Man, and Cybernetics* 2(3):408–421, 1972.
- Tomek, Two Modifications of CNN, *IEEE Trans. Systems, Man, and Cybernetics* 6(11):769–772, 1976.
