# M.Tech Project Report

## Title
**Hybrid PCA and Ensemble Learning for High-Dimensional Classification**

## Abstract
Write a 200–250 word abstract describing the high-dimensional classification problem, PCA-based dimensionality reduction, the ensemble approach, experimental protocol, key measured results and limitations. Insert actual values from `outputs/metrics.json`; do not invent results.

## 1. Introduction
Explain high-dimensional data, redundant/noisy features, dimensionality reduction and ensemble learning. State the motivation and practical importance.

## 2. Problem Statement
Develop a classification framework that reduces feature dimensionality using PCA and combines complementary classifiers using soft voting. Evaluate whether the hybrid model offers a useful balance of predictive quality and dimensionality reduction.

## 3. Objectives
- Study the mathematical basis of PCA using covariance, eigenvalues and eigenvectors.
- Build a reproducible high-dimensional classification pipeline.
- Compare original-feature classifiers, PCA + single classifier and PCA + ensemble.
- Evaluate models with multiple classification metrics and cross-validation.
- Save model artifacts and experiment results.

## 4. Literature Review
Add at least 8–12 peer-reviewed papers/books. For each, summarize the method, dataset, findings and limitation. Include foundational PCA, ensemble learning, Random Forest, Extra Trees, logistic regression and evaluation references. Verify bibliographic details through your institution's library.

## 5. Mathematical Formulation
Include standardization, covariance matrix, eigen-decomposition, explained variance, PCA projection, logistic probability, ensemble soft voting and evaluation metrics. Define every symbol.

## 6. Proposed Methodology
Describe dataset generation/source, preprocessing, train/test split, PCA threshold, classifiers, ensemble strategy, hyperparameters, validation and software versions. Include a workflow diagram.

## 7. Experimental Results
Insert:
- dataset profile;
- model comparison table;
- PCA components and explained variance;
- confusion matrix;
- ROC curves;
- five-fold cross-validation mean and standard deviation;
- training time comparison.

## 8. Discussion
Discuss whether the proposed model beats each baseline, the trade-off between dimensionality reduction and predictive quality, class-wise errors, variability and computational cost. Explain unexpected outcomes rather than hiding them.

## 9. Conclusion and Future Work
Summarize evidence from the experiment. Future work: real-world high-dimensional datasets, nested cross-validation, class imbalance strategies, feature stability, weighted/stacked ensembles and model interpretability.

## 10. Limitations
The included dataset is synthetic. Results depend on the data-generating process and random seed. PCA is linear and unsupervised; it may discard low-variance but class-discriminative information. Equal-weight soft voting is a design choice, not guaranteed to be optimal. A real application requires external validation.

## References
Use a consistent IEEE reference style. Cite original papers and dataset documentation.
