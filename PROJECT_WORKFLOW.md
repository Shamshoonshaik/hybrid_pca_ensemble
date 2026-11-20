# Workflow

```mermaid
flowchart TD
    A[Generate or load high-dimensional data] --> B[Stratified train/test split]
    B --> C[Standardize training features]
    C --> D[PCA: retain 95% variance]
    D --> E1[Logistic Regression]
    D --> E2[Random Forest]
    D --> E3[Extra Trees]
    E1 --> F[Soft probability voting]
    E2 --> F
    E3 --> F
    F --> G[Predictions]
    G --> H[Accuracy, Precision, Recall, F1, ROC-AUC]
    H --> I[Compare baselines and save artifacts]
```

## Experiment design
- Fixed random seed: 42.
- Dataset: 1,800 observations × 500 features.
- Split: 80% train / 20% test, stratified.
- PCA: retain at least 95% of training-set variance.
- Validation: 5-fold stratified cross-validation on training data.
- Test set: held out until final evaluation.
