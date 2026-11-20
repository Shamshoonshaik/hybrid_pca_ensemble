# Hybrid PCA and Ensemble Learning for High-Dimensional Classification

**M.Tech project | AIML with Applied Mathematics**

## 1. Project overview
This project studies whether Principal Component Analysis (PCA), followed by a soft-voting ensemble, can improve classification on high-dimensional data. It uses a reproducible synthetic binary dataset with 500 input features, including informative, redundant and noise features. The data are generated locally by scikit-learn, so no external download is needed.

The experiment compares:
1. Logistic Regression on standardized original features.
2. Random Forest on standardized original features.
3. Logistic Regression after PCA.
4. The proposed **PCA + soft-voting ensemble** (Logistic Regression + Random Forest + Extra Trees).

The project reports accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, ROC curve, PCA explained variance and dimensionality reduction. A stratified train/test split and cross-validation are used to reduce evaluation bias.

## 2. Mapping to AIML with Applied Maths syllabus
| Syllabus area | Project implementation |
|---|---|
| CO-1 Mathematical foundations | Vectors/matrices, covariance matrix, eigenvalues/eigenvectors, variance, probability and classification metrics |
| CO-2 Supervised learning | Logistic regression, Random Forest, Extra Trees, train/test split and cross-validation |
| CO-3 Unsupervised learning & embeddings | PCA as linear dimensionality reduction |
| CO-4 Optimization & training | Logistic regression optimization, tree learning, hyperparameter selection and reproducible training |
| CO-5 Model evaluation & selection | Precision, recall, F1, ROC-AUC, confusion matrix and cross-validation |
| CO-6 Production ML engineering | Reproducible pipeline, saved model, metrics/artifacts, optional MLflow tracking |

## 3. Research problem
High-dimensional datasets may contain redundant and irrelevant features. These can increase training cost, introduce noise and make interpretation difficult. The project investigates whether PCA can compress the feature space while retaining useful information, and whether combining different classifiers can improve predictive performance.

## 4. Proposed methodology
1. Generate a fixed-seed dataset with 1,800 samples and 500 features.
2. Split it into stratified training and test sets.
3. Standardize features using `StandardScaler` inside pipelines to avoid data leakage.
4. Fit PCA using a retained-variance threshold of 95%.
5. Train a soft-voting ensemble using Logistic Regression, Random Forest and Extra Trees.
6. Compare the proposed model with the three baselines.
7. Evaluate only on the held-out test set; also run stratified cross-validation on training data.
8. Save model, metrics, plots and a run summary.

## 5. Mathematical foundations

### Standardization
For feature \(j\):
\[
z_{ij} = \frac{x_{ij}-\mu_j}{\sigma_j}
\]

### PCA
For standardized data matrix \(X\), the sample covariance matrix is:
\[
C = \frac{1}{n-1}X^T X
\]
Find eigenpairs:
\[
Cv_k = \lambda_k v_k
\]
Sort eigenvalues in descending order. The explained-variance ratio of component \(k\) is:
\[
r_k = \frac{\lambda_k}{\sum_j \lambda_j}
\]
Choose the smallest number of components \(m\) for which:
\[
\sum_{k=1}^{m}r_k \geq 0.95
\]
The projected data are \(Z=XV_m\).

### Soft voting ensemble
If model \(q\) estimates class probability \(P_q(y=c\mid x)\), the equal-weight ensemble probability is:
\[
P(y=c\mid x)=\frac{1}{Q}\sum_{q=1}^{Q}P_q(y=c\mid x)
\]
The predicted class is the class with the largest averaged probability.

### Evaluation
\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]
\[
Precision=\frac{TP}{TP+FP},\quad Recall=\frac{TP}{TP+FN}
\]
\[
F1=2\frac{Precision\times Recall}{Precision+Recall}
\]
ROC-AUC measures ranking quality across classification thresholds. The final report should interpret all metrics, not accuracy alone.

## 6. Folder structure
```text
hybrid_pca_ensemble_project/
├── src/
│   ├── train.py
│   └── tracking.py
├── tests/
│   └── test_project.py
├── data/                 # optional user CSV
├── outputs/              # generated model, metrics and plots
├── reports/
│   └── project_report_template.md
├── requirements.txt
├── run_project.py
└── README.md
```

## 7. Installation and execution (Windows PowerShell)
```powershell
cd path\to\hybrid_pca_ensemble_project
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_project.py
```
If `py` is not recognized, use `python` instead.

Outputs are created in `outputs/`:
- `model_comparison.csv`
- `metrics.json`
- `hybrid_pca_ensemble.joblib`
- `confusion_matrix.png`
- `roc_curves.png`
- `pca_variance.png`
- `run_summary.txt`

Run tests:
```powershell
pytest -q
```

Optional MLflow tracking:
```powershell
pip install mlflow
mlflow ui
```
Then open the local tracking page shown in the terminal. The training script logs a run when MLflow is installed; otherwise it runs normally.

## 8. Using your own CSV dataset
The default run uses a reproducible synthetic dataset. To use a CSV, edit the `load_data()` function in `src/train.py`:
- load your file with `pandas.read_csv(...)`;
- set `X` to numeric feature columns;
- set `y` to the target column;
- encode categorical features and handle missing values before training;
- do not include IDs, target leakage columns, or the target itself in `X`.

For a fair study, document the dataset source, sample count, feature count, target definition and preprocessing decisions.

## 9. Research questions
- How much can PCA reduce the number of dimensions at 95% retained variance?
- Does PCA reduce training complexity while retaining predictive performance?
- Does soft voting perform better than individual classifiers?
- Which metric changes most after dimensionality reduction?
- Are results stable across stratified folds?

## 10. Important academic note
The dataset is synthetic and is intended for method development and reproducible experimentation. Do not claim that the results prove performance on a real medical, financial or industrial dataset. For a final M.Tech dissertation, repeat the protocol on a justified real-world high-dimensional dataset, cite its source, and discuss limitations.
