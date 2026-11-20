"""Reproducible PCA + ensemble classification experiment."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RANDOM_STATE = 42


def load_data():
    """Create a deterministic high-dimensional binary classification dataset."""
    X, y = make_classification(
        n_samples=1800,
        n_features=500,
        n_informative=35,
        n_redundant=65,
        n_repeated=10,
        n_classes=2,
        n_clusters_per_class=2,
        class_sep=1.15,
        flip_y=0.025,
        weights=[0.52, 0.48],
        random_state=RANDOM_STATE,
    )
    columns = [f"feature_{i+1:03d}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns), pd.Series(y, name="target")


def build_models():
    """Return comparison models; all preprocessing is kept inside pipelines."""
    raw_lr = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2500, C=1.0, solver="lbfgs")),
    ])
    raw_rf = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=250, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])
    pca_lr = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("classifier", LogisticRegression(max_iter=2500, C=1.0)),
    ])
    ensemble = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("classifier", VotingClassifier(
            estimators=[
                ("logistic", LogisticRegression(max_iter=2500, C=1.0)),
                ("random_forest", RandomForestClassifier(
                    n_estimators=250, max_features="sqrt", min_samples_leaf=2,
                    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
                )),
                ("extra_trees", ExtraTreesClassifier(
                    n_estimators=250, max_features="sqrt", min_samples_leaf=2,
                    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
                )),
            ],
            voting="soft",
            weights=[1, 1, 1],
            n_jobs=-1,
        )),
    ])
    return {
        "Logistic Regression (original features)": raw_lr,
        "Random Forest (original features)": raw_rf,
        "PCA + Logistic Regression": pca_lr,
        "Proposed: PCA + Voting Ensemble": ensemble,
    }


def _metric_row(name, model, X_test, y_test, elapsed):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
        "fit_seconds": elapsed,
    }


def run_experiment():
    OUT.mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    models = build_models()
    fitted = {}
    rows = []
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        fitted[name] = model
        rows.append(_metric_row(name, model, X_test, y_test, elapsed))
        print(f"Completed: {name}")

    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    comparison.to_csv(OUT / "model_comparison.csv", index=False)

    proposed_name = "Proposed: PCA + Voting Ensemble"
    proposed = fitted[proposed_name]
    predictions = proposed.predict(X_test)
    probabilities = proposed.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    # Cross-validation is performed on training data only.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(build_models()[proposed_name], X_train, y_train,
                                cv=cv, scoring="roc_auc", n_jobs=1)

    pca = proposed.named_steps["pca"]
    metrics = {
        "dataset": {
            "source": "sklearn.datasets.make_classification (synthetic)",
            "samples": int(X.shape[0]),
            "features": int(X.shape[1]),
            "classes": int(y.nunique()),
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "random_state": RANDOM_STATE,
        },
        "proposed_model": proposed_name,
        "pca": {
            "retained_variance_threshold": 0.95,
            "components_selected": int(pca.n_components_),
            "original_features": int(X.shape[1]),
            "explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
            "dimension_reduction_percent": float(
                100 * (1 - pca.n_components_ / X.shape[1])
            ),
        },
        "test_metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        },
        "cross_validation_roc_auc": {
            "fold_scores": [float(v) for v in cv_scores],
            "mean": float(cv_scores.mean()),
            "std": float(cv_scores.std()),
        },
        "classification_report": report,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT / "classification_report.txt").write_text(
        classification_report(y_test, predictions, zero_division=0), encoding="utf-8"
    )

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax, cmap="Blues")
    ax.set_title("Proposed PCA + Voting Ensemble")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # ROC curves
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, model in fitted.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.set_title("ROC Curve Comparison")
    fig.tight_layout()
    fig.savefig(OUT / "roc_curves.png", dpi=160)
    plt.close(fig)

    # PCA explained variance
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative)
    ax.axhline(0.95, linestyle="--", label="95% threshold")
    ax.axvline(pca.n_components_, linestyle=":", label=f"{pca.n_components_} components")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA Explained Variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "pca_variance.png", dpi=160)
    plt.close(fig)

    joblib.dump(proposed, OUT / "hybrid_pca_ensemble.joblib")
    summary = [
        "HYBRID PCA AND ENSEMBLE LEARNING FOR HIGH-DIMENSIONAL CLASSIFICATION",
        f"Input features: {X.shape[1]}",
        f"PCA components selected: {pca.n_components_}",
        f"Explained variance retained: {pca.explained_variance_ratio_.sum():.4f}",
        f"Dimension reduction: {metrics['pca']['dimension_reduction_percent']:.2f}%",
        f"Test ROC-AUC: {metrics['test_metrics']['roc_auc']:.4f}",
        f"Test F1-score: {metrics['test_metrics']['f1']:.4f}",
        f"5-fold CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}",
        "",
        "Model comparison:",
        comparison.to_string(index=False),
        "",
        "Note: synthetic benchmark results are not evidence of real-world deployment performance.",
    ]
    (OUT / "run_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    try:
        from .tracking import log_with_mlflow
        log_with_mlflow(metrics, OUT)
    except ImportError:
        print("MLflow not installed; skipping optional tracking.")

    print("\nExperiment complete. Results saved in:", OUT)
    print(comparison.to_string(index=False))
    print(f"\nPCA retained {pca.n_components_} of {X.shape[1]} features.")
    return metrics


if __name__ == "__main__":
    run_experiment()
