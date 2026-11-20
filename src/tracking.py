"""Optional MLflow tracking; training does not depend on MLflow."""
def log_with_mlflow(metrics, output_dir):
    import mlflow
    import mlflow.sklearn

    mlflow.set_experiment("Hybrid_PCA_Ensemble_Classification")
    with mlflow.start_run(run_name="PCA_Soft_Voting_Ensemble"):
        mlflow.log_param("dataset", metrics["dataset"]["source"])
        mlflow.log_param("original_features", metrics["dataset"]["features"])
        mlflow.log_param("pca_components", metrics["pca"]["components_selected"])
        mlflow.log_param("pca_variance_threshold", metrics["pca"]["retained_variance_threshold"])
        for key, value in metrics["test_metrics"].items():
            mlflow.log_metric(f"test_{key}", value)
        mlflow.log_metric("cv_roc_auc_mean", metrics["cross_validation_roc_auc"]["mean"])
        mlflow.log_artifacts(str(output_dir))
