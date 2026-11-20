from src.train import load_data, build_models


def test_dataset_shape_and_labels():
    X, y = load_data()
    assert X.shape == (1800, 500)
    assert set(y.unique()) == {0, 1}


def test_model_definitions():
    models = build_models()
    assert "Proposed: PCA + Voting Ensemble" in models
    assert len(models) == 4
