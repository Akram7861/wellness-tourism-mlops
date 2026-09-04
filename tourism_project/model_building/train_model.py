from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

try:
    import mlflow
except ImportError:
    mlflow = None

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DEPLOYMENT = ROOT / "tourism_project" / "deployment"

def main():
    train_df = pd.read_csv(ARTIFACTS / "train.csv")
    test_df = pd.read_csv(ARTIFACTS / "test.csv")

    X_train = train_df.drop(columns="ProdTaken")
    y_train = train_df["ProdTaken"]
    X_test = test_df.drop(columns="ProdTaken")
    y_test = test_df["ProdTaken"]

    categorical_features = X_train.select_dtypes(include="object").columns.tolist()
    numeric_features = [
        c for c in X_train.columns if c not in categorical_features
    ]

    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ),
    ])

    model = XGBClassifier(
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=2,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.05, 0.10],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=1,
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    probabilities = best_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
    }

    DEPLOYMENT.mkdir(parents=True, exist_ok=True)

    model_path = DEPLOYMENT / "model.joblib"
    metrics_path = DEPLOYMENT / "metrics.json"
    experiment_path = DEPLOYMENT / "experiment_record.json"

    joblib.dump(best_model, model_path)

    output = {
        **metrics,
        "best_params": {
            k: str(v) for k, v in grid.best_params_.items()
        },
        "cv_best_roc_auc": float(grid.best_score_),
    }

    metrics_path.write_text(
        json.dumps(output, indent=2)
    )

    experiment_record = {
        "experiment": "visit-with-us-wellness-tourism",
        "best_params": {
            k: str(v) for k, v in grid.best_params_.items()
        },
        "cv_best_roc_auc": float(grid.best_score_),
        "test_metrics": metrics,
    }

    experiment_path.write_text(
        json.dumps(experiment_record, indent=2)
    )

    if mlflow is not None:
        mlflow.set_experiment("visit-with-us-wellness-tourism")

        with mlflow.start_run():
            mlflow.log_params({
                k: str(v)
                for k, v in grid.best_params_.items()
            })
            mlflow.log_metric(
                "cv_best_roc_auc",
                float(grid.best_score_)
            )
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(metrics_path))

    print("Best parameters:", grid.best_params_)
    print("CV ROC-AUC:", round(grid.best_score_, 4))
    print("Test metrics:", metrics)

if __name__ == "__main__":
    main()
