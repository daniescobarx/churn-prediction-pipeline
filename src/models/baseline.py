"""Baselines (Dummy, Logistic, Random Forest) com tracking MLflow.

Cada função treina o pipeline (preprocessor + classificador) sob
``StratifiedKFold(5)`` e registra parâmetros, métricas e modelo final
no experimento ativo.
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.features.pipeline import build_preprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED = 42
CV_FOLDS = 5


def evaluate_cv(pipeline: Pipeline, X, y) -> dict[str, float]:
    """Cross-validation estratificada com accuracy, F1, ROC-AUC e PR-AUC."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring={
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
            "f1": "f1",
            "accuracy": "accuracy",
        },
    )
    return {
        "roc_auc":  float(np.mean(results["test_roc_auc"])),
        "pr_auc":   float(np.mean(results["test_average_precision"])),
        "f1_score": float(np.mean(results["test_f1"])),
        "accuracy": float(np.mean(results["test_accuracy"])),
    }


def train_dummy(X, y) -> dict[str, float]:
    """Treina DummyClassifier (``most_frequent``) e registra no MLflow."""
    logger.info("Treinando DummyClassifier...")
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("clf", DummyClassifier(strategy="most_frequent", random_state=SEED)),
    ])
    with mlflow.start_run(run_name="DummyClassifier"):
        mlflow.log_params({"model": "DummyClassifier", "strategy": "most_frequent", "seed": SEED})
        metrics = evaluate_cv(pipeline, X, y)
        mlflow.log_metrics(metrics)
        pipeline.fit(X, y)
        mlflow.sklearn.log_model(pipeline, name="model")
    logger.info("Dummy AUC-ROC: %.4f", metrics["roc_auc"])
    return metrics


def train_logistic(X, y) -> dict[str, float]:
    """Treina Logistic Regression e registra no MLflow."""
    logger.info("Treinando Regressão Logística...")
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")),
    ])
    with mlflow.start_run(run_name="LogisticRegression"):
        mlflow.log_params({"model": "LogisticRegression", "C": 1.0, "class_weight": "balanced", "seed": SEED})
        metrics = evaluate_cv(pipeline, X, y)
        mlflow.log_metrics(metrics)
        pipeline.fit(X, y)
        mlflow.sklearn.log_model(pipeline, name="model")
    logger.info("LogReg AUC-ROC: %.4f", metrics["roc_auc"])
    return metrics


def train_random_forest(X, y) -> dict[str, float]:
    """Treina Random Forest e registra no MLflow."""
    logger.info("Treinando Random Forest...")
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1)),
    ])
    with mlflow.start_run(run_name="RandomForest"):
        mlflow.log_params({"model": "RandomForest", "n_estimators": 200, "class_weight": "balanced", "seed": SEED})
        metrics = evaluate_cv(pipeline, X, y)
        mlflow.log_metrics(metrics)
        pipeline.fit(X, y)
        mlflow.sklearn.log_model(pipeline, name="model")
    logger.info("RF AUC-ROC: %.4f", metrics["roc_auc"])
    return metrics
