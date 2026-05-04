"""Pipeline reprodutível de features.

Define o ``ColumnTransformer`` compartilhado entre treino e inferência:
``StandardScaler`` nas numéricas e ``OneHotEncoder(handle_unknown='ignore')``
nas categóricas. O artefato treinado é persistido em ``preprocessor.joblib``
para garantir a mesma transformação em todos os ambientes.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED: int = 42

NUMERIC_COLS: list[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
]

CATEGORICAL_COLS: list[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURE_COLS: list[str] = NUMERIC_COLS + CATEGORICAL_COLS
TARGET_COL: str = "Churn"


def build_preprocessor() -> ColumnTransformer:
    """Retorna um ``ColumnTransformer`` não treinado."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",
    )


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa o DataFrame limpo em ``(X, y)`` com colunas em ordem canônica."""
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas ausentes no DataFrame: {missing}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(int)
    logger.info("Features separadas — X=%s | y=%s", X.shape, y.shape)
    return X, y


def fit_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Ajusta o ``ColumnTransformer`` apenas no conjunto de treino."""
    preprocessor = build_preprocessor()
    preprocessor.fit(X)
    n_features = preprocessor.transform(X.head(1)).shape[1]
    logger.info(
        "Preprocessor treinado em %d amostras (%d features apos OHE).",
        len(X),
        n_features,
    )
    return preprocessor


def export_preprocessor(
    preprocessor: ColumnTransformer, output_path: Path
) -> Path:
    """Persiste o pipeline treinado em ``.joblib`` e retorna o caminho final."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output_path)
    logger.info("Preprocessor exportado para %s", output_path)
    return output_path


def train_and_export_pipeline(
    df: pd.DataFrame, output_path: Path
) -> tuple[ColumnTransformer, Path]:
    """Helper end-to-end: ``prepare_features`` → ``fit`` → ``export``."""
    X, _ = prepare_features(df)
    preprocessor = fit_preprocessor(X)
    path = export_preprocessor(preprocessor, output_path)
    return preprocessor, path
