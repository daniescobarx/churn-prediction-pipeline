"""Pipeline reprodutível de features para o dataset Telco Customer Churn.

Este módulo define o ``ColumnTransformer`` usado tanto no treino quanto no
serviço de inferência, garantindo que a mesma transformação seja aplicada em
todos os ambientes (zero divergência treino/produção).

Convenções:
    * Numéricas → ``StandardScaler``
    * Categóricas → ``OneHotEncoder(handle_unknown='ignore')``
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
    """Constrói o ``ColumnTransformer`` reaproveitável para o dataset Telco.

    Returns:
        ColumnTransformer não treinado com escaladores numéricos e one-hot
        encoder para categóricas.
    """
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
    """Separa o DataFrame limpo em features (X) e target (y).

    Args:
        df: DataFrame já passado por ``src.data.loader.clean``.

    Returns:
        Tupla (X, y) com as colunas de features na ordem canônica e o target
        como inteiro 0/1.
    """
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas ausentes no DataFrame: {missing}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(int)
    logger.info("Features separadas — X=%s | y=%s", X.shape, y.shape)
    return X, y


def fit_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Treina o ``ColumnTransformer`` no conjunto de treino.

    Args:
        X: features de treino (apenas — para evitar data leakage).

    Returns:
        ColumnTransformer ajustado.
    """
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
    """Persiste o ``ColumnTransformer`` em um arquivo ``.joblib``.

    Args:
        preprocessor: pipeline já treinado.
        output_path: caminho do arquivo de saída.

    Returns:
        O ``Path`` final onde o artefato foi escrito.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output_path)
    logger.info("Preprocessor exportado para %s", output_path)
    return output_path


def train_and_export_pipeline(
    df: pd.DataFrame, output_path: Path
) -> tuple[ColumnTransformer, Path]:
    """Treina o preprocessor a partir de um DataFrame e persiste em disco.

    Útil para regenerar o artefato ``preprocessor.joblib`` a partir do dataset
    bruto (após ``load_raw`` + ``clean``).

    Args:
        df: DataFrame limpo contendo todas as ``FEATURE_COLS`` e ``TARGET_COL``.
        output_path: caminho do ``.joblib`` de saída.

    Returns:
        Tupla (preprocessor treinado, caminho final do artefato).
    """
    X, _ = prepare_features(df)
    preprocessor = fit_preprocessor(X)
    path = export_preprocessor(preprocessor, output_path)
    return preprocessor, path
