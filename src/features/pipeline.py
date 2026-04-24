import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)

BINARY_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_COLS = ["gender", "InternetService", "Contract", "PaymentMethod"]


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(lambda x: 1 if str(x).lower() == "yes" else 0)
    return df


def build_preprocessor() -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), NUMERIC_COLS),
            ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_features(df: pd.DataFrame):
    df = encode_binary(df)
    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    logger.info("Features: %s | Target: %s", X.shape, y.shape)
    return X, y