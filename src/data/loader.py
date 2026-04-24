from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema

from src.utils.logger import get_logger

logger = get_logger(__name__)

CHURN_SCHEMA = DataFrameSchema(
    {
        "customerID": Column(str),
        "gender": Column(str),
        "SeniorCitizen": Column(int),
        "tenure": Column(int),
        "MonthlyCharges": Column(float),
        "TotalCharges": Column(str),
        "Churn": Column(str),
    },
    coerce=True,
)


def load_raw(path: Path) -> pd.DataFrame:
    logger.info("Carregando dataset de %s", path)
    df = pd.read_csv(path)
    logger.info("Shape bruto: %s", df.shape)
    CHURN_SCHEMA.validate(df)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["customerID"])
    logger.info("Limpeza OK. Shape: %s | Churn rate: %.1f%%", df.shape, df["Churn"].mean() * 100)
    return df