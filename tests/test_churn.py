"""Testes automatizados — Etapa 4 do Tech Challenge.

Cobertura mínima exigida (≥ 3 testes passando):
    1. Schema dos dados validado por **pandera**
    2. Smoke test do **MLP PyTorch** (formato do tensor de saída)
    3. Integração da **API FastAPI** via ``TestClient`` em ``/predict``
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest
import torch
from fastapi.testclient import TestClient
from pandera.pandas import Column, DataFrameSchema

from src.api import main as api_main
from src.api.main import app
from src.features.pipeline import (
    build_preprocessor,
    prepare_features,
)
from src.models.mlp import ChurnInferenceService, ChurnMLP, EarlyStopping, predict_proba

SEED = 42


# ---------------------------------------------------------------------------
# Fixtures compartilhadas
# ---------------------------------------------------------------------------


def _make_synthetic_df(n: int = 64) -> pd.DataFrame:
    """Gera um DataFrame sintético cobrindo todos os valores categóricos."""
    rng = np.random.default_rng(SEED)
    return pd.DataFrame(
        {
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": rng.integers(0, 72, n),
            "PhoneService": rng.choice(["Yes", "No"], n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": rng.uniform(20, 120, n).round(2),
            "TotalCharges": rng.uniform(0, 8000, n).round(2),
            "Churn": rng.integers(0, 2, n),
        }
    )


@pytest.fixture
def sample_customer() -> dict:
    """Payload completo e válido para o endpoint /predict."""
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 845.45,
    }


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return _make_synthetic_df()


@pytest.fixture
def trained_service(synthetic_df: pd.DataFrame) -> ChurnInferenceService:
    """Constrói um :class:`ChurnInferenceService` em memória, sem disco."""
    torch.manual_seed(SEED)
    X, _ = prepare_features(synthetic_df)
    preprocessor = build_preprocessor()
    preprocessor.fit(X)
    n_features = preprocessor.transform(X).shape[1]
    model = ChurnMLP(input_dim=n_features)
    model.eval()
    return ChurnInferenceService(preprocessor=preprocessor, model=model)


@pytest.fixture
def client(trained_service: ChurnInferenceService) -> Iterator[TestClient]:
    """``TestClient`` com o serviço de inferência injetado."""
    api_main._state["service"] = trained_service
    with TestClient(app) as c:
        api_main._state["service"] = trained_service
        yield c
    api_main._state["service"] = None


# ---------------------------------------------------------------------------
# 1) Schema de dados — pandera
# ---------------------------------------------------------------------------


CHURN_INPUT_SCHEMA = DataFrameSchema(
    {
        "gender": Column(str, pa.Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(int, pa.Check.isin([0, 1])),
        "Partner": Column(str, pa.Check.isin(["Yes", "No"])),
        "Dependents": Column(str, pa.Check.isin(["Yes", "No"])),
        "tenure": Column(int, pa.Check.greater_than_or_equal_to(0)),
        "PhoneService": Column(str, pa.Check.isin(["Yes", "No"])),
        "MultipleLines": Column(str),
        "InternetService": Column(
            str, pa.Check.isin(["DSL", "Fiber optic", "No"])
        ),
        "OnlineSecurity": Column(str),
        "OnlineBackup": Column(str),
        "DeviceProtection": Column(str),
        "TechSupport": Column(str),
        "StreamingTV": Column(str),
        "StreamingMovies": Column(str),
        "Contract": Column(
            str, pa.Check.isin(["Month-to-month", "One year", "Two year"])
        ),
        "PaperlessBilling": Column(str, pa.Check.isin(["Yes", "No"])),
        "PaymentMethod": Column(str),
        "MonthlyCharges": Column(float, pa.Check.greater_than_or_equal_to(0)),
        "TotalCharges": Column(float, pa.Check.greater_than_or_equal_to(0)),
    },
    strict=False,
)


class TestSchemaPandera:
    """Validação do contrato de dados via pandera."""

    def test_schema_aceita_dados_validos(self, synthetic_df: pd.DataFrame) -> None:
        df = synthetic_df.drop(columns=["Churn"])
        validated = CHURN_INPUT_SCHEMA.validate(df)
        assert len(validated) == len(df)

    def test_schema_rejeita_gender_invalido(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        df = synthetic_df.drop(columns=["Churn"]).copy()
        df.loc[0, "gender"] = "Other"
        with pytest.raises(pa.errors.SchemaError):
            CHURN_INPUT_SCHEMA.validate(df)

    def test_schema_rejeita_monthly_charges_negativo(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        df = synthetic_df.drop(columns=["Churn"]).copy()
        df.loc[0, "MonthlyCharges"] = -1.0
        with pytest.raises(pa.errors.SchemaError):
            CHURN_INPUT_SCHEMA.validate(df)


# ---------------------------------------------------------------------------
# 2) Smoke tests — MLP PyTorch
# ---------------------------------------------------------------------------


class TestMLP:
    """Garantias mínimas sobre o forward pass e o early stopping."""

    def test_forward_pass_retorna_shape_correto(self) -> None:
        torch.manual_seed(SEED)
        model = ChurnMLP(input_dim=20)
        x = torch.randn(8, 20)
        out = model(x)
        assert out.shape == (8, 1)

    def test_predict_proba_em_intervalo_valido(self) -> None:
        torch.manual_seed(SEED)
        model = ChurnMLP(input_dim=20)
        X = np.random.RandomState(SEED).randn(5, 20).astype(np.float32)
        probs = predict_proba(model, X)
        assert probs.shape == (5,)
        assert ((probs >= 0.0) & (probs <= 1.0)).all()

    def test_early_stopping_aciona_apos_patience(self) -> None:
        torch.manual_seed(SEED)
        model = ChurnMLP(input_dim=10)
        es = EarlyStopping(patience=3)
        for _ in range(5):
            es.step(val_loss=1.0, model=model)
        assert es.should_stop

    def test_early_stopping_nao_aciona_com_melhora_continua(self) -> None:
        torch.manual_seed(SEED)
        model = ChurnMLP(input_dim=10)
        es = EarlyStopping(patience=3)
        for i in range(5):
            es.step(val_loss=1.0 - i * 0.1, model=model)
        assert not es.should_stop


# ---------------------------------------------------------------------------
# 3) Integração da API — TestClient
# ---------------------------------------------------------------------------


class TestAPI:
    """Testes de integração dos endpoints com ``TestClient``."""

    def test_health_retorna_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_predict_payload_valido(
        self, client: TestClient, sample_customer: dict
    ) -> None:
        response = client.post("/predict", json=sample_customer)
        assert response.status_code == 200, response.text
        data = response.json()
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert isinstance(data["churn_prediction"], bool)
        assert data["risk_level"] in {"low", "medium", "high"}

    def test_predict_inclui_header_de_latencia(
        self, client: TestClient, sample_customer: dict
    ) -> None:
        response = client.post("/predict", json=sample_customer)
        assert "X-Process-Time-Ms" in response.headers

    def test_predict_payload_incompleto_retorna_422(
        self, client: TestClient
    ) -> None:
        response = client.post("/predict", json={"gender": "Male"})
        assert response.status_code == 422

    def test_predict_categoria_invalida_retorna_422(
        self, client: TestClient, sample_customer: dict
    ) -> None:
        sample_customer["Contract"] = "Lifetime"
        response = client.post("/predict", json=sample_customer)
        assert response.status_code == 422
