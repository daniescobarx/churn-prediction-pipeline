"""API FastAPI para inferência do modelo de churn.

Endpoints:
    * ``GET  /health``  — liveness check + status do modelo
    * ``POST /predict`` — recebe features de um cliente e devolve probabilidade

Características de engenharia:
    * Validação rigorosa via Pydantic (categorias permitidas + ranges)
    * Logging estruturado em todos os pontos (sem ``print``)
    * Middleware de latência: registra ``method``, ``path``, ``status`` e ``ms``
    * ``ChurnInferenceService`` carregado uma única vez no startup
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src.models.mlp import ChurnInferenceService
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODELS_DIR = Path("models")
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_PATH = MODELS_DIR / "mlp.pt"

RISK_LOW_THRESHOLD = 0.3
RISK_HIGH_THRESHOLD = 0.6
DECISION_THRESHOLD = 0.5

_state: dict[str, ChurnInferenceService | None] = {"service": None}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Carrega o :class:`ChurnInferenceService` ao iniciar a API.

    Em ausência dos artefatos, mantém o serviço em ``None`` e responde
    ``503`` em ``/predict`` — útil em ambientes de teste sem modelo.
    """
    if _state.get("service") is None:
        if PREPROCESSOR_PATH.exists() and MODEL_PATH.exists():
            try:
                _state["service"] = ChurnInferenceService.from_paths(
                    preprocessor_path=PREPROCESSOR_PATH,
                    model_path=MODEL_PATH,
                )
                logger.info("Serviço de inferência pronto.")
            except Exception:
                logger.exception("Falha ao carregar serviço de inferência.")
                _state["service"] = None
        else:
            logger.warning(
                "Artefatos não encontrados (%s, %s) — execute scripts/train_mlp.py.",
                PREPROCESSOR_PATH,
                MODEL_PATH,
            )
    yield


app = FastAPI(
    title="Churn Predictor API",
    description="API de inferência do modelo de churn (Telco Customer Churn).",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def latency_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Mede e loga a latência de cada requisição HTTP."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "method=%s path=%s status=%d latency_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


# ---------------------------------------------------------------------------
# Schemas Pydantic
# ---------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
    """Contrato de entrada do endpoint ``/predict``.

    Cada campo replica fielmente os valores possíveis do dataset Telco.
    Categorias inválidas e tipos errados disparam ``HTTP 422``.
    """

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    """Contrato de saída do endpoint ``/predict``."""

    model_config = ConfigDict(extra="forbid")

    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_prediction: bool
    risk_level: Literal["low", "medium", "high"]


class HealthResponse(BaseModel):
    """Contrato do endpoint ``/health``."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"]
    model_loaded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Retorna o status da API e se o modelo está carregado em memória."""
    loaded = _state["service"] is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Faz inferência de churn para um cliente."""
    service = _state["service"]
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute scripts/train_mlp.py primeiro.",
        )

    df = pd.DataFrame([customer.model_dump()])
    prob = float(service.predict_proba(df)[0])

    if prob < RISK_LOW_THRESHOLD:
        risk = "low"
    elif prob < RISK_HIGH_THRESHOLD:
        risk = "medium"
    else:
        risk = "high"

    prediction = prob >= DECISION_THRESHOLD
    logger.info(
        "predict | prob=%.4f | prediction=%s | risk=%s",
        prob,
        prediction,
        risk,
    )

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=prediction,
        risk_level=risk,
    )
