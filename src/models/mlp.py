"""MLP PyTorch e serviço de inferência.

Expõe :class:`ChurnMLP`, :class:`EarlyStopping`, :func:`predict_proba`
e :class:`ChurnInferenceService` (encapsula preprocessor + pesos).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.features.pipeline import FEATURE_COLS
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED: int = 42
torch.manual_seed(SEED)


class ChurnMLP(nn.Module):
    """MLP para classificação binária.

    Arquitetura ``input → 64 (BN+ReLU+Dropout) → 32 (BN+ReLU+Dropout) →
    16 (ReLU) → 1``. Saída em logit cru — combine com
    ``BCEWithLogitsLoss`` no treino e ``sigmoid`` na inferência.
    """

    def __init__(self, input_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EarlyStopping:
    """Early stopping que salva e restaura os pesos da melhor época.

    Sem a restauração os pesos finais são, por construção, piores que os
    da época em que ``val_loss`` foi mínima.
    """

    def __init__(self, patience: int = 10, delta: float = 1e-4) -> None:
        self.patience = patience
        self.delta = delta
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.should_stop: bool = False
        self.best_weights: dict[str, torch.Tensor] | None = None

    def step(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = {
                k: v.clone().detach() for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model: nn.Module) -> None:
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info(
                "Pesos da melhor época restaurados (val_loss=%.4f)", self.best_loss
            )


def predict_proba(model: ChurnMLP, X: np.ndarray) -> np.ndarray:
    """Aplica sigmoid sobre os logits do modelo e retorna vetor ``(N,)``."""
    model.eval()
    tensor = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs.reshape(-1)


class ChurnInferenceService:
    """Encapsula preprocessor (``.joblib``) e MLP (``.pt``) para a API."""

    def __init__(self, preprocessor: Any, model: ChurnMLP) -> None:
        self.preprocessor = preprocessor
        self.model = model
        self.model.eval()

    @classmethod
    def from_paths(
        cls,
        preprocessor_path: Path,
        model_path: Path,
        device: str = "cpu",
    ) -> "ChurnInferenceService":
        """Constrói o serviço a partir dos arquivos persistidos."""
        preprocessor_path = Path(preprocessor_path)
        model_path = Path(model_path)

        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor não encontrado em {preprocessor_path}"
            )
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado em {model_path}")

        preprocessor = joblib.load(preprocessor_path)
        logger.info("Preprocessor carregado de %s", preprocessor_path)

        checkpoint = torch.load(
            model_path, map_location=device, weights_only=True
        )
        if "state_dict" not in checkpoint or "input_dim" not in checkpoint:
            raise KeyError(
                "Checkpoint inválido — chaves esperadas: 'state_dict', 'input_dim'."
            )

        model = ChurnMLP(input_dim=int(checkpoint["input_dim"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        logger.info(
            "MLP carregado de %s (input_dim=%d)",
            model_path,
            checkpoint["input_dim"],
        )

        return cls(preprocessor=preprocessor, model=model)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Aplica o pipeline + MLP e retorna probabilidades de churn."""
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Colunas ausentes na entrada: {missing}")

        X = df[FEATURE_COLS]
        X_t = self.preprocessor.transform(X).astype(np.float32)
        return predict_proba(self.model, X_t)

    def predict(
        self, df: pd.DataFrame, threshold: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retorna ``(probabilidades, classes binárias)``."""
        probs = self.predict_proba(df)
        preds = (probs >= threshold).astype(int)
        return probs, preds
