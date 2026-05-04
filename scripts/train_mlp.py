"""Script reproduzível de treino do MLP.

Executa o pipeline completo:
    1. Carrega e limpa o CSV bruto
    2. Faz split estratificado treino/val/teste
    3. Treina o ``ColumnTransformer`` apenas no treino
    4. Treina o :class:`ChurnMLP` com ``BCEWithLogitsLoss`` + ``pos_weight``
    5. Aplica ``EarlyStopping`` e restaura os melhores pesos
    6. Registra parâmetros, métricas e artefatos no MLflow
    7. Persiste ``preprocessor.joblib`` e ``mlp.pt`` em ``models/``

Uso::

    python scripts/train_mlp.py --data data/raw/telco_churn.csv --out models/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.data.loader import clean, load_raw
from src.features.pipeline import (
    export_preprocessor,
    fit_preprocessor,
    prepare_features,
)
from src.models.mlp import ChurnMLP, EarlyStopping
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED: int = 42
BATCH_SIZE: int = 64
EPOCHS: int = 100
PATIENCE: int = 10
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-5
MLFLOW_EXPERIMENT: str = "Telco_Churn_Etapa2"
DECISION_THRESHOLD: float = 0.5


class TelcoDataset(Dataset):
    """Wrapper PyTorch sobre arrays NumPy ``(N, F)`` e ``(N,)``."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def _set_global_seed(seed: int) -> None:
    """Fixa todas as seeds usadas no pipeline."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def _evaluate(model: ChurnMLP, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Calcula as 4 métricas técnicas no conjunto informado."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32))
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    preds = (probs >= DECISION_THRESHOLD).astype(int)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "f1_score": float(f1_score(y, preds)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "pr_auc": float(average_precision_score(y, probs)),
    }


def train(
    data_path: Path,
    output_dir: Path,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    patience: int = PATIENCE,
) -> None:
    """Executa o treino completo, registra no MLflow e exporta os artefatos."""
    _set_global_seed(SEED)

    df = clean(load_raw(data_path))
    X, y = prepare_features(df)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.125,
        random_state=SEED,
        stratify=y_trainval,
    )

    preprocessor = fit_preprocessor(X_train)
    X_train_t = preprocessor.transform(X_train).astype(np.float32)
    X_val_t = preprocessor.transform(X_val).astype(np.float32)
    X_test_t = preprocessor.transform(X_test).astype(np.float32)

    input_dim = X_train_t.shape[1]
    logger.info("input_dim=%d (após OneHotEncoder)", input_dim)

    train_loader = DataLoader(
        TelcoDataset(X_train_t, y_train.values.astype(np.float32)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TelcoDataset(X_val_t, y_val.values.astype(np.float32)),
        batch_size=batch_size,
        shuffle=False,
    )

    n_neg = float((y_train == 0).sum())
    n_pos = float((y_train == 1).sum())
    pos_weight_value = n_neg / max(n_pos, 1.0)
    logger.info(
        "Balanceamento — pos_weight=%.3f (neg=%d, pos=%d)",
        pos_weight_value,
        int(n_neg),
        int(n_pos),
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="PyTorch_MLP_train_script"):
        mlflow.log_params(
            {
                "architecture": "64-32-16-1",
                "activation": "ReLU",
                "loss_function": "BCEWithLogitsLoss",
                "optimizer": "Adam",
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "dropout": 0.3,
                "batch_size": batch_size,
                "epochs_max": epochs,
                "early_stopping_patience": patience,
                "pos_weight": round(pos_weight_value, 4),
                "input_dim": input_dim,
                "seed": SEED,
                "n_train": len(y_train),
                "n_val": len(y_val),
                "n_test": len(y_test),
            }
        )

        model = ChurnMLP(input_dim=input_dim)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32)
        )
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        early_stopping = EarlyStopping(patience=patience)

        last_epoch = 0
        for epoch in range(1, epochs + 1):
            last_epoch = epoch
            model.train()
            train_losses: list[float] = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            train_loss = float(np.mean(train_losses))

            model.eval()
            with torch.no_grad():
                val_losses = [
                    criterion(model(xb), yb).item() for xb, yb in val_loader
                ]
            val_loss = float(np.mean(val_losses))
            early_stopping.step(val_loss, model)

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
            )

            if epoch == 1 or epoch % 10 == 0:
                logger.info(
                    "Época %3d/%d | train_loss=%.4f | val_loss=%.4f",
                    epoch,
                    epochs,
                    train_loss,
                    val_loss,
                )

            if early_stopping.should_stop:
                logger.info("Early stopping acionado na época %d.", epoch)
                break

        early_stopping.restore_best(model)
        model.eval()

        val_metrics = _evaluate(model, X_val_t, y_val.values.astype(np.float32))
        test_metrics = _evaluate(model, X_test_t, y_test.values.astype(np.float32))
        mlflow.log_metrics(
            {
                "best_val_loss": early_stopping.best_loss,
                "epochs_run": last_epoch,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )
        logger.info(
            "Avaliação final — val: %s | test: %s", val_metrics, test_metrics
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        preprocessor_path = export_preprocessor(
            preprocessor, output_dir / "preprocessor.joblib"
        )
        model_path = output_dir / "mlp.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": int(input_dim),
            },
            model_path,
        )
        logger.info("Modelo PyTorch salvo em %s", model_path)

        mlflow.log_artifact(str(preprocessor_path), artifact_path="artifacts")
        mlflow.log_artifact(str(model_path), artifact_path="artifacts")
        mlflow.pytorch.log_model(model, name="mlp_model")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina o MLP e exporta preprocessor + pesos."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/telco_churn.csv"),
        help="Caminho do CSV bruto Telco.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models"),
        help="Diretório de saída dos artefatos.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        data_path=args.data,
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )
