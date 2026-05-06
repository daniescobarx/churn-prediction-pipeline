# Churn Prediction Pipeline

Projeto end-to-end de previsão de churn sobre o dataset **IBM Telco Customer
Churn**. Implementa todas as etapas do Tech Challenge: EDA, baselines,
MLP em PyTorch, pipeline reprodutível em Scikit-Learn, API FastAPI,
testes automatizados e documentação de deploy/monitoramento.

## Stack

Python 3.10+ · PyTorch · Scikit-Learn · FastAPI · Pydantic v2 · MLflow ·
Pytest · Pandera · Ruff

## Estrutura do repositório

```
churn-prediction-pipeline/
├── data/raw/telco_churn.csv          # Dataset Telco original
├── docs/
│   ├── model_card.md                 # Performance, limitações, vieses
│   ├── deployment.md                 # Arquitetura real-time + CI/CD
│   └── monitoring.md                 # Drift, alertas, retraining
├── models/                           # Artefatos (preprocessor.joblib + mlp.pt)
├── notebooks/
│   ├── 01_eda_and_baselines.ipynb    # Etapa 1
│   └── 02_pytorch_mlp.ipynb          # Etapa 2
├── scripts/
│   └── train_mlp.py                  # Treina e exporta artefatos
├── src/
│   ├── api/main.py                   # FastAPI (/health, /predict)
│   ├── data/loader.py                # load_raw + clean
│   ├── features/pipeline.py          # ColumnTransformer + train_and_export
│   ├── models/
│   │   ├── baseline.py               # Dummy + LogReg + RandomForest
│   │   └── mlp.py                    # ChurnMLP + EarlyStopping + Service
│   └── utils/logger.py               # Logger estruturado
└── tests/test_churn.py               # Pandera + smoke MLP + integração API
```

## Setup

```bash
make install        # cria ambiente local com dependências dev
```

> Requer Python 3.10+ e `pip` atualizado. Em Windows o Makefile pode ser
> executado com `mingw32-make` ou via WSL; alternativamente, rode os
> comandos `pip install -e .[dev]` manualmente.

## Comandos do Makefile

| Comando | O que faz |
|---|---|
| `make install`  | Instala dependências do projeto + dev (`pip install -e .[dev]`) |
| `make lint`     | `ruff check .` (não modifica arquivos) |
| `make lint-fix` | Aplica correções automáticas do ruff |
| `make test`     | Roda `pytest -v` |
| `make train`    | Treina o MLP e gera `models/preprocessor.joblib` + `models/mlp.pt` |
| `make run-api`  | Sobe a API FastAPI com hot-reload em `http://127.0.0.1:8000` |
| `make mlflow`   | Abre a UI do MLflow para inspecionar runs |

## Fluxo recomendado

1. **Treine os artefatos** (gera `models/`):
   ```bash
   make train
   ```
2. **Suba a API**:
   ```bash
   make run-api
   ```
3. **Teste o endpoint** em outro terminal:
   ```bash
   curl -X POST http://127.0.0.1:8000/predict \
        -H "Content-Type: application/json" \
        -d '{
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
              "TotalCharges": 845.45
            }'
   ```
   Resposta esperada:
   ```json
   {
     "churn_probability": 0.7321,
     "churn_prediction": true,
     "risk_level": "high"
   }
   ```

## Endpoints da API

| Método | Rota             | Descrição |
|---|---|---|
| `GET`  | `/health`        | Liveness + `model_loaded` |
| `POST` | `/predict`       | Inferência para um cliente (schema rigoroso) |
| `GET`  | `/docs`          | Swagger UI (auto-gerado) |
| `GET`  | `/openapi.json`  | Schema OpenAPI |

Todas as respostas trazem o header `X-Process-Time-Ms` com a latência da
requisição (também emitida via logger estruturado).

## Reprodutibilidade

- `SEED = 42` em todos os módulos (`torch`, `numpy`, `train_test_split`).
- Pipeline de pré-processamento exportado em joblib — mesma transformação
  no treino e na inferência.
- `scripts/train_mlp.py` registra **parâmetros, métricas (val/test) e
  artefatos** (`mlp.pt`, `preprocessor.joblib`, modelo PyTorch) no MLflow
  no experimento `Telco_Churn_Etapa2`. Inspecione os runs com `make mlflow`.

## Testes

`make test` executa três blocos (12 testes ao todo):

1. **Schema (pandera):** valida o contrato de entrada da API.
2. **Smoke MLP:** verifica shape do tensor de saída e early stopping.
3. **Integração API:** `TestClient` valida `/health`, `/predict` (200/422)
   e o header de latência.

## Documentação adicional

- [`docs/ml_canvas.md`](docs/ml_canvas.md) — proposta de valor, stakeholders, métricas de negócio e SLOs.
- [`docs/model_card.md`](docs/model_card.md) — performance, limitações e vieses.
- [`docs/deployment.md`](docs/deployment.md) — arquitetura de deploy.
- [`docs/monitoring.md`](docs/monitoring.md) — drift, alertas e retraining.

## Entregáveis do Tech Challenge

Para facilitar a navegação e avaliação do projeto, os requisitos foram divididos conforme a tabela abaixo:

| Fases do Projeto | Descrição da Entrega | Onde Encontrar |
|---|---|---|
| **1. Exploração e Baselines** | Análise exploratória de dados (EDA), limpeza e criação de modelos base de machine learning. | `notebooks/01_eda_and_baselines.ipynb`, `src/models/baseline.py` |
| **2. Modelagem Avançada** | Implementação de arquitetura Multilayer Perceptron (MLP) utilizando PyTorch. | `notebooks/02_pytorch_mlp.ipynb`, `src/models/mlp.py` |
| **3. Engenharia e Deploy** | Pipeline de dados reprodutível, API RESTful com FastAPI e suíte de testes automatizados. | `src/features/pipeline.py`, `src/api/main.py`, `tests/test_churn.py` |
| **4. Governança e MLOps** | Documentação de deploy, análise de vieses, model card e estratégias de monitoramento. | Diretório `docs/` |