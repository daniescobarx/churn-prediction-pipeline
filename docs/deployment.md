# Arquitetura de Deploy — Churn Predictor

## Decisão: Inferência **real-time** via FastAPI

A operadora precisa **agir no momento certo** — quando o cliente liga para o
call-center, abre uma fatura no app ou recebe contato proativo. Predições
batch (job diário/semanal) chegariam tarde para essas janelas de retenção.

| Critério | Real-time (escolhido) | Batch (descartado) |
|---|---|---|
| **Latência da decisão** | ms (call-center, app) | horas/dias |
| **Custo computacional** | baixo (modelo pequeno em CPU) | menor por inferência, mas frequência alta anula vantagem |
| **Frescor das features** | features do request | features defasadas até o próximo job |
| **Casos de uso suportados** | call-center, app, web, in-CRM | apenas listas estáticas |
| **Complexidade de produto** | a equipe já tem FastAPI e MLflow | não atende oferta dinâmica |

Como **fallback**, mantemos um job batch noturno (Airflow ou cron) que
recalcula scores para uso em campanhas e e-mail marketing — combinação
real-time **+** batch comum em retenção.

## Stack

```
Cliente (CRM, App, Call-center)
        │  HTTPS
        ▼
   ┌──────────────┐
   │   API GW     │   (autenticação JWT, rate-limit)
   └──────┬───────┘
          ▼
   ┌──────────────┐    K8s Deployment (3+ réplicas)
   │  FastAPI     │    - Liveness   /health
   │  uvicorn     │    - Readiness  /health (model_loaded=true)
   │  (este repo) │    - HPA por CPU + p95 latency
   └──────┬───────┘
          │ inferência local (state_dict + joblib)
          ▼
   ┌──────────────┐
   │  Modelo      │   - preprocessor.joblib (sklearn)
   │  in-memory   │   - mlp.pt (PyTorch eval)
   └──────────────┘
          │
          ▼
   ┌──────────────┐
   │  Logs JSON   │ → Loki / Elastic
   │  /metrics    │ → Prometheus
   │  Predições   │ → BigQuery (auditoria + monitoramento)
   └──────────────┘
```

### Por que carregar o modelo na própria réplica?

- **Latência:** evita um *hop* de rede a cada predição (modelo pequeno cabe
  na imagem Docker — ~5 MB para o `state_dict` MLP).
- **Resiliência:** falha de um *model server* externo não derruba a API.
- **Custo:** dispensa Triton/Seldon enquanto o modelo for monolito CPU.

Quando a complexidade aumentar (ensembles, GPU, A/B testing entre versões),
migrar para **MLflow Model Serving** ou **Triton** sem alterar o contrato
HTTP da API.

## CI/CD

1. **Pré-commit:** `ruff check .` + `pytest`.
2. **Pipeline GitHub Actions:**
   - lint → testes → build da imagem → push para o registry.
   - tag de imagem = `git sha`; modelo é referenciado por `MODEL_VERSION`
     em variável de ambiente.
3. **Promoção:** após aprovação manual, ArgoCD rolla a tag em produção
   (canary 10 % → 100 %, abortando em violação de SLO p95 < 200 ms).

## Versionamento de artefatos

- **Código:** git tag (`v0.1.0`).
- **Modelo:** registry MLflow (`Telco_Churn_Etapa2/PyTorch_MLP`) com stages
  `Staging` → `Production`. O artefato concreto (`mlp.pt`,
  `preprocessor.joblib`) vive em S3/GCS, montado na imagem ou baixado no
  startup conforme `MODEL_VERSION`.
- **Dados de treino:** snapshot versionado (DVC ou Delta Lake) referenciado
  pelo run MLflow.

## Segurança

- TLS terminado no API Gateway.
- Autenticação JWT no header `Authorization: Bearer …`.
- *Schema strict* via Pydantic (`extra="forbid"`) — payloads desconhecidos
  retornam 422 sem chegar ao modelo.
- Logs **não** registram PII; o `customerID` nunca entra na API
  (é dropado em `clean()` durante o treino e o schema da API o omite).

## Plano de rollback

1. ArgoCD reverte para a tag anterior (`kubectl rollout undo`).
2. O artefato da versão anterior continua disponível no MLflow / S3.
3. Alarmes do `docs/monitoring.md` disparam automaticamente o canary stop.
