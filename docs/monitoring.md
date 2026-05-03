# Plano de Monitoramento — Churn Predictor

Modelos de churn deterioram silenciosamente: o dataset muda (mix de planos,
campanhas, sazonalidade), mas a API continua respondendo `200 OK`. Este
plano define **o que observar**, **como medir** e **quando alarmar**.

## Pilares

| Pilar | Pergunta de negócio | Onde |
|---|---|---|
| Saúde da API | "A API está de pé e dentro do SLO?" | Prometheus + Grafana |
| Drift de dados | "As features que chegam ainda parecem treino?" | Job diário + dashboard |
| Drift de predição | "A taxa de churn predito ainda faz sentido?" | Job diário + dashboard |
| Performance real | "Quem o modelo flagrou realmente cancelou?" | Job semanal (com label real T+30d) |
| Equidade | "Métricas estão estáveis por subgrupo?" | Job semanal |

## 1. Saúde da API (real-time)

| Métrica | Fonte | Alvo (SLO) | Alerta |
|---|---|---|---|
| Disponibilidade | `/health` (probe K8s) | ≥ 99,9 % / mês | 5xx > 1 % por 5 min |
| Latência p50 | header `X-Process-Time-Ms` | ≤ 50 ms | p50 > 100 ms por 10 min |
| Latência p95 | mesmo header / Prometheus histogram | ≤ 200 ms | p95 > 300 ms por 10 min |
| Throughput | requisições/s | sem teto, observado | queda > 50 % vs baseline diário |
| `model_loaded=false` | `/health` | sempre `true` | qualquer ocorrência → page imediato |

> **Implementação:** o middleware da API já emite logs estruturados com
> `latency_ms`. Coletar via *log-shipper* (Vector/Fluent-bit) e exportar
> para Prometheus via `prometheus-fastapi-instrumentator` em produção.

## 2. Drift de dados (input)

Observar se as features de entrada continuam parecidas com a distribuição
de treino. Disparar revisão / retraining quando o desvio for material.

| Métrica | Como calcular | Limite |
|---|---|---|
| **PSI** (Population Stability Index) por feature | `PSI = Σ (p_obs - p_treino) · ln(p_obs / p_treino)` em bins (numéricas) ou categorias | `0,1` warn · `0,2` page |
| **Δ taxa de categoria nova** (OHE com `handle_unknown="ignore"` mascara) | `% linhas com ≥ 1 categoria não vista` | `> 5 %` warn · `> 10 %` page |
| **Nulos / fora de range** | `% rejeitado pela validação Pydantic` | `> 1 %` warn |

Frequência: job batch **diário** sobre as predições do dia (auditadas em
BigQuery). O dataset de referência (treino) é versionado no MLflow run.

## 3. Drift de predição (output)

Mudança brusca na distribuição de scores muitas vezes precede mudança em
features.

| Métrica | Limite |
|---|---|
| Média da `churn_probability` semanal | `±20 %` vs baseline (treino) |
| Proporção `risk_level=high` semanal | `±5 pp` vs baseline |
| KS-test entre score atual e score do treino | `p < 0,01` por 2 dias seguidos → page |

## 4. Performance real (com label T+30d)

Toda predição é registrada em `BigQuery (predictions)`; após 30 dias o
churn real é joinado a partir da base de billing.

| Métrica | Alvo | Alerta |
|---|---|---|
| ROC-AUC móvel (4 semanas) | ≥ 0,80 | < 0,75 → page |
| PR-AUC móvel | ≥ 0,55 | < 0,45 → page |
| Recall (`risk_level ≥ medium`) | ≥ 0,75 | < 0,60 → page |
| Calibração (Brier score) | ≤ 0,18 | > 0,25 → warn |

## 5. Equidade / Vieses

Métricas estratificadas por subgrupos sensíveis (vide `model_card.md`):

- ROC-AUC por `gender`, `SeniorCitizen`, `Contract`, `InternetService`.
- Δ recall entre subgrupos: `> 10 pp` → revisar antes de campanha.
- Disparate Impact (`P(pred=1|g=A) / P(pred=1|g=B)`): manter em
  `[0.8, 1.25]`.

## Alertas — pirâmide de severidade

| Severidade | Exemplo | Resposta |
|---|---|---|
| **P1 (page)** | API 5xx > 1 %, ROC-AUC < 0,75, drift PSI > 0,2 em ≥ 3 features | Plantão, mitigação imediata, RCA em 24 h |
| **P2 (warn)** | Latência p95 fora do SLO por 30 min, PSI > 0,1 | Revisão no próximo dia útil |
| **P3 (info)** | Mudança lenta de mix de categorias | Observar e documentar no review semanal |

## Cadência

- **Diário:** jobs de drift de input/output + relatório no Slack.
- **Semanal:** review de performance com label real, revisão de equidade.
- **Mensal:** decisão sobre retraining (gatilhos: PSI > 0,2 em features
  *core*; ROC-AUC abaixo do alvo; mudança de oferta comercial).

## Retraining

1. Snapshot dos últimos 12 meses do banco transacional.
2. Recalcula `train_and_export_pipeline` (mesmo código, mesma seed).
3. Retreina MLP com `scripts/train_mlp.py`.
4. Compara métricas no MLflow vs versão em produção.
5. Se ganho ≥ 1 pp em PR-AUC e equidade estável → promover via canary.
