# ML Canvas — Churn Predictor (Telco Customer Churn)

Síntese do problema de ML antes de qualquer linha de código. Define
**para quem** o modelo serve, **o que** ele decide, **como** o sucesso é
medido e **quais SLOs** ele precisa atender em produção.

Inspirado no template de [Louis Dorard / OCDevel](https://www.louisdorard.com/machine-learning-canvas).

---

## 1. Proposta de valor

| Campo | Descrição |
|---|---|
| **Problema** | Operadora de telecom perde clientes em ritmo acelerado. A diretoria não consegue agir antes do cancelamento porque hoje só descobre o churn quando ele já aconteceu. |
| **Quem é impactado** | (a) Cliente final, que sai por motivos resolvíveis (preço, suporte, qualidade); (b) área comercial, que perde receita recorrente; (c) operação, que paga aquisição (CAC) sem retorno. |
| **Solução proposta** | Modelo de classificação binária que dá um *score* de risco de churn nos próximos 30 dias para cada cliente ativo, exposto via API real-time. O time de retenção prioriza ação no topo do ranking. |
| **Alternativa atual** | Listas manuais de clientes "suspeitos" feitas em Excel pelo time comercial; reativa, baseada em heurísticas, sem evidência de eficácia. |
| **Por que ML** | Padrões de churn envolvem interações não-lineares entre dezenas de variáveis (contrato, consumo, suporte). Regras manuais não cobrem o espaço; ML tabular é solução comprovada para o problema. |

---

## 2. Stakeholders

| Papel | Responsabilidade | Interesse principal |
|---|---|---|
| **Diretoria comercial** (sponsor) | Aprova orçamento e KPIs | Reduzir taxa de churn mensal e CAC efetivo |
| **Time de retenção** (usuário direto) | Recebe a lista priorizada e age | Lista curta, alta precisão no topo, integrável ao CRM |
| **Time de Data Science** (owner do modelo) | Treina, avalia, mantém | Métricas técnicas dentro do alvo, ciclo de retraining viável |
| **Time de MLOps / Plataforma** (owner da API) | Empacota, deploya e observa | API estável, SLO de latência, alertas funcionando |
| **Risk & Compliance** | Aprovação ética e regulatória | Vieses controlados, sem PII em log, auditável |
| **Time de Engenharia de Dados** | Fornece o snapshot de billing/uso | Contrato de schema estável, alertas de quebra |
| **Plantão (oncall)** | Responde a incidentes | Dashboards claros, playbook em `docs/monitoring.md` |

---

## 3. Tarefa de ML

| Campo | Valor |
|---|---|
| **Tipo** | Classificação binária supervisionada |
| **Entrada** | 19 features tabulares por cliente (4 numéricas + 15 categóricas) |
| **Saída** | Probabilidade ∈ [0, 1], classe binária e nível de risco (`low`/`medium`/`high`) |
| **Threshold padrão** | 0,5 (ajustável conforme tolerância a FP/FN) |
| **Janela de predição** | Risco de churn nos próximos 30 dias |

---

## 4. Decisões suportadas

| Decisão | Quem decide | Como o score entra |
|---|---|---|
| Acionar retenção proativa (call, e-mail, oferta) | Time de retenção | Filtra `risk_level ∈ {medium, high}` e ordena por probabilidade |
| Priorizar fila do call-center quando o cliente liga | URA + agente | Score puxado em tempo real via API ao identificar o cliente |
| Definir orçamento de campanha por segmento | Comercial | Agregado mensal por `Contract`, `InternetService` |
| Acionar revisão tarifária | Comercial + financeiro | Top 1% de score combinado com `MonthlyCharges` alto |

A decisão **não** é tomada pelo modelo: ele prioriza, o humano age. Isso
limita o impacto de falsos positivos (ação extra) e mantém auditabilidade.

---

## 5. Fontes de dados

| Fonte | Conteúdo | Atualização |
|---|---|---|
| **Dataset Telco IBM** (treino atual) | 7.043 clientes históricos com label `Churn` (Yes/No) | Snapshot estático para POC |
| Base de billing (produção) | Contrato, mensalidade, total acumulado, método de pagamento | Diária |
| Base de produto | Serviços contratados (PhoneService, InternetService, addons) | Diária |
| Base de eventos | tenure, status atual, data de cancelamento (label) | Diária |

> **Limitação atual:** modelo treinado no dataset Telco público. Em
> produção, o pipeline de re-treino consumirá os snapshots internos
> equivalentes via DVC ou Delta Lake (vide `docs/deployment.md`).

---

## 6. Features

| Tipo | Colunas | Transformação |
|---|---|---|
| Numéricas (4) | `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen` | `StandardScaler` |
| Categóricas (15) | `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` | `OneHotEncoder(handle_unknown="ignore")` |

Pipeline encapsulado em `ColumnTransformer`, treinado **somente** no
conjunto de treino e exportado em `models/preprocessor.joblib` para
garantir paridade treino/produção.

---

## 7. Métricas

### 7.1 Técnicas (offline)

| Métrica | Por que escolhida | Alvo |
|---|---|---|
| **PR-AUC** (primária) | Robusta a desbalanceamento (classe positiva = 26,5%) | ≥ 0,60 |
| **Recall sobre churn** | Erro mais caro é deixar passar quem vai sair | ≥ 0,75 |
| **ROC-AUC** | Comparação entre modelos independente do threshold | ≥ 0,80 |
| **F1-Score** | Equilíbrio precision/recall | ≥ 0,55 |

Resultados atuais (test set, 1.409 clientes): **PR-AUC=0,63 · Recall=0,80
· ROC-AUC=0,84 · F1=0,61** — todos dentro do alvo.

### 7.2 De negócio (online + offline)

| Métrica | Definição | Por que importa |
|---|---|---|
| **Custo total mensal** | `FP × R$ 50 + FN × R$ 500` | Traduz erros em impacto financeiro direto |
| **Taxa de churn evitado** | (clientes flagrados que aceitaram retenção e ficaram) / (total flagrado) | Mede eficácia *real* do modelo + ação combinados |
| **Lift sobre baseline aleatório** | Conversão de retenção no top-K do modelo / conversão geral | Justifica priorização vs ação cega |
| **CAC efetivo** | Custo de aquisição / (clientes adquiridos − clientes perdidos) | Espera-se queda com o modelo em uso |

### 7.3 Premissas de custo

- **FP = R$ 50:** custo médio de uma ação de retenção desnecessária (cupom + tempo de agente).
- **FN = R$ 500:** receita média perdida em 12 meses por cliente que cancelou sem intervenção.

Premissas devem ser recalibradas trimestralmente com dados reais de LTV
e custo de campanha.

---

## 8. SLOs

### 8.1 SLOs do produto (API)

| SLO | Alvo | Janela de medição | Consequência se quebrar |
|---|---|---|---|
| **Disponibilidade** | ≥ 99,9% | mensal | Page imediato; rollback automático no canary |
| **Latência p50** | ≤ 50 ms | rolling 1h | Warn — investigar no próximo dia útil |
| **Latência p95** | ≤ 200 ms | rolling 1h | Page se >300 ms por 10 min |
| **Erro 5xx** | ≤ 1% | rolling 5min | Page imediato |
| **`model_loaded=true`** | sempre | contínuo | Page imediato — API em modo degradado |

### 8.2 SLOs do modelo (qualidade)

| SLO | Alvo | Janela | Ação se quebrar |
|---|---|---|---|
| ROC-AUC móvel (com label real T+30d) | ≥ 0,80 | 4 semanas | Retraining priorizado |
| PR-AUC móvel | ≥ 0,55 | 4 semanas | Retraining priorizado |
| Recall sobre churn | ≥ 0,75 | 4 semanas | Revisar threshold + retraining |
| Drift de input (PSI) | ≤ 0,2 em features core | mensal | Investigar pipeline de dados |
| Disparate Impact por `gender` | ∈ [0,8; 1,25] | mensal | Bloqueio ético antes de campanha |

Detalhamento operacional em [`docs/monitoring.md`](monitoring.md).

---

## 9. Construção e atualização do modelo

| Aspecto | Decisão |
|---|---|
| **Modelo principal** | MLP PyTorch (`64→32→16→1`, BatchNorm, Dropout 0.3) |
| **Baselines de comparação** | DummyClassifier, Logistic Regression, Random Forest |
| **Loss** | `BCEWithLogitsLoss` com `pos_weight = n_neg/n_pos` |
| **Cadência de retraining** | Mensal por padrão; antecipado se PSI > 0,2 ou ROC-AUC < 0,75 |
| **Critério de promoção** | Ganho ≥ 1 pp em PR-AUC vs versão em produção, equidade estável |
| **Reprodutibilidade** | `SEED=42` em torch/numpy/sklearn; pipeline serializado em joblib |
| **Tracking** | MLflow (experimento `Telco_Churn_Etapa2`) |

---

## 10. Inferência

| Aspecto | Decisão |
|---|---|
| **Modo** | Real-time via FastAPI (decisão em [`docs/deployment.md`](deployment.md)) |
| **Hardware** | CPU x86 — modelo cabe em <50 MB |
| **Carregamento** | Artefatos lidos uma vez no startup da API (`lifespan`) |
| **Volume esperado** | 50–200 RPS no horário comercial |
| **Fallback** | Job batch noturno como rede de segurança para campanhas em massa |

---

## 11. Live evaluation e feedback loop

| Sinal | Como capturar | Quando avaliar |
|---|---|---|
| **Probabilidade prevista** | Cada chamada `/predict` é gravada em BigQuery (sem PII) | Drift diário |
| **Label real** | Join T+30d com base de billing (cliente ainda ativo?) | Métrica de modelo semanal |
| **Conversão de retenção** | CRM marca se a ação proposta foi aceita | Eficácia de negócio mensal |
| **Subgrupos sensíveis** | Estratificação por `gender`, `SeniorCitizen`, `Contract`, `InternetService` | Equidade mensal |

---

## 12. Riscos e mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| Distribuição muda (novo plano comercial) | Alta | Alto | Monitor PSI + retraining antecipado |
| Vies por gênero ou faixa etária | Média | Alto | DI monitorado, bloqueio antes de campanha |
| API cai e CRM sem score | Baixa | Alto | Fallback ao último score persistido (≤ 24h) |
| Dataset de treino muito pequeno (~7k) | Média | Médio | Re-treinar com snapshot interno maior assim que disponível |
| Stakeholders interpretam score como certeza | Média | Médio | Documentação clara, `risk_level` em vez de probabilidade crua na UI |

---

## Resumo executivo

> Modelo de **classificação binária** que gera score de risco de churn em
> 30 dias, exposto via **API real-time** com SLO de p95 ≤ 200 ms. Métrica
> técnica primária **PR-AUC ≥ 0,60**, métrica de negócio **custo total
> mensal**. Owners distintos para modelo (DS) e API (MLOps); revisão
> ética por Risk & Compliance antes de qualquer campanha. Retraining
> mensal padrão, antecipado por gatilhos de drift definidos em
> `docs/monitoring.md`.
