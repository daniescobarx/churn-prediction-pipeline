# Model Card — Churn Predictor (Telco Customer Churn)

## Visão geral

| Campo | Valor |
|---|---|
| Nome | Churn Predictor MLP |
| Versão | 0.1.0 |
| Tipo | Classificação binária (Churn = 1 / Não-churn = 0) |
| Algoritmo | MLP em PyTorch (`64 → 32 → 16 → 1`, ReLU + BatchNorm + Dropout 0.3) |
| Loss | `BCEWithLogitsLoss` com `pos_weight = n_neg / n_pos` (≈2.77) |
| Otimizador | Adam (`lr=1e-3`, `weight_decay=1e-5`) |
| Regularização | Dropout 0.3, Early Stopping (`patience=10`) e BatchNorm |
| Seeds | `SEED = 42` em PyTorch e NumPy |
| Tracking | MLflow (experimento `Telco_Churn_Etapa2`) |

## Dados de treino

- **Fonte:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Tamanho:** 7.043 clientes, 19 features após pipeline de preparação
- **Distribuição do alvo:** ~26,5 % de churn (classe positiva)
- **Splits estratificados:** 70 % treino · 10 % validação · 20 % teste (`random_state=42`)

## Pipeline de features

| Tipo | Colunas | Transformação |
|---|---|---|
| Numéricas | `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen` | `StandardScaler` |
| Categóricas | `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` | `OneHotEncoder(handle_unknown="ignore")` |

O `ColumnTransformer` é treinado **apenas no conjunto de treino** e persistido em
`models/preprocessor.joblib`, evitando data leakage e divergência treino/produção.

## Performance no conjunto de teste (1.409 clientes)

| Métrica | MLP PyTorch | Random Forest (baseline) |
|---|---|---|
| Accuracy | 0,7282 | 0,7601 |
| F1-Score | 0,6120 | 0,4955 |
| ROC-AUC | 0,8398 | 0,8011 |
| PR-AUC | 0,6270 | 0,5770 |
| Recall (churn) | 80,7 % | 44,1 % |

> O MLP entrega **+36,6 pp de recall sobre churn** em troca de mais falsos positivos —
> alinhado ao objetivo de negócio (custo de FN >> custo de FP).

### Análise de custo (R$)

Premissas: `Custo FP = R$ 50` (ação de retenção desnecessária),
`Custo FN = R$ 500` (perda de receita).

| Modelo | TP | TN | FP | FN | Custo total |
|---|---|---|---|---|---|
| **MLP PyTorch** | 302 | 724 | 311 | 72 | **R$ 51.550** |
| Random Forest | 165 | 907 | 128 | 209 | R$ 110.900 |

Economia potencial do MLP vs RF ≈ **R$ 59 mil** no conjunto de teste.

## Uso pretendido

- **Caso de uso primário:** scoring online de clientes em risco para campanhas
  de retenção (call-center, e-mail, descontos targetados).
- **Decisão suportada:** priorizar carteira por probabilidade de churn ou por
  *risk_level* (`low` < 0.3, `medium` < 0.6, `high` ≥ 0.6).
- **Threshold padrão:** `0,5`. Ajuste-o conforme a tolerância da operadora ao
  custo de FP/FN (ver dashboard de monitoramento).

## Limitações conhecidas

1. **Desbalanceamento residual.** Mesmo com `pos_weight`, o modelo tende a
   superestimar churn em clientes com `Contract = Month-to-month`,
   `InternetService = Fiber optic` e `tenure` baixo.
2. **Estabilidade do `TotalCharges`.** O dataset bruto contém strings vazias
   nessa coluna (clientes recém-contratados); são preenchidas com `0.0` em
   `clean()`. O sinal pode estar artificialmente reduzido.
3. **Não captura sazonalidade.** Não há feature temporal (mês, trimestre,
   datas de promoções). Mudanças no mix de campanhas exigem retraining.
4. **Conjunto pequeno (~7 k linhas).** Métricas têm intervalo de confiança
   relativamente largo — replicar em fold maior antes de produção.

## Vieses potenciais

- **Gênero (`gender`).** Coluna com pouca variação informativa; o modelo
  pode aprender pseudo-correlações via interação com outras variáveis.
  Recomenda-se monitorar Disparate Impact por gênero antes de qualquer
  ação de marketing dirigida.
- **`SeniorCitizen`.** Subgrupo minoritário (~16 %); recall pode degradar
  rapidamente. Reportar métricas estratificadas por essa coluna.
- **Renda implícita.** `MonthlyCharges` e `PaymentMethod` são proxies de
  poder aquisitivo e podem reforçar exclusão de clientes de baixa renda.
- **Geografia.** O dataset não inclui dimensão geográfica; aplicar o modelo
  em regiões com perfil distinto pode degradar performance silenciosamente.

## Métricas para revisão contínua

- ROC-AUC e PR-AUC mensais por subgrupo (`gender`, `SeniorCitizen`,
  `Contract`, `InternetService`).
- Drift de features (PSI > 0,2 sinaliza mudança significativa).
- Razão custo total observada vs prevista no funil de retenção.

## Reprodutibilidade

```bash
make install
python scripts/train_mlp.py --data data/raw/telco_churn.csv --out models/
make test
```

Artefatos gerados:

- `models/preprocessor.joblib` — `ColumnTransformer` treinado
- `models/mlp.pt` — `state_dict` + `input_dim` do MLP
- Runs MLflow em `notebooks/mlruns/`

## Responsabilidades

| Função | Descrição |
|---|---|
| **Owner do modelo** | Time de Data Science |
| **Owner da API**    | Time de MLOps / Plataforma |
| **Aprovação ética** | Risk & Compliance (revisar lista de vieses acima) |
| **Plantão**         | Oncall observa o dashboard de drift / latência (ver `docs/monitoring.md`) |
