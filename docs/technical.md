# Documentação Técnica

## 1. Enquadramento do problema

- **Pergunta de negócio:** dado o perfil de um profissional de tecnologia, ele buscará tratamento de saúde mental?
- **Tipo:** classificação binária supervisionada. Alvo: `treatment` (yes/no).
- **Métrica principal:** F1-score. Justificativa: custo de falso negativo (pessoa que precisaria de ajuda e não é identificada) é alto. (A EDA mostrou alvo equilibrado — 50,6% yes / 49,4% no — então acurácia também seria aceitável; F1 mantido por dar peso ao recall da classe positiva.)
- **Métricas secundárias:** precisão, recall, ROC-AUC.
- **Baseline:** prever sempre a classe majoritária.

## 2. Dados

- **Fonte:** OSMI Mental Health in Tech Survey 2014 (Kaggle).
- **Volume:** ~1.250 respostas, 27 colunas.
- **Problemas conhecidos do dado bruto:**
  - `Age` com valores absurdos (negativos, > 100) — filtrar para faixa 18–75.
  - `Gender` em texto livre (~49 variações) — normalizar para categorias.
  - `self_employed` e `work_interfere` com nulos — imputar/categorizar como "unknown".
- Detalhes coluna a coluna: ver [data_dictionary.md](data_dictionary.md).

## 3. Preparação

- Split treino/teste estratificado (80/20, `random_state=42`).
- Todo pré-processamento dentro de `Pipeline` + `ColumnTransformer` do scikit-learn — evita vazamento de dados (fit só no treino).
- Categóricas: `OneHotEncoder(handle_unknown="ignore")`.
- Numéricas (`age`): `StandardScaler`.

## 4. Modelos avaliados

| Modelo | Por quê |
|---|---|
| LogisticRegression | baseline linear, interpretável |
| RandomForestClassifier | não-linear, robusto, pouca sintonia |
| GradientBoostingClassifier | costuma vencer em tabular pequeno |

Seleção via validação cruzada estratificada (5 folds) no treino. Ajuste fino com `GridSearchCV` no melhor. Teste usado **uma única vez**, no final.

## 5. Resultados

### 5.1 Descobertas da EDA

- Alvo equilibrado: 50,6% yes / 49,4% no (1.259 respostas brutas).
- `Age`: mínimo -1726, máximo 99999999999 — 8 linhas fora da faixa 18–75 removidas (sobram 1.251).
- `Gender`: 49 variações de texto livre → normalizado em 971 male / 242 female / 38 other.
- Nulos: `work_interfere` 264 (21%, virou "Unknown"), `self_employed` 18 (virou "No"), `state` 515 e `comments` 1.095 (colunas descartadas).
- Sinais fortes (taxa de treatment=yes por categoria): `family_history` yes 74% vs no 35%; `work_interfere` often 85% vs never 14%; female 69% vs male 45%. `remote_work` e `tech_company` ~50% dos dois lados (pouco informativas).

### 5.2 Comparação de modelos (validação cruzada, 5 folds, treino)

| Modelo | F1 CV (média ± desvio) |
|---|---|
| Baseline (classe majoritária) | 0.672 (no teste) |
| **LogisticRegression** | **0.840 ± 0.022** |
| GradientBoostingClassifier | 0.823 ± 0.008 |
| RandomForestClassifier | 0.802 ± 0.023 |

**Vencedor: LogisticRegression.** Dataset pequeno com features categóricas e relações quase lineares com o alvo — ensembles não têm padrão não-linear para explorar e só adicionam variância.

### 5.3 Ajuste fino

- GridSearchCV (5 folds, scoring F1) sobre `C ∈ {0.01, 0.1, 1, 10}`.
- **Melhor: `C = 0.1`** — regularização mais forte que o default (C=1), coerente com dataset pequeno.

### 5.4 Avaliação final (teste, 251 amostras, usado uma única vez)

| Métrica | Valor |
|---|---|
| F1 (yes) | 0.86 |
| Recall (yes) | 0.92 |
| Precisão (yes) | 0.81 |
| Acurácia | 0.85 |
| ROC-AUC | 0.921 |

Matriz de confusão:

|  | Previsto: no | Previsto: yes |
|---|---|---|
| **Real: no** | 97 (VN) | 27 (FP) |
| **Real: yes** | 10 (FN) | 117 (VP) |

Apenas 10 falsos negativos — o erro de maior custo do problema é o mais raro.

### 5.5 Coeficientes (interpretação)

Positivo empurra para "busca tratamento"; negativo, para "não busca":

| Feature | Coeficiente |
|---|---|
| work_interfere = Often | +1.15 |
| work_interfere = Sometimes | +1.10 |
| work_interfere = Rarely | +0.60 |
| family_history = Yes | +0.45 |
| care_options = Yes | +0.38 |
| benefits = Yes | +0.26 |
| gender = male | -0.21 |
| family_history = No | -0.44 |
| work_interfere = Never | -1.02 |
| work_interfere = Unknown | -1.82 |

Coerente com a literatura: percepção de interferência no trabalho e histórico familiar são os maiores preditores de busca por tratamento. O peso forte de `work_interfere = Unknown` (-1.82) sugere que **não responder** essa pergunta já é sinal informativo (quem não percebe interferência tende a não responder — e a não buscar tratamento).

## 6. Deploy

- Modelo final serializado com `joblib` (pipeline completo: pré-processamento + estimador).
- API FastAPI (`api/main.py`), endpoint `POST /predict`. Validação de entrada com Pydantic.

## 7. Monitoramento

- Cada predição logada (input + output + timestamp) em `monitoring/predictions.log` (JSON lines).
- `monitoring/drift.py`: compara distribuição das entradas recentes vs distribuição do treino (teste qui-quadrado para categóricas, KS para `age`). Alerta se p-valor < 0.05.

## 8. Limitações

- Dado de 2014, viés de auto-seleção (survey voluntária), maioria respondentes EUA.
- Modelo NÃO é ferramenta de diagnóstico. Uso educacional.

## 9. Decisões e histórico

- **2026-07-16 — Corte de colunas (27 → 11):** `Timestamp` (sem valor preditivo), `comments` (87% vazio), `state` (41% nulo, só EUA), `Country` (60% EUA, criaria categorias raras demais). Colunas de política da empresa (`anonymity`, `leave`, `coworkers`, `supervisor` etc.) cortadas para manter escopo didático — experimento futuro: reincluir e medir impacto no F1.
- **2026-07-16 — Métrica F1 mantida mesmo com alvo equilibrado:** prioriza recall da classe positiva (falso negativo é o erro mais caro).
- **2026-07-16 — Modelo final: LogisticRegression (C=0.1)** em vez dos ensembles — melhor F1 na CV e mais interpretável (coeficientes).
