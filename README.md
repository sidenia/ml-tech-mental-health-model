# Mental Health Treatment Predictor (Model)

Modelo de classificação que prevê se um profissional de tecnologia buscará tratamento de saúde mental, com base na pesquisa [OSMI Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey).

> Projeto de aprendizado ponta a ponta: enquadramento do problema, obtenção de dados, análise exploratória, preparação, treino, ajuste, deploy (API) e monitoramento.

## O problema

Empresas de tecnologia têm altos índices de burnout e transtornos de saúde mental, mas nem todos os profissionais buscam tratamento. Prever quem tende a **não** buscar ajuda permite direcionar ações de RH e programas de bem-estar.

- **Tipo:** classificação binária — alvo `treatment` (yes/no)
- **Métrica principal:** F1-score
- **Baseline a superar:** prever sempre a classe majoritária

## Resultados

Comparação por validação cruzada (5 folds, conjunto de treino):

| Modelo | F1 (CV) |
|---|---|
| Baseline (classe majoritária) | 0.672 |
| **Regressão Logística** | **0.840 (± 0.022)** |
| Gradient Boosting | 0.823 (± 0.008) |
| Random Forest | 0.802 (± 0.023) |

Modelo final: **Regressão Logística** com `C=0.1` (via GridSearchCV). O modelo mais simples venceu — dataset pequeno (~1.250 linhas) com relações quase lineares favorece modelos regularizados em vez de ensembles complexos.

Avaliação no conjunto de teste (251 amostras, usado uma única vez):

| Métrica | Valor |
|---|---|
| F1 (classe "yes") | 0.86 |
| Recall (classe "yes") | 0.92 |
| Precisão (classe "yes") | 0.81 |
| Acurácia | 0.85 |
| ROC-AUC | 0.92 |

Matriz de confusão: 117 verdadeiros positivos, 97 verdadeiros negativos, 27 falsos positivos, **10 falsos negativos** — o erro mais caro (pessoa que precisaria de ajuda e passa despercebida) é o mais raro.

Fatores que mais pesam na predição (coeficientes): condição mental interferindo no trabalho com frequência (`work_interfere=Often`, +1.15) e histórico familiar (`family_history=Yes`, +0.45) empurram para "busca tratamento"; responder "Never" (-1.02) ou não informar (-1.82) empurram para "não busca".

## Como rodar

### 1. Instalação

```bash
git clone https://github.com/<seu-usuario>/ml-tech-mental-health-model.git
cd ml-tech-mental-health-model
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

### 2. Baixar e preparar os dados

Requer credenciais do Kaggle configuradas ([instruções](https://github.com/Kaggle/kagglehub#authenticate)).

```bash
python -m src.data
```

Baixa o CSV bruto para `data/raw/` e gera a versão limpa em `data/processed/`.

### 3. Treinar

```bash
python -m src.train
```

Treina os modelos, imprime métricas de validação cruzada e salva o melhor em `models/model.joblib`.

### 4. Servir a API

```bash
uvicorn api.main:app --reload
```

Documentação interativa em http://127.0.0.1:8000/docs

Exemplo de predição:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 29, "gender": "male", "family_history": "yes", "work_interfere": "sometimes", "benefits": "yes", "care_options": "no", "remote_work": "no", "tech_company": "yes"}'
```

Resposta:

```json
{"treatment": "yes", "probability": 0.81}
```

## Estrutura do projeto

```
├── README.md                  # este arquivo (documentação de usuário)
├── docs/
│   ├── technical.md           # documentação técnica: decisões, métricas, limitações
│   └── data_dictionary.md     # dicionário de dados
├── data/
│   ├── raw/                   # dado original (nunca editado, fora do git)
│   └── processed/             # dado limpo
├── notebooks/
│   ├── 01_exploracao.ipynb    # EDA: visualizar e descobrir
│   ├── 02_preparacao.ipynb    # limpeza, encoding, split
│   └── 03_modelagem.ipynb     # treino, comparação, tuning
├── src/
│   ├── data.py                # download + limpeza reproduzível
│   ├── train.py               # pipeline de treino
│   └── predict.py             # carrega modelo salvo e prediz
├── models/                    # modelos serializados (.joblib)
├── api/main.py                # FastAPI: endpoint /predict
├── monitoring/                # log de predições + análise de drift
└── tests/                     # testes com pytest
```

# Objetivo
Entender o conceito de classifier (Classificação).

## Documentação

- [Documentação técnica](docs/technical.md) — decisões de modelagem e porquês
- [Dicionário de dados](docs/data_dictionary.md) — cada coluna explicada

## Licença

MIT. Dados: OSMI, licença [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
