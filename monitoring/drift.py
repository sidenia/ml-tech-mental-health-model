"""Análise de data drift básica.

Uso:
    python -m monitoring.drift

Compara a distribuição das entradas recentes (monitoring/predictions.log)
com a distribuição do treino (data/processed/clean.csv):
- age (numérica): teste Kolmogorov-Smirnov
- categóricas: teste qui-quadrado

p-valor < 0.05 = distribuição mudou = investigar (possível drift).
"""

import json
from pathlib import Path

import pandas as pd
from scipy import stats

LOG_PATH = Path("monitoring/predictions.log")
CLEAN_CSV = Path("data/processed/clean.csv")
ALPHA = 0.05


def load_logged_inputs() -> pd.DataFrame:
    records = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines()]
    return pd.DataFrame([r["input"] for r in records])


def main() -> None:
    if not LOG_PATH.exists():
        print("Sem predições logadas ainda. Rode a API e faça predições primeiro.")
        return

    train = pd.read_csv(CLEAN_CSV)
    recent = load_logged_inputs()
    recent.columns = [c.lower() for c in recent.columns]
    print(f"Comparando {len(recent)} predições recentes vs {len(train)} do treino\n")

    # numérica: KS
    ks = stats.ks_2samp(train["age"], recent["age"])
    flag = "DRIFT?" if ks.pvalue < ALPHA else "ok"
    print(f"age: KS p-valor = {ks.pvalue:.4f} [{flag}]")

    # categóricas: qui-quadrado sobre frequências
    for col in recent.columns:
        if col == "age" or col not in train.columns:
            continue
        train_freq = train[col].str.lower().value_counts()
        recent_freq = recent[col].str.lower().value_counts()
        categories = train_freq.index.union(recent_freq.index)
        table = pd.DataFrame(
            {
                "train": train_freq.reindex(categories, fill_value=0),
                "recent": recent_freq.reindex(categories, fill_value=0),
            }
        )
        # qui-quadrado exige contagens > 0 em pelo menos uma célula por linha
        table = table[table.sum(axis=1) > 0]
        chi2 = stats.chi2_contingency(table.T)
        flag = "DRIFT?" if chi2.pvalue < ALPHA else "ok"
        print(f"{col}: qui2 p-valor = {chi2.pvalue:.4f} [{flag}]")


if __name__ == "__main__":
    main()
