"""Obtenção e limpeza dos dados.

Uso:
    python -m src.data

Baixa o dataset OSMI Mental Health in Tech Survey (Kaggle) para data/raw/
e gera a versão limpa em data/processed/clean.csv.

Requer credenciais Kaggle: https://github.com/Kaggle/kagglehub#authenticate
"""

from pathlib import Path
import shutil

import kagglehub
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_CSV = RAW_DIR / "survey.csv"
CLEAN_CSV = PROCESSED_DIR / "clean.csv"

KAGGLE_DATASET = "osmi/mental-health-in-tech-survey"

# Normalização do campo Gender (texto livre no bruto).
# Completar as variações restantes durante a exploração (notebook 01).
GENDER_MAP = {
    "male": "male",
    "m": "male",
    "man": "male",
    "female": "female",
    "f": "female",
    "woman": "female",
}

COLUMNS_TO_KEEP = [
    "Age",
    "Gender",
    "self_employed",
    "family_history",
    "work_interfere",
    "no_employees",
    "remote_work",
    "tech_company",
    "benefits",
    "care_options",
    "treatment",  # alvo
]


def download() -> Path:
    """Baixa o dataset do Kaggle e copia o CSV para data/raw/."""
    dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    source = next(dataset_dir.glob("*.csv"))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, RAW_CSV)
    print(f"Dado bruto salvo em {RAW_CSV}")
    return RAW_CSV


def clean(raw_csv: Path = RAW_CSV) -> pd.DataFrame:
    """Aplica a limpeza documentada em docs/data_dictionary.md."""
    df = pd.read_csv(raw_csv)
    df = df[COLUMNS_TO_KEEP].copy()

    # Age: valores absurdos no bruto (negativos, >100)
    df = df[df["Age"].between(18, 75)]

    # Categóricas em minúsculo — API e treino precisam falar a mesma língua,
    # senão o OneHotEncoder ignora silenciosamente valores com caixa diferente
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].str.strip().str.lower()

    # Gender: texto livre -> male/female/other
    df["Gender"] = df["Gender"].map(GENDER_MAP).fillna("other")

    # Nulos
    df["self_employed"] = df["self_employed"].fillna("no")
    df["work_interfere"] = df["work_interfere"].fillna("unknown")

    df.columns = [c.lower() for c in df.columns]
    return df


def main() -> None:
    if not RAW_CSV.exists():
        download()
    df = clean()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"{len(df)} linhas limpas salvas em {CLEAN_CSV}")


if __name__ == "__main__":
    main()
