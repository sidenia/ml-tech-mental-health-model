"""Treino e seleção de modelo.

Uso:
    python -m src.train

Lê data/processed/clean.csv, compara modelos via validação cruzada,
ajusta o melhor com GridSearchCV e salva o pipeline completo em
models/model.joblib. Métricas impressas no console — copiar para
docs/technical.md (seção 5).
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLEAN_CSV = Path("data/processed/clean.csv")
MODEL_PATH = Path("models/model.joblib")

TARGET = "treatment"
NUMERIC = ["age"]
CATEGORICAL = [
    "gender",
    "self_employed",
    "family_history",
    "work_interfere",
    "no_employees",
    "remote_work",
    "tech_company",
    "benefits",
    "care_options",
]

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}

# Grades de ajuste fino — expandir durante o notebook 03
PARAM_GRIDS = {
    "logistic_regression": {"model__C": [0.01, 0.1, 1, 10]},
    "random_forest": {
        "model__n_estimators": [100, 300],
        "model__max_depth": [None, 5, 10],
    },
    "gradient_boosting": {
        "model__n_estimators": [100, 300],
        "model__learning_rate": [0.05, 0.1],
    },
}


def build_pipeline(model) -> Pipeline:
    """Pré-processamento + modelo num único pipeline (evita vazamento)."""
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def main() -> None:
    df = pd.read_csv(CLEAN_CSV)
    X = df.drop(columns=[TARGET])
    y = (df[TARGET].str.lower() == "yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    baseline_f1 = f1_score(y_test, [y_train.mode()[0]] * len(y_test))
    print(f"Baseline (classe majoritária) F1: {baseline_f1:.3f}\n")

    # 1) comparação por validação cruzada
    results = {}
    for name, model in MODELS.items():
        pipe = build_pipeline(model)
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1")
        results[name] = scores.mean()
        print(f"{name}: F1 CV = {scores.mean():.3f} (+/- {scores.std():.3f})")

    best_name = max(results, key=results.get)
    print(f"\nMelhor na CV: {best_name}. Ajustando com GridSearchCV...")

    # 2) ajuste fino do melhor
    grid = GridSearchCV(
        build_pipeline(MODELS[best_name]),
        PARAM_GRIDS[best_name],
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"Melhores hiperparâmetros: {grid.best_params_}")

    # 3) avaliação final no teste — UMA única vez
    y_pred = grid.predict(X_test)
    print("\nAvaliação no conjunto de teste:")
    print(classification_report(y_test, y_pred, target_names=["no", "yes"]))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(grid.best_estimator_, MODEL_PATH)
    print(f"Modelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()
