"""Testes da limpeza de dados. Rodar: pytest"""

import pandas as pd
import pytest

from src.data import GENDER_MAP, clean


@pytest.fixture
def raw_csv(tmp_path):
    """CSV bruto mínimo com os problemas conhecidos do dado real."""
    df = pd.DataFrame(
        {
            "Age": [29, -5, 320, 45],
            "Gender": ["Male", "F", "Woman", "non-binary"],
            "self_employed": [None, "Yes", "No", "No"],
            "family_history": ["Yes", "No", "Yes", "No"],
            "work_interfere": ["Sometimes", None, "Often", "Never"],
            "no_employees": ["26-100", "1-5", "1000+", "6-25"],
            "remote_work": ["No", "Yes", "No", "Yes"],
            "tech_company": ["Yes", "Yes", "No", "Yes"],
            "benefits": ["Yes", "No", "Don't know", "Yes"],
            "care_options": ["No", "Yes", "Not sure", "No"],
            "treatment": ["Yes", "No", "Yes", "No"],
        }
    )
    path = tmp_path / "survey.csv"
    df.to_csv(path, index=False)
    return path


def test_age_out_of_range_removed(raw_csv):
    df = clean(raw_csv)
    assert df["age"].between(18, 75).all()
    assert len(df) == 2  # -5 e 320 removidos


def test_gender_normalized(raw_csv):
    df = clean(raw_csv)
    assert set(df["gender"]).issubset({"male", "female", "other"})


def test_no_nulls_after_clean(raw_csv):
    df = clean(raw_csv)
    assert not df.isnull().any().any()


def test_columns_lowercase(raw_csv):
    df = clean(raw_csv)
    assert all(c == c.lower() for c in df.columns)


def test_gender_map_values_valid():
    assert set(GENDER_MAP.values()).issubset({"male", "female", "other"})
