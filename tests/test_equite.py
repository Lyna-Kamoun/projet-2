import numpy as np
import pandas as pd
import pytest

from src import donnees, equite, modeles_supervises
from tests.test_desequilibre import jeu_fraude


def test_tranches_d_age_aux_bornes():
    ages = pd.Series([18.0, 24.99, 25.0, 34.99, 35.0, 64.99, 65.0, 95.0])
    t = equite.tranche_age(ages).astype(str).tolist()
    assert t == [
        "moins de 25 ans", "moins de 25 ans", "25 à 34 ans", "25 à 34 ans",
        "35 à 44 ans", "55 à 64 ans", "65 ans et plus", "65 ans et plus",
    ]


def jeu_exemple():
    return pd.DataFrame(
        {
            "is_fraud": [1, 1, 0, 0, 0, 0, 1, 0],
            "cc_num": [1, 1, 2, 2, 3, 3, 4, 4],
        }
    )


def test_indicateurs_par_groupe():
    jeu = jeu_exemple()
    alertes = [1, 0, 1, 0, 0, 0, 1, 1]
    groupes = ["a", "a", "a", "a", "b", "b", "b", "b"]
    t = equite.indicateurs_par_groupe(jeu, alertes, groupes)
    assert t.loc["a", "transactions"] == 4
    assert t.loc["a", "cartes"] == 2
    assert t.loc["a", "fraudes"] == 2
    assert t.loc["a", "taux_alerte"] == pytest.approx(0.5)
    assert t.loc["a", "rappel"] == pytest.approx(0.5)
    assert t.loc["a", "precision"] == pytest.approx(0.5)
    assert t.loc["a", "taux_fausses_alertes"] == pytest.approx(0.5)
    assert t.loc["b", "rappel"] == pytest.approx(1.0)
    assert t.loc["b", "taux_fausses_alertes"] == pytest.approx(1 / 3)


def test_groupe_sans_fraude_donne_nan():
    jeu = pd.DataFrame({"is_fraud": [0, 0, 1], "cc_num": [1, 2, 3]})
    t = equite.indicateurs_par_groupe(jeu, [1, 0, 1], ["x", "x", "y"])
    assert np.isnan(t.loc["x", "rappel"])
    assert t.loc["y", "rappel"] == 1.0


def test_ecarts():
    t = pd.DataFrame(
        {
            "taux_alerte": [0.01, 0.02, 0.04],
            "rappel": [0.9, 0.8, np.nan],
            "taux_fausses_alertes": [0.0, 0.01, 0.02],
        }
    )
    e = equite.ecarts(t)
    assert e["rapport_taux_alerte"] == pytest.approx(4.0)
    assert e["ecart_rappel"] == pytest.approx(0.1)
    assert np.isnan(e["rapport_taux_fausses_alertes"])


def test_variante_avec_age_et_genre():
    df = jeu_fraude(n=24000)
    e, v, t = donnees.decouper_chronologiquement(df)
    petit = {"n_estimators": 40, "max_depth": 3}
    base = modeles_supervises.entrainer_xgboost(e, petit, n_jobs=1)
    variante = modeles_supervises.entrainer_xgboost(
        e, petit, n_jobs=1, optionnelles=True, nom="xgboost_avec_age_genre"
    )
    colonnes_base = list(base.matrice(v).columns)
    colonnes_variante = list(variante.matrice(v).columns)
    assert "age" not in colonnes_base and "genre_f" not in colonnes_base
    assert {"age", "genre_f"} <= set(colonnes_variante)
    assert variante.nom == "xgboost_avec_age_genre"
    par_genre, par_age = equite.tables_equite(v, base.probabilites(v), 0.05)
    assert set(par_genre.index) == {"femmes", "hommes"}
    assert par_genre["transactions"].sum() == len(v)
    assert par_age["transactions"].sum() == len(v)
