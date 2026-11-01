import pandas as pd
import pytest

from src import exploration


@pytest.fixture
def jeu():
    return pd.DataFrame(
        {
            "trans_date_trans_time": pd.to_datetime(
                [
                    "2019-01-01 10:00:00",
                    "2019-01-01 10:30:00",
                    "2019-01-02 23:00:00",
                    "2019-02-03 02:00:00",
                    "2019-02-03 10:15:00",
                    "2019-02-04 10:45:00",
                ]
            ),
            "category": [
                "grocery_pos",
                "misc_net",
                "grocery_pos",
                "travel",
                "shopping_net",
                "misc_net",
            ],
            "amt": [10.0, 100.0, 20.0, 300.0, 50.0, 40.0],
            "state": ["NC", "NC", "WA", "WA", "ID", "ID"],
            "is_fraud": [0, 1, 0, 1, 0, 1],
        }
    )


def test_canal():
    assert exploration.canal("grocery_pos") == "magasin"
    assert exploration.canal("shopping_net") == "en ligne"
    assert exploration.canal("travel") == "non precise"


def test_resume_global(jeu):
    r = exploration.resume_global(jeu).iloc[0]
    assert r["transactions"] == 6
    assert r["fraudes"] == 3
    assert r["taux_fraude"] == pytest.approx(0.5)
    assert r["montant_total"] == pytest.approx(520.0)
    assert r["montant_fraudes"] == pytest.approx(440.0)
    assert r["part_montant_fraudes"] == pytest.approx(440.0 / 520.0)


def test_montants_par_classe(jeu):
    t = exploration.montants_par_classe(jeu)
    assert t.loc[0, "transactions"] == 3
    assert t.loc[0, "moyenne"] == pytest.approx(80.0 / 3)
    assert t.loc[1, "mediane"] == pytest.approx(100.0)
    assert t.loc[1, "maximum"] == pytest.approx(300.0)


def test_taux_par_mois(jeu):
    t = exploration.taux_par_mois(jeu)
    assert list(t.index) == ["2019-01", "2019-02"]
    assert t.loc["2019-01", "fraudes"] == 1
    assert t.loc["2019-02", "fraudes"] == 2
    assert t.loc["2019-02", "montant_fraudes"] == pytest.approx(340.0)


def test_taux_par_heure(jeu):
    t = exploration.taux_par_heure(jeu)
    assert t.loc[10, "transactions"] == 4
    assert t.loc[10, "fraudes"] == 2
    assert t.loc[10, "taux_fraude"] == pytest.approx(0.5)
    assert t.loc[2, "taux_fraude"] == pytest.approx(1.0)
    assert t.loc[23, "taux_fraude"] == pytest.approx(0.0)


def test_taux_par_canal(jeu):
    t = exploration.taux_par_canal(jeu)
    assert t.loc["magasin", "transactions"] == 2
    assert t.loc["magasin", "fraudes"] == 0
    assert t.loc["en ligne", "transactions"] == 3
    assert t.loc["en ligne", "fraudes"] == 2
    assert t.loc["non precise", "fraudes"] == 1


def test_taux_par_categorie_trie(jeu):
    t = exploration.taux_par_categorie(jeu)
    assert t["taux_fraude"].is_monotonic_decreasing
    assert t.iloc[0]["taux_fraude"] == pytest.approx(1.0)
    assert t.loc["grocery_pos", "taux_fraude"] == pytest.approx(0.0)


def test_taux_par_jour_semaine(jeu):
    t = exploration.taux_par_jour_semaine(jeu)
    assert set(t.index) <= set(exploration.JOURS.values())
    assert t["transactions"].sum() == len(jeu)


@pytest.mark.parametrize(
    "fonction",
    [
        exploration.taux_par_mois,
        exploration.taux_par_heure,
        exploration.taux_par_jour_semaine,
        exploration.taux_par_categorie,
        exploration.taux_par_canal,
        exploration.taux_par_etat,
    ],
)
def test_conservation_des_effectifs(jeu, fonction):
    t = fonction(jeu)
    assert t["transactions"].sum() == len(jeu)
    assert t["fraudes"].sum() == int(jeu["is_fraud"].sum())
    assert t["montant_fraudes"].sum() == pytest.approx(440.0)
