import numpy as np
import pandas as pd
import pytest

from src import donnees, reglage
from tests.test_desequilibre import jeu_fraude


@pytest.fixture(scope="module")
def entrainement():
    df = jeu_fraude(n=20000)
    e, v, t = donnees.decouper_chronologiquement(df)
    return e


def test_plis_temporels(entrainement):
    plis = reglage.plis_temporels(entrainement)
    assert len(plis) == reglage.NOMBRE_BLOCS - 1
    tailles_evaluation = [len(p[1]) for p in plis]
    assert max(tailles_evaluation) - min(tailles_evaluation) <= 1
    col = donnees.COLONNE_DATE
    for appris, evalue in plis:
        assert appris[col].max() <= evalue[col].min()
        assert not set(appris.index) & set(evalue.index)
    assert [len(p[0]) for p in plis] == sorted(len(p[0]) for p in plis)


def test_plis_fenetre_croissante(entrainement):
    plis = reglage.plis_temporels(entrainement)
    for (a1, e1), (a2, e2) in zip(plis[:-1], plis[1:]):
        assert len(a2) == len(a1) + len(e1)


def test_sous_echantillonnage_dans_chaque_pli(entrainement):
    from src import preparation

    plis = reglage.preparer_plis(entrainement, preparation.pipeline_arbres, True, 0)
    for x_appris, y_appris, x_evalue, y_evalue in plis:
        assert (y_appris == 0).sum() == reglage.RATIO_SOUS_ECHANTILLONNAGE * y_appris.sum()
        assert len(x_evalue) == len(y_evalue)
        assert y_evalue.mean() < 0.2


def test_convertir_types():
    out = reglage.convertir({"a": np.int64(3), "b": np.float64(0.5)})
    assert type(out["a"]) is int
    assert type(out["b"]) is float


def test_recherche_xgboost(entrainement):
    resultats = reglage.rechercher_xgboost(entrainement, configurations=3, n_jobs=1)
    assert len(resultats) == 4
    assert resultats["reference"].sum() == 1
    assert resultats["pr_auc_moyenne"].is_monotonic_decreasing
    assert resultats["pr_auc_moyenne"].between(0, 1).all()
    for nom in reglage.DISTRIBUTIONS_XGBOOST:
        assert nom in resultats.columns
    assert resultats["max_depth"].between(3, 10).all()
    assert resultats["learning_rate"].between(0.03, 0.3).all()
    autres = resultats[~resultats["reference"]]
    assert autres["n_estimators"].between(150, 600).all()


def test_recherche_xgboost_reproductible(entrainement):
    a = reglage.rechercher_xgboost(entrainement, configurations=2, graine=5, n_jobs=1)
    b = reglage.rechercher_xgboost(entrainement, configurations=2, graine=5, n_jobs=1)
    cles = list(reglage.DISTRIBUTIONS_XGBOOST)
    pd.testing.assert_frame_equal(
        a[cles].reset_index(drop=True), b[cles].reset_index(drop=True)
    )


def test_recherche_logistique_et_meilleurs(entrainement):
    log = reglage.rechercher_logistique(entrainement, valeurs=(0.01, 1.0))
    xgb = reglage.rechercher_xgboost(entrainement, configurations=2, n_jobs=1)
    assert len(log) == 2
    assert log["pr_auc_moyenne"].is_monotonic_decreasing
    meilleurs = reglage.meilleurs_parametres(xgb, log)
    assert set(meilleurs) == {"xgboost", "logistique"}
    assert meilleurs["logistique"]["C"] in (0.01, 1.0)
    assert set(meilleurs["xgboost"]) == set(reglage.DISTRIBUTIONS_XGBOOST)


def resultats_xgboost_fictifs(gain):
    base = dict(reglage.desequilibre.PARAMETRES_XGBOOST)
    ligne_reference = dict(base, reference=True, pr_auc_moyenne=0.9000)
    tire = {
        "n_estimators": 400,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.6,
        "min_child_weight": 5,
        "gamma": 0.5,
        "reg_lambda": 3.0,
        "reference": False,
        "pr_auc_moyenne": 0.9000 + gain,
    }
    return pd.DataFrame([tire, ligne_reference]).sort_values(
        "pr_auc_moyenne", ascending=False
    ).reset_index(drop=True)


def test_reference_conservee_si_gain_negligeable():
    for gain in (0.0005, -0.01, 0.001):
        choisi = reglage.choisir_xgboost(resultats_xgboost_fictifs(gain))
        assert choisi == reglage.convertir(dict(reglage.desequilibre.PARAMETRES_XGBOOST))
        assert not any(pd.isna(v) for v in choisi.values())
        assert "gamma" not in choisi


def test_configuration_tiree_adoptee_si_gain_suffisant():
    choisi = reglage.choisir_xgboost(resultats_xgboost_fictifs(0.01))
    assert choisi["max_depth"] == 8
    assert choisi["gamma"] == pytest.approx(0.5)
    assert type(choisi["n_estimators"]) is int


def test_choix_de_c_le_plus_regularise_parmi_les_equivalents():
    resultats = pd.DataFrame(
        {
            "C": [100.0, 10.0, 1.0, 0.1, 0.01],
            "pr_auc_moyenne": [0.4500, 0.4499, 0.4495, 0.4449, 0.4281],
        }
    )
    assert reglage.choisir_c(resultats) == 1.0
    plat = pd.DataFrame({"C": [10.0, 0.1], "pr_auc_moyenne": [0.45, 0.45]})
    assert reglage.choisir_c(plat) == 0.1
