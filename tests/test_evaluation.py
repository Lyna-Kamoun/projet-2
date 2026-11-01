import numpy as np
import pandas as pd
import pytest

from src import evaluation, figures, metriques


def faux_jeu(n=6000, graine=0, debut="2020-04-01"):
    rng = np.random.default_rng(graine)
    secondes = np.sort(rng.integers(0, 40 * 86400, n))
    y = (rng.random(n) < 0.03).astype(int)
    jeu = pd.DataFrame(
        {
            "trans_date_trans_time": pd.Timestamp(debut) + pd.to_timedelta(secondes, unit="s"),
            "amt": rng.gamma(2.0, 60.0, n).round(2) + 1.0,
            "is_fraud": y,
        }
    )
    force = {"xgboost": 3.0, "logistique": 1.5, "isolation_forest": 0.8,
             "one_class_svm": 0.6, "autoencodeur": 0.5, "lof": 0.2}
    for methode, f in force.items():
        brut = rng.normal(0, 1, n) + f * y
        jeu[methode] = 1.0 / (1.0 + np.exp(-brut))
    return jeu


@pytest.fixture(scope="module")
def jeux():
    return {"validation": faux_jeu(graine=1), "test": faux_jeu(graine=2, debut="2020-06-01")}


def test_tableau_comparatif_structure(jeux):
    t = evaluation.tableau_comparatif(jeux)
    assert len(t) == len(evaluation.METHODES) * 2
    assert set(t["jeu"]) == {"validation", "test"}
    for colonne in ["pr_auc", "roc_auc", "precision_a_k", "rappel_a_k", "f2", "mcc", "cout", "economie"]:
        assert colonne in t.columns
    assert t["pr_auc"].between(0, 1).all()


def test_seuil_choisi_sur_la_validation_et_applique_au_test(jeux):
    t = evaluation.tableau_comparatif(jeux, methodes=("xgboost",))
    val = t[t["jeu"] == "validation"].iloc[0]
    test = t[t["jeu"] == "test"].iloc[0]
    optimum = evaluation.seuil_optimal_validation(jeux, "xgboost")
    assert val["seuil"] == pytest.approx(optimum["seuil"])
    assert test["seuil"] == pytest.approx(val["seuil"])
    assert val["cout"] == pytest.approx(optimum["cout"])
    assert val["alertes"] == optimum["alertes"]
    j = jeux["test"]
    recalcul = metriques.cout_au_seuil(j["is_fraud"], j["xgboost"], j["amt"], optimum["seuil"], 5.0)
    assert test["cout"] == pytest.approx(recalcul)
    assert test["cout"] >= test["cout_optimal_du_test"] - 1e-9
    assert test["economie"] == pytest.approx(1 - test["cout"] / test["cout_sans_modele"])


def test_le_meilleur_modele_coute_moins(jeux):
    t = evaluation.tableau_comparatif(jeux)
    v = t[t["jeu"] == "validation"].set_index("methode")
    assert v.loc["xgboost", "cout"] < v.loc["lof", "cout"]
    assert v.loc["xgboost", "pr_auc"] > v.loc["lof", "pr_auc"]


def test_sensibilite_au_cout_d_analyse(jeux):
    s = evaluation.sensibilite_cout(jeux, "xgboost")
    assert list(s["cout_alerte"]) == list(evaluation.VALEURS_COUT_ALERTE)
    assert s["alertes_validation"].is_monotonic_decreasing
    assert (s["cout_validation"] <= jeux["validation"].eval("amt * is_fraud").sum() + 1e-9).all()


def test_selon_budget_croissant(jeux):
    b = evaluation.selon_budget(jeux, methodes=("xgboost",))
    for nom in ("validation", "test"):
        r = b[b["jeu"] == nom].sort_values("alertes_par_jour")["rappel_a_k"]
        assert r.is_monotonic_increasing
    assert set(b["alertes_par_jour"]) == set(evaluation.BUDGETS_PAR_JOUR)


def test_complementarite_sur_exemple():
    jeu = pd.DataFrame(
        {
            "trans_date_trans_time": pd.date_range("2020-01-01", periods=10, freq="1D"),
            "amt": 10.0,
            "is_fraud": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "base": [0.9, 0.8, 0.1, 0.1, 0.7, 0.6, 0.0, 0.0, 0.0, 0.0],
            "appoint": [0.0, 0.0, 0.9, 0.2, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1],
        }
    )
    resultat = evaluation.complementarite(jeu, "base", "appoint", alertes_par_jour=0.5)
    assert resultat["jeu_k"] == 4
    assert resultat["detectees_par_base"] == 2
    assert resultat["manquees_par_base"] == 2
    assert resultat["recuperees_par_appoint"] == 2
    assert resultat["rappel_union"] == pytest.approx(1.0)
    assert resultat["rappel_base"] == pytest.approx(0.5)


def test_courbe_precision_rappel(jeux):
    rappel, precision = evaluation.courbe_precision_rappel(jeux["validation"], "xgboost")
    assert rappel.max() == pytest.approx(1.0) and rappel.min() == pytest.approx(0.0)
    assert len(rappel) <= 1500
    assert precision.min() >= 0 and precision.max() <= 1


def test_courbe_cout(jeux):
    c = evaluation.courbe_cout_selon_alertes(jeux["validation"], "xgboost")
    assert c["alertes"].iloc[0] == 0
    assert c["alertes_par_jour"].is_monotonic_increasing
    assert c["cout"].iloc[0] == pytest.approx(jeux["validation"].eval("amt * is_fraud").sum())


def test_figures_creees(jeux, tmp_path):
    a = figures.figure_courbes_precision_rappel(jeux, tmp_path / "pr.png")
    b = figures.figure_cout_selon_alertes(jeux, tmp_path / "cout.png")
    assert a.stat().st_size > 20000
    assert b.stat().st_size > 10000


def test_chargement_refuse_des_index_differents(tmp_path):
    base = pd.DataFrame(
        {"trans_date_trans_time": pd.date_range("2020-01-01", periods=3), "amt": [1.0, 2.0, 3.0],
         "is_fraud": [0, 1, 0], "proba_xgboost": 0.1, "proba_logistique": 0.2}
    )
    base.to_parquet(tmp_path / "scores_validation.parquet")
    base.to_parquet(tmp_path / "scores_test.parquet")
    autre = pd.DataFrame({m: [0.1, 0.2, 0.3] for m in evaluation.METHODES_NON_SUPERVISEES}, index=[5, 6, 7])
    autre.to_parquet(tmp_path / "scores_non_supervise_validation.parquet")
    autre.to_parquet(tmp_path / "scores_non_supervise_test.parquet")
    with pytest.raises(ValueError):
        evaluation.charger_scores(tmp_path)
