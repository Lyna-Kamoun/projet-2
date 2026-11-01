import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import desequilibre, donnees, variables

CATEGORIES = ["grocery_pos", "misc_net", "travel", "shopping_net"]


def jeu_fraude(n=9000, graine=0):
    rng = np.random.default_rng(graine)
    secondes = np.sort(rng.integers(0, 90 * 86400, n))
    dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(secondes, unit="s")
    montant = rng.gamma(2.0, 40.0, n).round(2) + 1.0
    nuit = np.isin(dates.hour, [22, 23, 0, 1, 2, 3])
    logit = -6.0 + 1.6 * nuit + 0.012 * montant
    fraude = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(
        {
            "trans_date_trans_time": dates,
            "cc_num": rng.integers(1, 60, n),
            "category": rng.choice(CATEGORIES, n),
            "amt": montant,
            "lat": rng.uniform(30, 45, n),
            "long": rng.uniform(-120, -75, n),
            "merch_lat": rng.uniform(30, 45, n),
            "merch_long": rng.uniform(-120, -75, n),
            "dob": "1975-03-02",
            "gender": rng.choice(["F", "M"], n),
            "is_fraud": fraude,
        }
    )
    return variables.construire_variables(df)


def matrice(n=4000, taux=0.03, graine=0):
    rng = np.random.default_rng(graine)
    y = (rng.random(n) < taux).astype(int)
    x = pd.DataFrame(
        {"a": rng.normal(0, 1, n) + 1.5 * y, "b": rng.normal(0, 1, n) - 1.0 * y}
    )
    return x, y


def test_correction_exemple():
    assert desequilibre.corriger_probabilites(0.5, 0.1) == pytest.approx(0.05 / 0.55)
    assert desequilibre.corriger_probabilites(0.3, 1.0) == pytest.approx(0.3)
    p = desequilibre.corriger_probabilites(np.array([0.1, 0.5, 0.9]), 0.05)
    assert (np.diff(p) > 0).all()
    assert (p < np.array([0.1, 0.5, 0.9])).all()


def test_sous_echantillonnage_effectifs_et_beta():
    x, y = matrice()
    x_res, y_res, beta = desequilibre.sous_echantillonner(x, y, 10, 0)
    fraudes = int(y.sum())
    assert int(y_res.sum()) == fraudes
    assert int((y_res == 0).sum()) == 10 * fraudes
    assert beta == pytest.approx(10 * fraudes / int((y == 0).sum()))
    assert len(x_res) == len(y_res)


def test_sous_echantillonnage_depend_de_la_graine_et_ne_modifie_pas_l_entree():
    x, y = matrice()
    copie = x.copy()
    a = desequilibre.sous_echantillonner(x, y, 5, 0)[0]
    b = desequilibre.sous_echantillonner(x, y, 5, 1)[0]
    assert not a.index.equals(b.index)
    pd.testing.assert_frame_equal(x, copie)


def test_smote_effectifs_et_interpolation():
    x, y = matrice()
    x_res, y_res = desequilibre.sur_echantillonner_smote(x, y, 10, 0)
    normales = int((y == 0).sum())
    assert int((y_res == 0).sum()) == normales
    assert int(y_res.sum()) == pytest.approx(normales / 10, abs=1)
    originales = x[y == 1]
    nouvelles = x_res.iloc[len(x):]
    assert (y_res[len(x):] == 1).all()
    for colonne in x.columns:
        assert nouvelles[colonne].min() >= originales[colonne].min() - 1e-9
        assert nouvelles[colonne].max() <= originales[colonne].max() + 1e-9
    pd.testing.assert_frame_equal(x_res.iloc[: len(x)].reset_index(drop=True), x.reset_index(drop=True))


def test_correction_restaure_le_taux_reel():
    rng = np.random.default_rng(3)
    n = 60000
    x = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    logit = -4.2 + 1.2 * x["a"] - 0.8 * x["b"]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    x_res, y_res, beta = desequilibre.sous_echantillonner(x, y, 5, 0)
    modele = LogisticRegression(max_iter=1000).fit(x_res, y_res)
    brut = modele.predict_proba(x)[:, 1]
    corrige = desequilibre.corriger_probabilites(brut, beta)
    assert brut.mean() > 3 * y.mean()
    assert corrige.mean() == pytest.approx(y.mean(), rel=0.2)


def test_comparaison_des_strategies():
    df = jeu_fraude()
    e, v, t = donnees.decouper_chronologiquement(df)
    detail = desequilibre.comparer_strategies(
        e,
        v,
        graines=(0, 1),
        n_jobs=1,
        parametres_xgboost={"n_estimators": 25, "max_depth": 3},
    )
    attendues = {"aucune", "ponderation", "smote_1_10"} | {
        f"sous_1_{r}" for r in desequilibre.RATIOS_SOUS_ECHANTILLONNAGE
    }
    assert set(detail["strategie"]) == attendues
    assert set(detail["modele"]) == {"logistique", "xgboost"}
    assert len(detail) == 12 + 14
    assert len(detail[(detail.modele == "xgboost") & (detail.strategie == "aucune")]) == 2
    assert len(detail[(detail.modele == "logistique") & (detail.strategie == "aucune")]) == 1
    assert (detail["pr_auc"] > v["is_fraud"].mean()).all()
    assert detail["pr_auc"].between(0, 1).all()
    resume = desequilibre.resumer_strategies(detail)
    assert (resume.loc[resume.index.get_level_values("strategie") == "sous_1_5", "tirages"] == 2).all()
    assert resume.groupby(level="modele")["pr_auc"].apply(lambda s: s.is_monotonic_decreasing).all()


def test_filtre_sur_modele_et_strategie():
    df = jeu_fraude()
    e, v, t = donnees.decouper_chronologiquement(df)
    detail = desequilibre.comparer_strategies(
        e,
        v,
        graines=(1, 2),
        n_jobs=1,
        parametres_xgboost={"n_estimators": 20, "max_depth": 3},
        modeles=("xgboost",),
        strategies=("aucune", "ponderation"),
    )
    assert set(detail["modele"]) == {"xgboost"}
    assert set(detail["strategie"]) == {"aucune", "ponderation"}
    assert sorted(detail["graine"].unique()) == [1, 2]
    assert len(detail) == 4
