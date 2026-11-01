import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from src import metriques

TABLEAU_COURS = {
    "logistique": ((27, 30, 15, 56889), dict(mcc=0.551)),
    "logistique_smote": (
        (46, 11, 140, 56764),
        dict(accuracy=0.9973, precision=0.247, rappel=0.807, f1=0.379, f2=0.556, mcc=0.446),
    ),
    "isolation_forest": (
        (16, 41, 554, 56350),
        dict(accuracy=0.9896, precision=0.028, rappel=0.281, f1=0.051, f2=0.100, mcc=0.086),
    ),
    "autoencodeur": (
        (41, 16, 529, 56375),
        dict(accuracy=0.9904, precision=0.072, rappel=0.719, f1=0.131, f2=0.257, mcc=0.226),
    ),
    "dbscan": (
        (34, 23, 508, 56396),
        dict(accuracy=0.9907, precision=0.063, rappel=0.596, f1=0.114, f2=0.221, mcc=0.191),
    ),
}


@pytest.mark.parametrize("nom", list(TABLEAU_COURS))
def test_indicateurs_contre_le_cours(nom):
    comptes, attendus = TABLEAU_COURS[nom]
    obtenus = metriques.indicateurs_confusion(*comptes)
    for cle, valeur in attendus.items():
        assert obtenus[cle] == pytest.approx(valeur, abs=1.5e-3), (nom, cle)


def test_alertes_du_cours():
    assert metriques.indicateurs_confusion(27, 30, 15, 56889)["alertes"] == 42
    assert metriques.indicateurs_confusion(49, 8, 3313, 53591)["alertes"] == 3362


def test_modele_muet():
    r = metriques.indicateurs_confusion(0, 57, 0, 56904)
    assert r["accuracy"] == pytest.approx(56904 / 56961)
    assert r["precision"] == 0.0
    assert r["rappel"] == 0.0
    assert r["mcc"] == 0.0


def test_matrice_confusion():
    y = [1, 1, 0, 0, 0, 1]
    alertes = [1, 0, 1, 0, 0, 1]
    assert metriques.matrice_confusion(y, alertes) == (2, 1, 1, 2)


def test_precision_rappel_a_k():
    y = [1, 0, 1, 0, 0, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.1]
    assert metriques.precision_a_k(y, scores, 3) == pytest.approx(2 / 3)
    assert metriques.rappel_a_k(y, scores, 3) == pytest.approx(2 / 3)
    assert metriques.precision_a_k(y, scores, 1) == pytest.approx(1.0)
    assert metriques.rappel_a_k(y, scores, 0) == 0.0
    assert metriques.rappel_a_k(y, scores, 100) == pytest.approx(1.0)


def test_auc_mann_whitney():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(int)
    scores = rng.random(500) + 0.4 * y
    rangs = rankdata(scores)
    n1 = y.sum()
    n0 = len(y) - n1
    attendu = (rangs[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    assert metriques.roc_auc(y, scores) == pytest.approx(attendu)


def test_pr_auc_reference():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    parfait = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.05, 0.04, 0.03, 0.02, 0.01])
    assert metriques.pr_auc(y, parfait) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    y = (rng.random(20000) < 0.01).astype(int)
    hasard = rng.random(20000)
    assert metriques.pr_auc(y, hasard) == pytest.approx(y.mean(), abs=0.01)
    assert metriques.roc_auc(y, hasard) == pytest.approx(0.5, abs=0.07)


def test_cout_au_seuil():
    y = [1, 1, 0, 0, 1]
    scores = [0.9, 0.4, 0.8, 0.1, 0.3]
    montants = [100.0, 200.0, 50.0, 10.0, 400.0]
    assert metriques.cout_au_seuil(y, scores, montants, 0.5, 5.0) == pytest.approx(600.0 + 10.0)
    assert metriques.cout_au_seuil(y, scores, montants, 0.0, 5.0) == pytest.approx(25.0)
    assert metriques.cout_au_seuil(y, scores, montants, 2.0, 5.0) == pytest.approx(700.0)
    assert metriques.cout_sans_modele(y, montants) == pytest.approx(700.0)


def brute_force(y, scores, montants, ca):
    candidats = sorted(set(scores)) + [max(scores) + 1.0]
    return min(metriques.cout_au_seuil(y, scores, montants, s, ca) for s in candidats)


@pytest.mark.parametrize("graine", [0, 1, 2, 3])
def test_seuil_optimal_equivaut_au_balayage(graine):
    rng = np.random.default_rng(graine)
    n = 300
    y = (rng.random(n) < 0.08).astype(int)
    scores = np.round(rng.random(n) * 0.6 + 0.4 * y * rng.random(n), 2)
    montants = rng.gamma(2.0, 60.0, n).round(2)
    resultat = metriques.seuil_optimal(y, scores, montants, 5.0)
    assert resultat["cout"] == pytest.approx(brute_force(y, list(scores), montants, 5.0))
    recalcul = metriques.cout_au_seuil(y, scores, montants, resultat["seuil"], 5.0)
    assert recalcul == pytest.approx(resultat["cout"])
    assert resultat["cout"] <= resultat["cout_sans_modele"] + 1e-9


def test_seuil_optimal_sans_alerte_si_cout_prohibitif():
    y = [1, 0, 0, 1]
    scores = [0.9, 0.5, 0.4, 0.3]
    montants = [2.0, 10.0, 10.0, 1.0]
    r = metriques.seuil_optimal(y, scores, montants, 50.0)
    assert r["alertes"] == 0
    assert r["cout"] == pytest.approx(3.0)


def test_courbe_cout_ex_aequo():
    y = [1, 0, 1, 0]
    scores = [0.5, 0.5, 0.5, 0.2]
    montants = [100.0, 10.0, 100.0, 10.0]
    courbe = metriques.courbe_cout(y, scores, montants, 5.0)
    assert list(courbe["alertes"]) == [0, 3, 4]


def test_budget_total():
    debut = pd.Timestamp("2020-04-07 07:00:00")
    fin = pd.Timestamp("2020-04-17 07:00:00")
    assert metriques.budget_total(30, debut, fin) == 300


def test_evaluer():
    rng = np.random.default_rng(5)
    n = 2000
    y = (rng.random(n) < 0.02).astype(int)
    scores = rng.random(n) + 0.8 * y
    montants = rng.gamma(2.0, 50.0, n)
    r = metriques.evaluer(y, scores, montants, 5.0, k=40)
    assert r["alertes"] == r["vp"] + r["fp"]
    assert r["cout"] <= r["cout_sans_modele"] + 1e-9
    assert 0.0 <= r["pr_auc"] <= 1.0
    assert r["roc_auc"] > 0.5
