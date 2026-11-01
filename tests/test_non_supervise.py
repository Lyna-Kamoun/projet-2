import numpy as np
import pandas as pd
import pytest

from src import donnees, metriques, non_supervise
from tests.test_desequilibre import jeu_fraude


def nuage(n=2500, anomalies=60, graine=0, dimension=6):
    rng = np.random.default_rng(graine)
    normales = rng.normal(0, 1, (n, dimension))
    lointaines = rng.normal(0, 1, (anomalies, dimension)) * 0.5 + 6.0
    x = np.vstack([normales, lointaines])
    y = np.r_[np.zeros(n, dtype=int), np.ones(anomalies, dtype=int)]
    ordre = rng.permutation(len(y))
    return pd.DataFrame(x[ordre]), y[ordre]


@pytest.fixture(scope="module")
def jeux():
    df = jeu_fraude(n=16000)
    return donnees.decouper_chronologiquement(df)


@pytest.fixture(scope="module")
def modeles(jeux):
    e = jeux[0]
    return {
        "isolation_forest": non_supervise.entrainer_isolation_forest(e, n_jobs=1),
        "lof": non_supervise.entrainer_lof(e, taille=2000, n_jobs=1),
        "one_class_svm": non_supervise.entrainer_one_class_svm(e, "is_fraud", taille=1500),
        "autoencodeur": non_supervise.entrainer_autoencodeur(e, taille=5000),
    }


def test_scores_detectent_des_points_lointains():
    x, y = nuage()
    apprentissage = x[y == 0]
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM

    scores = {
        "isolation": -IsolationForest(random_state=0).fit(apprentissage).score_samples(x),
        "lof": -LocalOutlierFactor(novelty=True).fit(apprentissage).score_samples(x),
        "svm": -OneClassSVM(nu=0.01).fit(apprentissage).decision_function(x),
    }
    for valeurs in scores.values():
        assert metriques.roc_auc(y, valeurs) > 0.99


def test_autoencodeur_erreur_de_reconstruction_plus_forte_pour_les_anomalies():
    x, y = nuage()
    from sklearn.neural_network import MLPRegressor

    reseau = MLPRegressor(hidden_layer_sizes=(4, 2, 4), max_iter=300, random_state=0)
    reseau.fit(x[y == 0], x[y == 0])
    modele = non_supervise.ModeleAnomalie("autoencodeur", None, reseau)
    scores = modele.scores_matrice(x)
    assert metriques.roc_auc(y, scores) > 0.99


def test_orientation_et_forme_des_scores(jeux, modeles):
    v = jeux[1]
    for nom, modele in modeles.items():
        scores = modele.scores(v)
        assert scores.shape == (len(v),)
        assert np.isfinite(scores).all()
        assert nom == modele.nom
    assert (modeles["autoencodeur"].scores(v) >= 0).all()


def test_les_methodes_font_mieux_que_le_hasard(jeux, modeles):
    v = jeux[1]
    y = v["is_fraud"].to_numpy()
    for nom in ("isolation_forest", "one_class_svm"):
        assert metriques.roc_auc(y, modeles[nom].scores(v)) > 0.55, nom


def test_ocsvm_entraine_sur_des_normales(jeux, modeles):
    assert modeles["one_class_svm"].parametres["taille"] <= 1500
    assert modeles["one_class_svm"].parametres["nu"] == pytest.approx(0.01)


def test_echantillonnage_reproductible():
    x, _ = nuage()
    a = non_supervise.echantillonner(x, 100, 3)
    b = non_supervise.echantillonner(x, 100, 3)
    c = non_supervise.echantillonner(x, 100, 4)
    pd.testing.assert_frame_equal(a, b)
    assert not a.index.equals(c.index)
    assert non_supervise.echantillonner(x, None, 0) is x


def test_epsilon_au_coude_sur_courbe_synthetique():
    plat = np.linspace(0.5, 1.0, 900)
    montee = np.linspace(1.0, 12.0, 100) ** 1.5
    distances = np.sort(np.r_[plat, montee])
    epsilon, position = non_supervise.epsilon_au_coude(distances)
    assert 0.9 < epsilon < 5.0
    assert 850 < position < 990
    constante = non_supervise.epsilon_au_coude(np.ones(10))
    assert constante == (1.0, 0)


def nuage_avec_anomalies_eparses(graine=0):
    rng = np.random.default_rng(graine)
    coeur = rng.normal(0, 1, (1500, 4))
    candidats = rng.uniform(-12, 12, (400, 4))
    eparses = candidats[np.linalg.norm(candidats, axis=1) > 6][:40]
    x = np.vstack([coeur, eparses])
    y = np.r_[np.zeros(len(coeur), dtype=int), np.ones(len(eparses), dtype=int)]
    return pd.DataFrame(x), y


def test_dbscan_isole_les_anomalies_eparses():
    x, y = nuage_avec_anomalies_eparses()
    parametres = non_supervise.parametres_dbscan(x, taille=len(x))
    assert parametres["min_points"] == 8
    bruit = non_supervise.bruit_dbscan(x, parametres["epsilon"], parametres["min_points"], n_jobs=1)
    assert bruit[y == 1].mean() > 0.8
    assert bruit[y == 0].mean() < 0.1


def test_dbscan_ne_traite_pas_un_petit_amas_dense_comme_du_bruit():
    x, y = nuage(n=1500, anomalies=60, dimension=4)
    bruit = non_supervise.bruit_dbscan(x, 1.5, 8, n_jobs=1)
    assert bruit[y == 1].mean() < 0.1


def test_distances_au_kieme_voisin_ordre_et_dimension():
    x, _ = nuage(n=300, anomalies=10, dimension=3)
    d = non_supervise.distances_au_kieme_voisin(x, 5)
    assert len(d) == len(x)
    assert (np.diff(d) >= 0).all()
    assert (d > 0).all()


def test_seuil_top_donne_le_bon_nombre_d_alertes():
    scores = np.random.default_rng(0).random(1000)
    seuil = non_supervise.seuil_top(scores, 0.01)
    assert int((scores >= seuil).sum()) == 10


def test_evaluer_anomalie():
    rng = np.random.default_rng(2)
    n = 5000
    y = (rng.random(n) < 0.02).astype(int)
    scores = rng.random(n) + 0.7 * y
    montants = rng.gamma(2.0, 50.0, n)
    seuil = non_supervise.seuil_top(scores, 0.01)
    r = non_supervise.evaluer_anomalie(y, scores, montants, 5.0, 100, seuil)
    assert r["alertes"] == 50
    assert r["cout_minimal"] <= r["cout"] + 1e-9
    assert r["cout_minimal"] <= r["cout_sans_modele"] + 1e-9
    assert 0 <= r["precision"] <= 1


def test_sauvegarde(jeux, modeles, tmp_path):
    import joblib

    v = jeux[1]
    for nom in ("isolation_forest", "one_class_svm", "autoencodeur"):
        chemin = non_supervise.sauvegarder(modeles[nom], tmp_path)
        assert chemin.name == f"{nom}.joblib"
        recharge = joblib.load(chemin)
        np.testing.assert_allclose(recharge.scores(v), modeles[nom].scores(v))
