import numpy as np
import pytest

from src import donnees, metriques, modeles_supervises, preparation, reglage
from tests.test_desequilibre import jeu_fraude


@pytest.fixture(scope="module")
def jeux():
    df = jeu_fraude(n=24000)
    return donnees.decouper_chronologiquement(df)


@pytest.fixture(scope="module")
def modeles(jeux):
    e, v, t = jeux
    petit = {"n_estimators": 60, "max_depth": 4}
    return {
        "xgboost": modeles_supervises.entrainer_xgboost(e, petit, n_jobs=1),
        "logistique": modeles_supervises.entrainer_logistique(e, 1.0),
    }


def test_beta_du_sous_echantillonnage(jeux, modeles):
    e = jeux[0]
    fraudes = int(e["is_fraud"].sum())
    normales = len(e) - fraudes
    attendu = reglage.RATIO_SOUS_ECHANTILLONNAGE * fraudes / normales
    assert modeles["xgboost"].beta == pytest.approx(attendu)
    assert modeles["logistique"].beta == 1.0


def test_scores_et_probabilites(jeux, modeles):
    v = jeux[1]
    for modele in modeles.values():
        scores = modele.scores(v)
        probas = modele.probabilites(v)
        assert len(scores) == len(v)
        assert ((scores >= 0) & (scores <= 1)).all()
        assert ((probas >= 0) & (probas <= 1)).all()
        assert (np.argsort(scores, kind="stable") == np.argsort(probas, kind="stable")).mean() > 0.99
    xgb = modeles["xgboost"]
    assert xgb.probabilites(v).mean() < xgb.scores(v).mean()


def test_probabilites_corrigees_proches_du_taux_reel(jeux, modeles):
    v = jeux[1]
    reel = v["is_fraud"].mean()
    corrige = modeles["xgboost"].probabilites(v).mean()
    brut = modeles["xgboost"].scores(v).mean()
    assert abs(corrige - reel) < abs(brut - reel)
    assert corrige == pytest.approx(reel, rel=0.5)


def test_les_modeles_detectent_mieux_que_le_hasard(jeux, modeles):
    v = jeux[1]
    y = preparation.cible(v)
    for modele in modeles.values():
        assert metriques.pr_auc(y, modele.scores(v)) > 3 * y.mean()


def test_sauvegarde_et_rechargement(jeux, modeles, tmp_path):
    v = jeux[1]
    for nom, modele in modeles.items():
        chemin = modeles_supervises.sauvegarder(modele, tmp_path)
        assert chemin.exists()
        recharge = modeles_supervises.charger(nom, tmp_path)
        assert recharge.nom == nom
        assert recharge.beta == modele.beta
        np.testing.assert_allclose(recharge.scores(v), modele.scores(v))
        assert list(recharge.matrice(v).columns) == list(modele.matrice(v).columns)


def test_colonnes_attendues(jeux, modeles):
    v = jeux[1]
    colonnes_xgb = list(modeles["xgboost"].matrice(v).columns)
    colonnes_log = list(modeles["logistique"].matrice(v).columns)
    assert "amt" in colonnes_xgb and "log_montant" not in colonnes_xgb
    assert "log_montant" in colonnes_log and "amt" not in colonnes_log


def test_parametres_charges(tmp_path):
    chemin = tmp_path / "p.json"
    chemin.write_text('{"xgboost": {"max_depth": 6}, "logistique": {"C": 1.0}}', encoding="utf-8")
    p = modeles_supervises.charger_parametres(chemin)
    assert p["logistique"]["C"] == 1.0
