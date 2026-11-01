import json

import numpy as np
import pandas as pd
import pytest

from src import donnees, evaluation, modeles_supervises, ressources_application as r
from tests.test_desequilibre import jeu_fraude
from tests.test_evaluation import faux_jeu


@pytest.fixture(scope="module")
def contexte():
    df = jeu_fraude(n=24000)
    df["state"] = np.random.default_rng(0).choice(["NC", "WA", "ID"], len(df))
    e, v, t = donnees.decouper_chronologiquement(df)
    modele = modeles_supervises.entrainer_xgboost(e, {"n_estimators": 40, "max_depth": 3}, n_jobs=1)
    jeux = {"validation": faux_jeu(graine=1), "test": faux_jeu(graine=2, debut="2020-06-01")}
    return df, e, v, t, modele, jeux


def test_histogramme_conserve_les_effectifs(contexte):
    df = contexte[0]
    h = r.histogramme_montants(df)
    assert len(h) == r.CLASSES_HISTOGRAMME
    assert h["normales"].sum() == int((df["is_fraud"] == 0).sum())
    assert h["fraudes"].sum() == int(df["is_fraud"].sum())
    assert (h["borne_haute"] > h["borne_basse"]).all()
    assert h["borne_haute"].is_monotonic_increasing


def test_tables_exploration(contexte):
    t = r.tables_exploration(contexte[0])
    assert "exploration_par_heure.csv" in t and "exploration_par_canal.csv" in t
    assert t["exploration_par_heure.csv"]["transactions"].sum() == len(contexte[0])
    assert "heure" in t["exploration_par_heure.csv"].columns


def test_courbes_pr_limitees_en_points(contexte):
    c = r.courbes_pr(contexte[5], points=300)
    assert set(c["jeu"]) == {"validation", "test"}
    assert set(c["methode"]) == set(evaluation.METHODES)
    assert c.groupby(["jeu", "methode"]).size().max() <= 300


def test_scores_compacts(contexte):
    jeu = contexte[5]["validation"]
    s = r.scores_compacts(jeu)
    assert set(s) == {"montant", "fraude", *r.METHODES_CURSEUR}
    assert all(len(v) == len(jeu) for v in s.values())
    assert s["fraude"].dtype == np.int8
    assert s["xgboost"].dtype == np.float32
    np.testing.assert_allclose(s["xgboost"], jeu["xgboost"].to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(s["isolation_forest"], np.round(jeu["isolation_forest"].to_numpy(), 5), atol=1e-6)


def test_echantillon_contient_toutes_les_alertes_et_les_fraudes(contexte):
    _, _, v, t, modele, _ = contexte
    proba = modele.probabilites(t)
    seuil = 0.05
    e = r.echantillon_explication(t, proba, seuil, normales=200)
    assert (e["probabilite"] >= seuil).sum() == int((proba >= seuil).sum())
    assert e["is_fraud"].sum() == int(t["is_fraud"].sum())
    assert "cc_num" not in e.columns
    assert e["identifiant"].is_unique
    attendues = {"trans_date_trans_time", "is_fraud", "probabilite", "identifiant", *r.COLONNES_EXPLICATION}
    assert attendues == set(e.columns)


def test_echantillon_permet_de_reproduire_les_probabilites(contexte):
    _, _, v, t, modele, _ = contexte
    proba = modele.probabilites(t)
    e = r.echantillon_explication(t, proba, 0.05, normales=100)
    recalcul = modele.probabilites(e)
    np.testing.assert_allclose(recalcul, e["probabilite"].to_numpy(), atol=1e-5)


def test_ecriture_complete(contexte, tmp_path):
    df, e, v, t, modele, jeux = contexte
    seuils = {m: 0.5 for m in r.METHODES_CURSEUR}
    fichiers = r.ecrire(tmp_path, df, e, v, t, jeux, modele, modele.probabilites(t), seuils, 5.0, 20.0)
    attendus = {
        "exploration_resume.csv", "exploration_par_mois.csv", "exploration_par_heure.csv",
        "exploration_par_jour.csv", "exploration_par_categorie.csv", "exploration_par_canal.csv",
        "exploration_par_etat.csv", "exploration_montants_par_classe.csv",
        "exploration_histogramme_montants.csv", "decoupage.csv", "courbes_precision_rappel.csv",
        "scores_validation.npz", "scores_test.npz", "echantillon_explication.parquet", "parametres.json",
    }
    assert attendus == set(fichiers)
    contenu = json.loads((tmp_path / "parametres.json").read_text(encoding="utf-8"))
    assert contenu["cout_alerte"] == 5.0
    assert contenu["beta"] == pytest.approx(modele.beta)
    assert set(contenu["jeux"]) == {"validation", "test"}
    assert contenu["jeux"]["test"]["jours"] > 0
    assert contenu["jeux"]["test"]["cout_sans_modele"] == pytest.approx(
        float((t["amt"] * t["is_fraud"]).sum())
    )
    charge = np.load(tmp_path / "scores_validation.npz")
    assert set(charge.files) == {"montant", "fraude", *r.METHODES_CURSEUR}
    assert pd.read_parquet(tmp_path / "echantillon_explication.parquet").shape[0] > 0
