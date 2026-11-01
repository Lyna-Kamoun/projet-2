import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src import application, donnees, explicabilite, graphiques, metriques, modeles_supervises
from tests.test_desequilibre import jeu_fraude


def test_formatage_francais():
    assert application.nombre(1234567) == "1 234 567"
    assert application.nombre(0.5, 2) == "0,50"
    assert application.nombre(float("nan")) == "n.d."
    assert application.pourcentage(0.9512) == "95,1 %"
    assert application.pourcentage(0.00364, 2) == "0,36 %"
    assert application.pourcentage(None) == "n.d."


def jeu_scores(n=5000, graine=0):
    rng = np.random.default_rng(graine)
    fraude = (rng.random(n) < 0.03).astype("int8")
    scores = np.clip(rng.random(n) * 0.5 + 0.5 * fraude * rng.random(n), 0, 1)
    montant = rng.gamma(2.0, 60.0, n) + 1.0
    return scores, fraude, montant


def test_indicateurs_coherents_avec_les_metriques():
    scores, fraude, montant = jeu_scores()
    ind = application.indicateurs(scores, fraude, montant, 0.4, 5.0, 10.0)
    attendu = metriques.indicateurs_au_seuil(fraude, scores, 0.4)
    for cle in ("precision", "rappel", "f2", "mcc", "alertes"):
        assert ind[cle] == pytest.approx(attendu[cle])
    assert ind["alertes_par_jour"] == pytest.approx(ind["alertes"] / 10.0)
    assert ind["cout"] == pytest.approx(metriques.cout_au_seuil(fraude, scores, montant, 0.4, 5.0))
    assert ind["cout_sans_modele"] == pytest.approx(float(montant[fraude == 1].sum()))
    assert ind["economie"] == pytest.approx(1.0 - ind["cout"] / ind["cout_sans_modele"])


def test_indicateurs_de_classement():
    scores, fraude, montant = jeu_scores()
    ind = application.indicateurs_classement(scores, fraude, 20.0, 5.0)
    assert ind["k"] == 100
    assert ind["pr_auc"] == pytest.approx(metriques.pr_auc(fraude, scores))
    assert ind["roc_auc"] == pytest.approx(metriques.roc_auc(fraude, scores))
    precision, rappel = metriques.precision_rappel_au_rang(fraude, scores, 100)
    assert ind["precision_a_k"] == pytest.approx(precision)
    assert ind["rappel_a_k"] == pytest.approx(rappel)
    assert application.indicateurs_classement(scores, fraude, 10000.0, 5.0)["k"] == len(scores)


def test_seuil_budget_donne_le_bon_nombre_d_alertes():
    scores, fraude, montant = jeu_scores()
    seuil = application.seuil_budget(scores, 20.0, 5.0)
    assert int((scores >= seuil).sum()) == 100
    assert application.seuil_budget(scores, 10000.0, 5.0) == pytest.approx(float(scores.min()))
    assert application.seuil_budget(scores, 0.0, 5.0) == pytest.approx(float(scores.max()))


def test_seuil_optimal_minimise_le_cout():
    scores, fraude, montant = jeu_scores()
    optimum = application.seuil_optimal(scores, fraude, montant, 5.0)
    for seuil in np.linspace(0.05, 0.95, 19):
        assert application.indicateurs(scores, fraude, montant, seuil, 5.0, 1.0)["cout"] >= optimum["cout"] - 1e-6


def test_courbe_de_cout_bornee_et_reduite():
    scores, fraude, montant = jeu_scores()
    courbe = application.courbe_cout(scores, fraude, montant, 5.0, jours=5.0, maximum_par_jour=100.0, points=50)
    assert len(courbe) <= 50
    assert courbe["alertes_par_jour"].max() <= 100.0
    assert courbe["alertes_par_jour"].is_monotonic_increasing
    assert courbe["cout"].iloc[0] == pytest.approx(float(montant[fraude == 1].sum()))


def test_curseur_logarithmique():
    mini, maxi, valeur = application.bornes_curseur_log(0.0272)
    assert (mini, maxi) == (-4.0, 0.0)
    assert 10.0**valeur == pytest.approx(0.0272, rel=0.02)


@pytest.fixture(scope="module")
def modele_synthetique():
    df = jeu_fraude(n=24000)
    e, v, t = donnees.decouper_chronologiquement(df)
    return modeles_supervises.entrainer_xgboost(e, {"n_estimators": 40, "max_depth": 3}, n_jobs=1)


def test_transaction_personnalisee_est_explicable(modele_synthetique):
    ligne = application.transaction_personnalisee("2020-10-15 23:00:00", 950.0, "shopping_net", 0, 3, 1.0, 43200.0, 0.0, 78.0)
    assert ligne.shape[0] == 1
    assert ligne.loc[0, "nuit"] == 1
    assert ligne.loc[0, "heure_sin"] == pytest.approx(np.sin(2 * np.pi * 23 / 24))
    proba = modele_synthetique.probabilites(ligne)
    assert 0 <= proba[0] <= 1
    tableau = explicabilite.cascade(modele_synthetique, ligne, 0)
    assert len(tableau) == 10
    sans = application.transaction_personnalisee("2020-10-15 12:00:00", 40.0, "travel")
    assert sans.loc[0, "nuit"] == 0
    assert np.isnan(sans.loc[0, "ratio_moy_30j"])
    assert 0 <= modele_synthetique.probabilites(sans)[0] <= 1


def csv_synthetique(n=600, graine=0, retirer=None):
    rng = np.random.default_rng(graine)
    secondes = np.sort(rng.integers(0, 20 * 86400, n))
    dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(secondes, unit="s")
    df = pd.DataFrame(
        {
            "Unnamed: 0": np.arange(n),
            "trans_date_trans_time": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "cc_num": rng.integers(1, 15, n),
            "category": rng.choice(application.CATEGORIES, n),
            "amt": rng.gamma(2.0, 50.0, n).round(2) + 1.0,
            "lat": rng.uniform(30, 45, n),
            "long": rng.uniform(-120, -75, n),
            "merch_lat": rng.uniform(30, 45, n),
            "merch_long": rng.uniform(-120, -75, n),
            "dob": "1980-05-01",
            "gender": rng.choice(["F", "M"], n),
            "is_fraud": (rng.random(n) < 0.03).astype(int),
        }
    )
    if retirer:
        df = df.drop(columns=retirer)
    return io.StringIO(df.to_csv(index=False))


def test_import_valide(modele_synthetique):
    resultat = application.analyser_import(csv_synthetique(), modele_synthetique)
    assert len(resultat) == 600
    assert resultat["probabilite"].between(0, 1).all()
    assert resultat["trans_date_trans_time"].is_monotonic_increasing
    for colonne in ("nb_tx_24h", "ratio_moy_30j", "distance_km"):
        assert colonne in resultat.columns


def test_import_sans_modele_et_sans_cible(modele_synthetique):
    resultat = application.analyser_import(csv_synthetique(retirer=["is_fraud"]))
    assert "probabilite" not in resultat.columns
    assert "is_fraud" not in resultat.columns


def test_import_refuse_les_colonnes_manquantes():
    with pytest.raises(ValueError, match="Colonnes manquantes : dob"):
        application.analyser_import(csv_synthetique(retirer=["dob"]))


def test_import_refuse_les_fichiers_trop_gros(monkeypatch):
    monkeypatch.setattr(application, "LIGNES_MAXIMUM_IMPORT", 100)
    with pytest.raises(ValueError, match="limite"):
        application.analyser_import(csv_synthetique())


def test_import_refuse_un_fichier_vide():
    vide = io.StringIO(",".join(application.COLONNES_IMPORT) + "\n")
    with pytest.raises(ValueError, match="aucune transaction"):
        application.analyser_import(vide)


def test_graphiques_de_base():
    mois = pd.DataFrame({"mois": ["2019-01", "2019-02", "2019-03"], "taux_fraude": [0.01, 0.008, 0.006], "transactions": [100, 120, 90]})
    assert isinstance(graphiques.ligne(mois, "mois", "taux_fraude", "t", "a"), go.Figure)
    assert graphiques.ligne(mois, "mois", "taux_fraude", "t", "a").layout.xaxis.type == "category"
    barres = graphiques.barres(mois, "mois", "taux_fraude", "t", "a")
    assert barres.layout.xaxis.type == "category"
    horizontal = graphiques.barres(mois, "mois", "taux_fraude", "t", "a", horizontal=True)
    assert horizontal.data[0].orientation == "h"
    histogramme = pd.DataFrame({"borne_basse": [1.0, 10.0, 100.0], "borne_haute": [10.0, 100.0, 1000.0], "normales": [50, 40, 10], "fraudes": [1, 2, 7]})
    figure = graphiques.histogramme_montants(histogramme)
    assert len(figure.data) == 2
    assert float(sum(figure.data[1].y)) == pytest.approx(1.0)


def test_graphiques_de_modeles():
    courbes = pd.DataFrame(
        {
            "jeu": ["test"] * 6,
            "methode": ["xgboost"] * 3 + ["lof"] * 3,
            "rappel": [0.0, 0.5, 1.0] * 2,
            "precision": [1.0, 0.8, 0.1, 1.0, 0.1, 0.01],
        }
    )
    figure = graphiques.courbes_precision_rappel(courbes, "test", ["xgboost", "lof"], point=(0.5, 0.8), methode_point="xgboost", taux=0.004)
    assert len(figure.data) == 4
    assert figure.layout.xaxis.range == (0, 1)
    sans_point = graphiques.courbes_precision_rappel(courbes, "test", ["xgboost"])
    assert len(sans_point.data) == 1
    cout = pd.DataFrame({"alertes_par_jour": [0.0, 10.0, 50.0], "cout": [1000.0, 100.0, 300.0]})
    assert len(graphiques.courbe_cout(cout, point=(10.0, 100.0)).data) == 2
    importance = pd.Series({"amt": 3.0, "heure": 1.0})
    assert graphiques.importance(importance, "t", "x").data[0].orientation == "h"
    resume = pd.DataFrame(
        {"modele": ["xgboost"] * 2, "libelle": ["A", "B"], "pr_auc": [0.9, 0.91], "pr_auc_ecart_type": [0.001, 0.002], "tirages": [3, 3]}
    )
    assert len(graphiques.comparaison_strategies(resume, "xgboost").data) == 1
    tableau = pd.DataFrame({"methode": ["xgboost", "lof"], "pr_auc": [0.9, 0.03]})
    assert len(graphiques.comparaison_methodes(tableau, "pr_auc", "t", "x").data) == 1


def test_graphique_cascade(modele_synthetique):
    ligne = application.transaction_personnalisee("2020-10-15 23:00:00", 950.0, "shopping_net", 0, 3, 1.0, 43200.0, 0.0, 78.0)
    tableau = explicabilite.cascade(modele_synthetique, ligne, 0)
    figure = graphiques.cascade(tableau)
    assert len(figure.data) == 1
    assert len(figure.data[0].y) == 9
    assert len(figure.layout.annotations) == 2
