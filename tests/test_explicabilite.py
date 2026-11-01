import numpy as np
import pandas as pd
import pytest

from src import explicabilite, modeles_supervises, donnees
from tests.test_desequilibre import jeu_fraude


@pytest.fixture(scope="module")
def contexte():
    df = jeu_fraude(n=24000)
    e, v, t = donnees.decouper_chronologiquement(df)
    modele = modeles_supervises.entrainer_xgboost(
        e, {"n_estimators": 60, "max_depth": 4}, n_jobs=1
    )
    echantillon = v.iloc[:1500]
    resultat = explicabilite.calculer_shap(modele, echantillon)
    return modele, v, echantillon, resultat


def test_groupes_couvrent_toutes_les_colonnes_une_fois(contexte):
    modele, v, _, resultat = contexte
    groupes = explicabilite.groupes_colonnes(resultat.valeurs.columns)
    toutes = [c for membres in groupes.values() for c in membres]
    assert sorted(toutes) == sorted(resultat.valeurs.columns)
    assert len(toutes) == len(set(toutes))
    assert set(groupes["heure"]) == {"heure_sin", "heure_cos", "nuit"}
    assert all(c.startswith("category_") for c in groupes["category"])
    assert "amt" in groupes and "distance_km" in groupes


def test_additivite_shap(contexte):
    modele, v, echantillon, resultat = contexte
    reconstruit = resultat.base + resultat.valeurs.sum(axis=1).to_numpy()
    np.testing.assert_allclose(reconstruit, explicabilite.marge(modele, echantillon), atol=1e-3)


def test_regroupement_conserve_les_sommes(contexte):
    _, _, _, resultat = contexte
    groupees = resultat.groupees()
    np.testing.assert_allclose(
        groupees.sum(axis=1).to_numpy(), resultat.valeurs.sum(axis=1).to_numpy(), atol=1e-5
    )
    assert groupees.shape[1] < resultat.valeurs.shape[1]


def test_importance_shap_parts(contexte):
    _, _, _, resultat = contexte
    imp = explicabilite.importance_shap(resultat.groupees())
    assert imp["part"].sum() == pytest.approx(1.0)
    assert imp["importance"].is_monotonic_decreasing
    assert (imp["importance"] >= 0).all()


def test_permutation_distingue_les_variables_utiles(contexte):
    modele, v, _, _ = contexte
    imp = explicabilite.importance_permutation(modele, v, repetitions=2, graine=0)
    assert set(imp.index) == set(explicabilite.COLONNES_PERMUTATION)
    assert imp["perte_pr_auc"].is_monotonic_decreasing
    assert imp.loc["heure", "perte_pr_auc"] > 0.005
    assert "heure" in list(imp.index[:3])
    assert imp.loc["distance_km", "perte_pr_auc"] < imp.loc["heure", "perte_pr_auc"]
    again = explicabilite.importance_permutation(modele, v, repetitions=2, graine=0)
    pd.testing.assert_frame_equal(imp, again)


def test_permutation_ne_modifie_pas_le_jeu(contexte):
    modele, v, _, _ = contexte
    copie = v.copy()
    explicabilite.importance_permutation(modele, v, repetitions=1)
    pd.testing.assert_frame_equal(v, copie)


def test_cascade_somme_et_ordre(contexte):
    modele, v, echantillon, resultat = contexte
    c = explicabilite.cascade(modele, echantillon, 3)
    assert c["contribution"].abs().is_monotonic_decreasing
    assert c.attrs["base"] + c["contribution"].sum() == pytest.approx(
        explicabilite.marge(modele, echantillon.iloc[[3]])[0], abs=1e-3
    )
    assert 0 <= c.attrs["probabilite_corrigee"] <= 1
    assert c.attrs["probabilite_corrigee"] == pytest.approx(
        modele.probabilites(echantillon.iloc[[3]])[0], abs=1e-4
    )


def test_codes_motifs(contexte):
    modele, v, echantillon, resultat = contexte
    fraudes = np.flatnonzero(echantillon["is_fraud"].to_numpy() == 1)
    position = int(fraudes[0]) if len(fraudes) else 0
    motifs = explicabilite.codes_motifs(modele, echantillon, position, k=4)
    assert 1 <= len(motifs) <= 4
    contributions = [m["contribution"] for m in motifs]
    assert all(c > 0 for c in contributions)
    assert contributions == sorted(contributions, reverse=True)
    assert all(m["code"] in explicabilite.CODES.values() for m in motifs)
    assert all(len(m["libelle"]) > 5 for m in motifs)
    plein = explicabilite.cascade(modele, echantillon, position)
    assert motifs[0]["code"] == plein[plein["contribution"] > 0].iloc[0]["code"]


def test_libelles_historique_manquant():
    ligne = pd.Series(
        {
            "trans_date_trans_time": pd.Timestamp("2020-05-04 23:15:00"),
            "amt": 812.5,
            "nb_tx_1h": 0,
            "nb_tx_24h": 2,
            "ratio_moy_30j": np.nan,
            "delai_precedent": np.nan,
            "z_montant_30j": np.nan,
            "distance_km": 77.4,
            "category": "shopping_net",
        }
    )
    assert explicabilite.libelle("heure", ligne) == "Heure de la transaction : 23 h"
    assert "lundi" in explicabilite.libelle("jour", ligne)
    assert explicabilite.libelle("amt", ligne) == "Montant : 812,50"
    assert "Première" in explicabilite.libelle("delai_precedent", ligne)
    assert "Aucun historique" in explicabilite.libelle("ratio_moy_30j", ligne)
    assert "non calculable" in explicabilite.libelle("z_montant_30j", ligne)
    assert "shopping_net" in explicabilite.libelle("category", ligne)


def test_duree_lisible():
    assert explicabilite.duree_lisible(1800) == "30 min"
    assert explicabilite.duree_lisible(7200) == "2,0 h"
    assert explicabilite.duree_lisible(5 * 86400) == "5,0 jours"


def test_figures_explicabilite(contexte, tmp_path):
    from src import figures

    modele, v, echantillon, resultat = contexte
    imp = explicabilite.importance_shap(resultat.groupees())
    perm = explicabilite.importance_permutation(modele, v, repetitions=1)
    fraudes = np.flatnonzero(echantillon["is_fraud"].to_numpy() == 1)
    tableau = explicabilite.cascade(modele, echantillon, int(fraudes[0]))
    chemins = [
        figures.figure_importance_shap(imp, tmp_path / "shap.png"),
        figures.figure_importance_permutation(perm, tmp_path / "perm.png"),
        figures.figure_nuage_abeilles(resultat, tmp_path / "abeilles.png"),
        figures.figure_cascade(tableau, tmp_path / "cascade.png", "Alerte"),
    ]
    for chemin in chemins:
        assert chemin.exists() and chemin.stat().st_size > 15000


def test_formatage_des_nombres():
    assert explicabilite.nombre(3.456) == "3,5"
    assert explicabilite.nombre(3.456, 2) == "3,46"
    assert explicabilite.nombre(2.0, 1, True) == "+2,0"
    assert explicabilite.nombre(-2.04, 1, True) == "-2,0"
    assert explicabilite.nombre(690.4, 0) == "690"


def test_choix_des_exemples():
    from src import lancer_explicabilite as lanceur

    test = pd.DataFrame(
        {
            "is_fraud": [1, 1, 1, 0, 0, 1, 1],
            "amt": [10.0, 20.0, 30.0, 40.0, 50.0, 900.0, 100.0],
        }
    )
    probabilites = np.array([0.9, 0.5, 0.7, 0.8, 0.4, 0.01, 0.02])
    exemples = lanceur.choisir_exemples(test, probabilites, 0.45)
    assert exemples["vrai_positif"] == 2
    assert exemples["faux_positif"] == 3
    assert exemples["fraude_manquee"] == 5


def test_figure_dependance(contexte, tmp_path):
    from src import figures

    _, _, _, resultat = contexte
    chemin = figures.figure_dependance(resultat, "amt", tmp_path / "dep.png")
    assert chemin.exists() and chemin.stat().st_size > 15000
