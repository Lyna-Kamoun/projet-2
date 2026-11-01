from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

RACINE = Path(__file__).resolve().parent.parent
ENTREE = str(RACINE / "app" / "Accueil.py")
PAGES = {
    "vues/accueil.py": "Détection de fraude par carte bancaire",
    "vues/donnees.py": "1. Données",
    "vues/preparation.py": "2. Préparation et variables",
    "vues/modelisation.py": "3. Modélisation",
    "vues/evaluation_et_seuil.py": "4. Évaluation et seuil d'alerte",
    "vues/explicabilite.py": "5. Explicabilité",
    "vues/api.py": "6. API de scoring et d'explication",
}


def ouvrir(page="vues/accueil.py"):
    at = AppTest.from_file(ENTREE, default_timeout=180)
    at.run()
    if page != "vues/accueil.py":
        at.switch_page(page)
        at.run()
    return at


def valeurs(at, nom=None):
    resultat = {}
    for m in at.metric:
        resultat.setdefault(m.label, m.value)
    return resultat


@pytest.mark.parametrize("page,titre", list(PAGES.items()))
def test_chaque_page_s_affiche_sans_erreur(page, titre):
    at = ouvrir(page)
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == titre


def test_accueil_affiche_les_resultats_du_modele_retenu():
    at = ouvrir()
    m = valeurs(at, "accueil")
    assert m["PR-AUC"] == "0,900"
    assert m["Fraudes retrouvées"] == "95,1 %"
    assert m["Précision des alertes"] == "40,8 %"
    assert m["Coût évité"] == "96,1 %"


def test_donnees_indicateurs_globaux():
    at = ouvrir("vues/donnees.py")
    m = valeurs(at, "donnees")
    assert m["Transactions"] == "1 852 394"
    assert m["Fraudes"] == "9 651"
    assert m["Taux de fraude"] == "0,52 %"
    assert len(at.tabs) == 6


def test_evaluation_reproduit_les_resultats_de_l_evaluation():
    at = ouvrir("vues/evaluation_et_seuil.py")
    tableau = at.dataframe[0].value.set_index("Indicateur")
    assert tableau.loc["Alertes", "Validation"] == "3 763"
    assert tableau.loc["Alertes", "Test"] == "3 144"
    assert tableau.loc["Précision", "Validation"] == "48,3 %"
    assert tableau.loc["Rappel", "Test"] == "95,1 %"
    assert tableau.loc["Coût (fraudes manquées + analyses)", "Validation"] == "30 109"
    assert tableau.loc["Coût (fraudes manquées + analyses)", "Test"] == "28 908"
    assert tableau.loc["Économie", "Test"] == "96,1 %"


def test_evaluation_affiche_les_metriques_du_module_4():
    at = ouvrir("vues/evaluation_et_seuil.py")
    tableau = at.dataframe[0].value.set_index("Indicateur")
    assert tableau.loc["PR-AUC", "Validation"] == "0,917"
    assert tableau.loc["PR-AUC", "Test"] == "0,900"
    assert tableau.loc["ROC-AUC", "Test"] == "0,998"
    assert tableau.loc["Précision@k (k = 20 alertes par jour)", "Validation"] == "62,8 %"
    assert tableau.loc["Rappel@k (k = 20 alertes par jour)", "Validation"] == "92,4 %"
    assert tableau.loc["Précision@k (k = 20 alertes par jour)", "Test"] == "49,1 %"
    assert tableau.loc["Rappel@k (k = 20 alertes par jour)", "Test"] == "93,8 %"
    assert any("1,5 fois le nombre moyen de fraudes par jour" in c.value for c in at.caption)


def test_evaluation_mode_budget():
    at = ouvrir("vues/evaluation_et_seuil.py")
    at.radio(key="mode_seuil").set_value("Budget d'alertes par jour").run()
    at.slider(key="curseur_budget").set_value(10).run()
    assert not at.exception
    tableau = at.dataframe[0].value.set_index("Indicateur")
    assert tableau.loc["Alertes par jour", "Validation"] == "10,0"
    assert tableau.loc["Précision", "Validation"] == "95,6 %"
    assert tableau.loc["Rappel", "Validation"] == "70,3 %"
    assert tableau.loc["Précision@k (k = 10 alertes par jour)", "Validation"] == tableau.loc["Précision", "Validation"]


def test_evaluation_mode_curseur_et_autre_methode():
    at = ouvrir("vues/evaluation_et_seuil.py")
    at.radio(key="mode_seuil").set_value("Curseur de seuil").run()
    at.slider(key="curseur_seuil_xgboost").set_value(-1.0).run()
    assert not at.exception
    tableau = at.dataframe[0].value.set_index("Indicateur")
    assert tableau.loc["Alertes par jour", "Validation"] == "18,5"
    at.selectbox(key="methode_seuil").set_value("isolation_forest").run()
    assert not at.exception
    assert "Seuil du score d'anomalie" in [s.label for s in at.slider]


def test_evaluation_le_cout_d_analyse_modifie_le_seuil_optimal():
    at = ouvrir("vues/evaluation_et_seuil.py")
    base = at.dataframe[0].value.set_index("Indicateur").loc["Alertes", "Validation"]
    at.slider(key="cout_alerte").set_value(50.0).run()
    plus_cher = at.dataframe[0].value.set_index("Indicateur").loc["Alertes", "Validation"]
    assert base == "3 763"
    assert plus_cher == "2 321"


def test_preparation_bascule_sur_la_logistique():
    at = ouvrir("vues/preparation.py")
    at.radio(key="modele_desequilibre").set_value("logistique").run()
    assert not at.exception
    assert any("aucun rééchantillonnage" in s.value for s in at.success)


def test_modelisation_changement_de_periode():
    at = ouvrir("vues/modelisation.py")
    tableau = at.dataframe[0].value
    assert tableau["Méthode"].tolist()[0] == "XGBoost"
    assert "DBSCAN (échantillon)" in tableau["Méthode"].tolist()
    at.radio(key="jeu_modelisation").set_value("validation").run()
    assert not at.exception
    assert at.dataframe[0].value.set_index("Méthode").loc["XGBoost", "PR-AUC"] == "0,917"


def test_explicabilite_fraude_detectee_puis_fausse_alerte():
    at = ouvrir("vues/explicabilite.py")
    assert not at.exception
    m = valeurs(at, "explicabilite")
    assert m["Décision au seuil retenu"] == "Alerte"
    assert m["Réalité"] == "Fraude"
    at.radio(key="statut_explication").set_value("Fausse alerte").run()
    assert not at.exception
    m = valeurs(at, "explicabilite")
    assert m["Décision au seuil retenu"] == "Alerte"
    assert m["Réalité"] == "Transaction normale"
    at.radio(key="statut_explication").set_value("Fraude manquée").run()
    assert not at.exception
    m = valeurs(at, "explicabilite")
    assert m["Décision au seuil retenu"] == "Pas d'alerte"
    assert m["Réalité"] == "Fraude"


def test_explicabilite_transaction_personnalisee():
    at = ouvrir("vues/explicabilite.py")
    assert not at.exception
    at.number_input(key="montant_saisi").set_value(20.0).run()
    at.slider(key="heure_saisie").set_value(14).run()
    at.selectbox(key="categorie_saisie").set_value("travel").run()
    assert not at.exception
    probabilites = [m.value for m in at.metric if m.label == "Probabilité de fraude"]
    assert probabilites[-1] != "n.d."
    at.checkbox(key="sans_historique").check().run()
    assert not at.exception
