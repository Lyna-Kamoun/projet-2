import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import composants as c

c.configurer("Accueil")

import streamlit as st

from src import application

c.entete(
    "Détection de fraude par carte bancaire",
    "Comparer des approches supervisées et non supervisées, fixer un seuil d'alerte selon un coût en montants, et expliquer chaque alerte.",
)

parametres = c.parametres()
evaluation = c.lire("evaluation_comparative.csv", source="resultats")
test = evaluation[(evaluation["methode"] == "xgboost") & (evaluation["jeu"] == "test")].iloc[0]
resume = c.lire("exploration_resume.csv").iloc[0]

st.markdown(
    f"""
Les équipes d'analyse ne peuvent examiner qu'un nombre limité de transactions par jour. Cette application classe les transactions
par niveau de risque, signale les plus suspectes dans un budget d'alertes donné, et explique pourquoi.

Les données sont celles du simulateur **Sparkov** (Kaggle) : **{application.nombre(resume['transactions'])} transactions** de
999 cartes, du 1er janvier 2019 au 31 décembre 2020, dont **{application.pourcentage(resume['taux_fraude'], 2)}** de fraudes.
Le découpage est chronologique : 60 % pour l'entraînement, 20 % pour la validation, 20 % pour le test.
"""
)

st.subheader("Résultat du modèle retenu (XGBoost) sur la période de test")
colonnes = st.columns(4)
colonnes[0].metric("PR-AUC", application.nombre(test["pr_auc"], 3))
colonnes[1].metric("Fraudes retrouvées", application.pourcentage(test["rappel"]))
colonnes[2].metric("Précision des alertes", application.pourcentage(test["precision"]))
colonnes[3].metric("Coût évité", application.pourcentage(test["economie"]), help="Part du montant des fraudes évitée, après déduction du coût d'analyse des alertes.")
st.caption(
    f"Seuil choisi sur la validation en minimisant le coût (montants des fraudes manquées + {application.nombre(parametres['cout_alerte'])} par alerte analysée), "
    "puis confirmé sur le test."
)

st.subheader("Parcours")
modules = [
    ("vues/donnees.py", "1. Données", "Taux de fraude, montants, évolution dans le temps, heure, catégorie, canal. Import d'un fichier."),
    ("vues/preparation.py", "2. Préparation", "Découpage chronologique, variables, stratégie contre le déséquilibre, réglage."),
    ("vues/modelisation.py", "3. Modélisation", "Comparaison des méthodes supervisées et non supervisées."),
    ("vues/evaluation_et_seuil.py", "4. Évaluation et seuil", "Courbes précision-rappel, curseurs de seuil et de budget, coût."),
    ("vues/explicabilite.py", "5. Explicabilité", "SHAP global et local, codes motifs, équité."),
    ("vues/api.py", "6. API", "Service FastAPI de scoring et d'explication, interrogé depuis l'application."),
]
for chemin, titre, description in modules:
    gauche, droite = st.columns([1, 3])
    gauche.page_link(chemin, label=titre)
    droite.write(description)

st.warning(
    "**Limites.** Les données sont synthétiques : les schémas de fraude sont ceux du simulateur (concentration nocturne, montants "
    "propres à chaque catégorie, plafond de 1 376 sur les fraudes). Le modèle s'appuie surtout sur le montant et la catégorie. "
    "Les performances ne se transposent pas telles quelles à des données réelles. L'unité monétaire n'est pas précisée par la source."
)
