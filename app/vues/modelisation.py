import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import composants as c

c.configurer("Modélisation")

import pandas as pd
import streamlit as st

from src import application, graphiques

c.entete(
    "3. Modélisation",
    "Deux modèles supervisés et cinq méthodes non supervisées, comparés sur les mêmes périodes et au même seuil de coût.",
)

jeu = st.radio("Période", ["validation", "test"], index=1, format_func=str.capitalize, horizontal=True, key="jeu_modelisation")
evaluation = c.lire("evaluation_comparative.csv", source="resultats")
sous = evaluation[evaluation["jeu"] == jeu].copy()
non_supervise = c.lire("comparaison_non_supervise.csv", source="resultats")
dbscan = non_supervise[(non_supervise["methode"] == "dbscan") & (non_supervise["jeu"] == jeu)]

tableau = pd.DataFrame(
    {
        "Méthode": sous["methode"].map(graphiques.LIBELLES),
        "PR-AUC": sous["pr_auc"].map(lambda v: application.nombre(v, 3)),
        "ROC-AUC": sous["roc_auc"].map(lambda v: application.nombre(v, 3)),
        "Préc.@k": sous["precision_a_k"].map(lambda v: application.nombre(v, 3)),
        "Rapp.@k": sous["rappel_a_k"].map(lambda v: application.nombre(v, 3)),
        "Alertes": sous["alertes"].map(application.nombre),
        "Précision": sous["precision"].map(lambda v: application.nombre(v, 3)),
        "Rappel": sous["rappel"].map(lambda v: application.nombre(v, 3)),
        "F2": sous["f2"].map(lambda v: application.nombre(v, 3)),
        "MCC": sous["mcc"].map(lambda v: application.nombre(v, 3)),
        "Coût": sous["cout"].map(application.nombre),
        "Économie": sous["economie"].map(application.pourcentage),
    }
)
if not dbscan.empty:
    ligne = dbscan.iloc[0]
    tableau.loc[len(tableau)] = [
        "DBSCAN (échantillon)",
        "-",
        "-",
        "-",
        "-",
        application.nombre(ligne["alertes"]),
        application.nombre(ligne["precision"], 3),
        application.nombre(ligne["rappel"], 3),
        application.nombre(ligne["f2"], 3),
        application.nombre(ligne["mcc"], 3),
        "-",
        "-",
    ]
st.dataframe(tableau, hide_index=True)
st.caption(
    "Supervisées : XGBoost et régression logistique. Non supervisées : Isolation Forest, One-Class SVM, autoencodeur, LOF et DBSCAN. Le seuil de chaque méthode est celui qui minimise le coût sur la validation (montants des fraudes manquées + 5 par alerte), "
    "appliqué tel quel à la période affichée. Précision@k et rappel@k : budget de 20 alertes par jour, soit environ 1,5 fois le nombre moyen de fraudes par jour en validation (justification en page 4). "
    "DBSCAN n'a pas de score : il signale les transactions classées en bruit (échantillon de 40 000 transactions) et n'a donc ni PR-AUC ni seuil de coût."
)

gauche, droite = st.columns(2)
gauche.plotly_chart(graphiques.comparaison_methodes(sous, "pr_auc", "PR-AUC", "PR-AUC"))
droite.plotly_chart(graphiques.comparaison_methodes(sous, "economie", "Coût évité par rapport à l'absence de modèle", "Part du coût évité", ".1%"))

st.subheader("Lecture")
st.markdown(
    "- **Le gradient boosting domine** : conformément au chapitre 7, avec des fraudes étiquetées en nombre suffisant, il est le point de départ conseillé.\n"
    "- **La régression logistique** reste la référence lisible, mais elle ne représente pas l'interaction entre le montant et la catégorie.\n"
    "- **Isolation Forest** est la meilleure méthode non supervisée. **LOF** est la plus faible et la plus lente.\n"
    "- **La ROC-AUC est trompeuse** : l'Isolation Forest atteint 0,89 de ROC-AUC avec une PR-AUC de 0,08 sur le test, pour un taux de fraude de 0,36 %.\n"
    "- **« Inhabituel » ne veut pas dire « frauduleux »** : les méthodes non supervisées produisent des dizaines de fausses alertes par fraude retrouvée."
)

st.subheader("Que gagne-t-on à ajouter une méthode non supervisée ?")
complementarite = c.lire("complementarite.csv", source="resultats")
sous_complement = complementarite[complementarite["jeu"] == jeu]
tableau_complement = pd.DataFrame(
    {
        "Méthode ajoutée": sous_complement["appoint"].map(graphiques.LIBELLES),
        "Fraudes manquées par XGBoost": sous_complement["manquees_par_base"].map(application.nombre),
        "Récupérées par la méthode ajoutée": sous_complement["recuperees_par_appoint"].map(application.nombre),
        "Alertes au total": sous_complement["alertes_union"].map(application.nombre),
        "Rappel combiné": sous_complement["rappel_union"].map(lambda v: application.nombre(v, 3)),
        "Rappel de XGBoost au même budget": sous_complement["rappel_base_au_meme_budget"].map(lambda v: application.nombre(v, 3)),
    }
)
st.dataframe(tableau_complement, hide_index=True)
st.caption(
    "Au budget de 20 alertes par jour, ajouter les alertes d'une méthode non supervisée récupère très peu de fraudes manquées, et fait moins bien que donner le même budget supplémentaire à XGBoost. "
    "Leur place est le cas sans étiquettes, ou la surveillance de la dérive."
)
