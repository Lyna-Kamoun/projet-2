import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import composants as c

c.configurer("Évaluation et seuil")

import numpy as np
import pandas as pd
import streamlit as st

from src import application, graphiques

c.entete(
    "4. Évaluation et seuil d'alerte",
    "Le seuil est choisi sur la validation en minimisant un coût en montants, puis appliqué tel quel au test.",
)

parametres = c.parametres()
jours = {jeu: parametres["jeux"][jeu]["jours"] for jeu in ("validation", "test")}
scores = {jeu: c.scores(jeu) for jeu in ("validation", "test")}


@st.cache_data(show_spinner=False)
def courbe_de_cout(jeu, methode, cout_alerte):
    donnees = scores[jeu]
    return application.courbe_cout(
        donnees[methode], donnees["fraude"], donnees["montant"], cout_alerte, jours[jeu]
    )


@st.cache_data(show_spinner=False)
def classement(jeu, methode, par_jour):
    donnees = scores[jeu]
    return application.indicateurs_classement(donnees[methode], donnees["fraude"], par_jour, jours[jeu])


reglages = st.columns([2, 2, 3])
methode = reglages[0].selectbox(
    "Méthode",
    ["xgboost", "logistique", "isolation_forest"],
    format_func=lambda m: graphiques.LIBELLES[m],
    key="methode_seuil",
)
cout_alerte = reglages[1].slider(
    "Coût d'analyse par alerte",
    1.0,
    50.0,
    float(parametres["cout_alerte"]),
    step=1.0,
    key="cout_alerte",
    help="Coût de l'examen d'une alerte par un analyste, dans la même unité que les montants. Une fraude manquée coûte son montant.",
)
mode = reglages[2].radio(
    "Réglage du seuil",
    ["Curseur de seuil", "Budget d'alertes par jour", "Seuil optimal de coût"],
    index=2,
    key="mode_seuil",
)

validation = scores["validation"]
defaut = parametres["seuils"][methode]
budget_jour = int(parametres["alertes_par_jour"])
if mode == "Curseur de seuil":
    if methode == "isolation_forest":
        minimum, maximum = float(validation[methode].min()), float(validation[methode].max())
        seuil = st.slider(
            "Seuil du score d'anomalie",
            minimum,
            maximum,
            float(np.clip(defaut, minimum, maximum)),
            step=(maximum - minimum) / 500.0,
            format="%.3f",
            key=f"curseur_seuil_{methode}",
        )
    else:
        position = st.slider(
            "Seuil de probabilité de fraude (échelle logarithmique)",
            -4.0,
            0.0,
            round(float(np.log10(defaut)), 2),
            step=0.01,
            format="%.2f",
            key=f"curseur_seuil_{methode}",
            help="Valeur affichée : puissance de 10. Par exemple -2 correspond à un seuil de 1 %.",
        )
        seuil = float(10.0**position)
        st.caption(f"Seuil retenu : {application.pourcentage(seuil, 2)}")
elif mode == "Budget d'alertes par jour":
    budget_jour = st.slider("Alertes par jour (calibré sur la validation)", 1, 200, budget_jour, key="curseur_budget")
    seuil = application.seuil_budget(validation[methode], budget_jour, jours["validation"])
    st.caption(f"Seuil correspondant sur la validation : {application.nombre(seuil, 4)}")
else:
    optimum = application.seuil_optimal(validation[methode], validation["fraude"], validation["montant"], cout_alerte)
    seuil = optimum["seuil"]
    st.caption(f"Seuil qui minimise le coût sur la validation : {application.nombre(seuil, 4)}")

resultats = {}
for jeu in ("validation", "test"):
    donnees = scores[jeu]
    resultats[jeu] = application.indicateurs(
        donnees[methode], donnees["fraude"], donnees["montant"], seuil, cout_alerte, jours[jeu]
    )
    resultats[jeu].update(classement(jeu, methode, budget_jour))

test = resultats["test"]
principaux = st.columns(4)
principaux[0].metric("Alertes par jour (test)", application.nombre(test["alertes_par_jour"], 1), help=f"{application.nombre(test['alertes'])} alertes au total sur la période de test.")
principaux[1].metric("Fraudes retrouvées (test)", application.pourcentage(test["rappel"]))
principaux[2].metric("Précision des alertes (test)", application.pourcentage(test["precision"]))
principaux[3].metric("Coût évité (test)", application.pourcentage(test["economie"]))

lignes = [
    ("PR-AUC", lambda r: application.nombre(r["pr_auc"], 3)),
    ("ROC-AUC", lambda r: application.nombre(r["roc_auc"], 3)),
    (f"Précision@k (k = {budget_jour} alertes par jour)", lambda r: application.pourcentage(r["precision_a_k"])),
    (f"Rappel@k (k = {budget_jour} alertes par jour)", lambda r: application.pourcentage(r["rappel_a_k"])),
    ("Alertes", lambda r: application.nombre(r["alertes"])),
    ("Alertes par jour", lambda r: application.nombre(r["alertes_par_jour"], 1)),
    ("Précision", lambda r: application.pourcentage(r["precision"])),
    ("Rappel", lambda r: application.pourcentage(r["rappel"])),
    ("F2", lambda r: application.nombre(r["f2"], 3)),
    ("MCC", lambda r: application.nombre(r["mcc"], 3)),
    ("Coût (fraudes manquées + analyses)", lambda r: application.nombre(r["cout"])),
    ("Coût sans modèle", lambda r: application.nombre(r["cout_sans_modele"])),
    ("Économie", lambda r: application.pourcentage(r["economie"])),
]
comparaison = pd.DataFrame(
    {
        "Indicateur": [nom for nom, _ in lignes],
        "Validation": [formatter(resultats["validation"]) for _, formatter in lignes],
        "Test": [formatter(resultats["test"]) for _, formatter in lignes],
    }
)
st.dataframe(comparaison, hide_index=True, height="content")
fraudes_par_jour = parametres["jeux"]["validation"]["fraudes"] / jours["validation"]
st.caption(
    f"PR-AUC et ROC-AUC ne dépendent pas du seuil. Précision@k et rappel@k portent sur les k transactions les mieux classées, avec k = budget d'alertes par jour × nombre de jours de la période. "
    f"Budget par défaut : {application.nombre(parametres['alertes_par_jour'])} alertes par jour, soit environ {application.nombre(parametres['alertes_par_jour'] / fraudes_par_jour, 1)} fois le nombre moyen de fraudes par jour mesuré "
    f"en validation ({application.nombre(fraudes_par_jour, 1)}) : assez pour retrouver la plupart des fraudes malgré des fausses alertes. Le curseur de budget le modifie."
)

jeu_graphique = st.radio("Période des graphiques", ["validation", "test"], format_func=str.capitalize, horizontal=True, key="jeu_graphique")
point = resultats[jeu_graphique]
gauche, droite = st.columns(2)
courbes = c.lire("courbes_precision_rappel.csv")
taux = parametres["jeux"][jeu_graphique]["fraudes"] / parametres["jeux"][jeu_graphique]["transactions"]
gauche.plotly_chart(
    graphiques.courbes_precision_rappel(
        courbes,
        jeu_graphique,
        ["xgboost", "logistique", "isolation_forest", "one_class_svm", "autoencodeur", "lof"],
        point=(point["rappel"], point["precision"]),
        methode_point=methode,
        taux=taux,
    )
)
courbe = courbe_de_cout(jeu_graphique, methode, cout_alerte)
droite.plotly_chart(
    graphiques.courbe_cout(
        courbe,
        point=(point["alertes_par_jour"], max(point["cout"], 1.0)) if point["alertes_par_jour"] <= 100 else None,
    )
)

with st.expander("Sensibilité du seuil optimal au coût d'analyse"):
    sensibilite = c.lire("sensibilite_cout.csv", source="resultats")
    sensibilite = sensibilite[sensibilite["methode"] == methode]
    tableau = pd.DataFrame(
        {
            "Coût par alerte": sensibilite["cout_alerte"].map(lambda v: application.nombre(v)),
            "Seuil": sensibilite["seuil"].map(lambda v: application.nombre(v, 4)),
            "Alertes (validation)": sensibilite["alertes_validation"].map(application.nombre),
            "Rappel (validation)": sensibilite["rappel_validation"].map(application.pourcentage),
            "Précision (validation)": sensibilite["precision_validation"].map(application.pourcentage),
            "Coût (test)": sensibilite["cout_test"].map(application.nombre),
            "Économie (test)": sensibilite["economie_test"].map(application.pourcentage),
        }
    )
    st.dataframe(tableau, hide_index=True)
    st.caption("Calculé pour 1, 2, 5, 10, 20 et 50 par alerte. Le seuil optimal reste le même pour 5 et 10 pour XGBoost.")

st.caption(
    "Le seuil de probabilité d'XGBoost est celui des probabilités corrigées du sous-échantillonnage. Les scores des méthodes non supervisées ne sont pas des probabilités."
)
