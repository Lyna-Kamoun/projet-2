import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import composants as c

c.configurer("Préparation")

import pandas as pd
import streamlit as st

from src import application, graphiques

c.entete("2. Préparation et variables", "Découpage chronologique, variables sans fuite de données, stratégie contre le déséquilibre.")

st.subheader("Découpage chronologique")
decoupage = c.lire("decoupage.csv")
affichage = pd.DataFrame(
    {
        "Jeu": decoupage["jeu"].map({"entrainement": "Entraînement (60 %)", "validation": "Validation (20 %)", "test": "Test (20 %)"}),
        "Transactions": decoupage["transactions"].map(application.nombre),
        "Fraudes": decoupage["fraudes"].map(application.nombre),
        "Taux de fraude": decoupage["taux_fraude"].map(lambda v: application.pourcentage(v, 2)),
        "Début": pd.to_datetime(decoupage["debut"]).dt.strftime("%d/%m/%Y"),
        "Fin": pd.to_datetime(decoupage["fin"]).dt.strftime("%d/%m/%Y"),
        "Cartes": decoupage["cartes"].map(application.nombre),
    }
)
st.dataframe(affichage, hide_index=True)
st.markdown(
    "Les deux fichiers du jeu (entraînement et test de Kaggle) se suivent dans le temps : ils sont fusionnés, triés par date, puis recoupés "
    "à 60 / 20 / 20. Le taux de fraude baisse d'une période à l'autre (0,58 %, 0,51 %, 0,36 %) : c'est le glissement de comportement que "
    "la validation chronologique permet de mesurer. Les cartes sont les mêmes d'une période à l'autre, ce qui est normal pour un modèle de fraude par carte "
    "mais constitue une fuite de groupe à garder en tête."
)

st.subheader("Variables")
variables = pd.DataFrame(
    [
        ("Temporelles", "heure_sin, heure_cos", "Heure en codage cyclique : 23 h et 1 h restent voisines."),
        ("Temporelles", "jour_sin, jour_cos", "Jour de la semaine en codage cyclique."),
        ("Temporelles", "nuit", "Vaut 1 de 22 h à 3 h 59, fenêtre lue sur l'entraînement."),
        ("Montant", "amt", "Montant de la transaction."),
        ("Comportementales", "nb_tx_1h, nb_tx_24h", "Transactions antérieures de la même carte sur la dernière heure et les dernières 24 heures."),
        ("Comportementales", "ratio_moy_30j", "Montant divisé par la moyenne des montants antérieurs de la carte sur 30 jours."),
        ("Comportementales", "delai_precedent", "Secondes écoulées depuis la transaction précédente de la carte."),
        ("Comportementales", "z_montant_30j", "(montant - moyenne) / écart-type sur 30 jours, si la carte a au moins 2 transactions antérieures."),
        ("Géographique", "distance_km", "Distance entre la position du porteur et celle du marchand."),
        ("Catégorielle", "category", "Catégorie d'achat, encodée en 14 colonnes."),
    ],
    columns=["Famille", "Variable", "Définition"],
)
st.dataframe(variables, hide_index=True)
st.markdown(
    "- **Aucune fuite** : les variables d'historique ne regardent que le passé strict de chaque carte (la transaction courante et le futur sont exclus). "
    "Un test vérifie que le résultat est identique quand on tronque les données après la transaction.\n"
    "- **Valeurs manquantes** : première transaction d'une carte ou historique trop court. XGBoost les traite nativement ; la régression logistique "
    "reçoit une imputation par la médiane de l'entraînement et un indicateur d'absence.\n"
    "- **Exclus** : identifiants, identité, localisation fine (`city_pop` quasi-identifiant de la carte), États, marchands, professions. "
    "L'âge et le genre sont aussi exclus (voir l'onglet Équité de la page Explicabilité)."
)

st.subheader("Stratégie contre le déséquilibre")
resume = c.lire("comparaison_desequilibre.csv", source="resultats")
resume["libelle"] = resume["strategie"].map(application.LIBELLES_STRATEGIES)
modele = st.radio(
    "Modèle",
    ["xgboost", "logistique"],
    format_func=lambda m: graphiques.LIBELLES[m],
    horizontal=True,
    key="modele_desequilibre",
)
st.plotly_chart(graphiques.comparaison_strategies(resume, modele))
if modele == "xgboost":
    st.success(
        "**Retenu : sous-échantillonnage 1:20.** SMOTE a la PR-AUC moyenne la plus haute (0,918), mais son avance sur 1:20 (0,917) n'est pas distinguable du bruit "
        "entre tirages ; 1:20 bat l'absence de rééchantillonnage et la pondération, et entraîne six fois plus vite. Les probabilités sont corrigées par le facteur "
        "β = 0,1159 (part des transactions normales conservées)."
    )
else:
    st.success(
        "**Retenu : aucun rééchantillonnage.** La PR-AUC est la meilleure sans rééchantillonnage (0,452). La ROC-AUC monte pourtant légèrement avec "
        "le rééchantillonnage : elle est trompeuse sur ce problème."
    )
st.caption("Règle fixée à l'avance : meilleure PR-AUC en validation, sauf si l'avance sur une stratégie plus simple est inférieure au bruit entre tirages. Trois graines par stratégie à tirage.")

st.subheader("Réglage des hyperparamètres")
st.markdown(
    "Recherche aléatoire (30 tirages) pour XGBoost et grille sur C pour la régression logistique, par validation croisée chronologique à fenêtre "
    "croissante dans l'entraînement. La validation reste réservée au choix du seuil."
)
gauche, droite = st.columns(2)
xgboost = c.lire("reglage_xgboost.csv", source="resultats")
meilleurs = xgboost.head(4)[["reference", "pr_auc_moyenne", "n_estimators", "max_depth", "learning_rate"]].copy()
meilleurs["reference"] = meilleurs["reference"].map({True: "Configuration par défaut", False: "Tirage aléatoire"})
meilleurs.columns = ["Configuration", "PR-AUC moyenne", "Arbres", "Profondeur", "Taux d'apprentissage"]
meilleurs["PR-AUC moyenne"] = meilleurs["PR-AUC moyenne"].map(lambda v: application.nombre(v, 4))
meilleurs["Taux d'apprentissage"] = meilleurs["Taux d'apprentissage"].map(lambda v: application.nombre(v, 3))
gauche.markdown("**XGBoost, quatre meilleures configurations**")
gauche.dataframe(meilleurs, hide_index=True)
gauche.caption("La configuration par défaut est la meilleure : aucun tirage ne la dépasse de plus de 0,001. Elle est conservée.")
logistique = c.lire("reglage_logistique.csv", source="resultats")[["C", "pr_auc_moyenne", "pr_auc_ecart_type"]].copy()
logistique.columns = ["C", "PR-AUC moyenne", "Écart-type entre plis"]
for colonne in ("PR-AUC moyenne", "Écart-type entre plis"):
    logistique[colonne] = logistique[colonne].map(lambda v: application.nombre(v, 4))
logistique["C"] = logistique["C"].map(lambda v: f"{v:g}".replace(".", ","))
droite.markdown("**Régression logistique, grille sur C**")
droite.dataframe(logistique, hide_index=True)
droite.caption("La PR-AUC plafonne dès C = 1, retenu comme le plus petit C à moins de 0,001 du meilleur.")
