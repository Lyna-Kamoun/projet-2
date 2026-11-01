import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import composants as c

c.configurer("Explicabilité")

import numpy as np
import pandas as pd
import streamlit as st

from src import application, explicabilite, graphiques

c.entete("5. Explicabilité", "Comprendre le modèle dans son ensemble, puis expliquer une alerte à un analyste.")

parametres = c.parametres()
seuil = parametres["seuils"]["xgboost"]
modele = c.modele("xgboost")

onglets = st.tabs(["Importance globale", "Dépendances", "Expliquer une alerte", "Transaction personnalisée", "Équité"])


def presenter(tableau, probabilite, fraude=None):
    colonnes = st.columns(3)
    colonnes[0].metric("Probabilité de fraude", application.pourcentage(probabilite, 1))
    colonnes[1].metric("Décision au seuil retenu", "Alerte" if probabilite >= seuil else "Pas d'alerte", help=f"Seuil : {application.pourcentage(seuil, 2)}")
    if fraude is not None:
        colonnes[2].metric("Réalité", "Fraude" if fraude == 1 else "Transaction normale")
    positifs = tableau[tableau["contribution"] > 0].head(5)
    negatifs = tableau[tableau["contribution"] < 0].head(3)
    if len(positifs):
        phrase = "; ".join(positifs["libelle"].head(3).str[0].str.lower() + positifs["libelle"].head(3).str[1:])
        if probabilite >= seuil:
            st.markdown(f"**Alerte déclenchée principalement par :** {phrase}.")
        else:
            st.markdown(f"**Pas d'alerte. Principaux facteurs qui poussent vers l'alerte :** {phrase}.")
    if len(negatifs):
        phrase = "; ".join(negatifs["libelle"].head(2).str[0].str.lower() + negatifs["libelle"].head(2).str[1:])
        st.markdown(f"**Facteurs qui rassurent :** {phrase}.")
    st.plotly_chart(graphiques.cascade(tableau))
    motifs = tableau[tableau["contribution"] > 0].head(5)[["code", "libelle", "contribution"]].copy()
    motifs["contribution"] = motifs["contribution"].map(lambda v: application.nombre(v, 2))
    motifs.columns = ["Code motif", "Motif", "Contribution (log-odds)"]
    st.markdown("**Codes motifs** (variables qui contribuent le plus à l'alerte)")
    st.dataframe(motifs, hide_index=True)


with onglets[0]:
    st.markdown(
        "SHAP répartit la prédiction entre les variables : la contribution d'une variable est son effet moyen sur la prédiction, en log-odds. "
        "Les colonnes dérivées d'une même variable sont regroupées (14 colonnes de catégorie, heure avec son indicateur de nuit)."
    )
    gauche, droite = st.columns(2)
    population = c.lire("importance_shap_population.csv", source="resultats", index_col=0)
    gauche.plotly_chart(graphiques.importance(population["importance"], "Importance SHAP (échantillon de 20 000 transactions)", "Contribution moyenne absolue"))
    permutation = c.lire("importance_permutation.csv", source="resultats", index_col=0)
    droite.plotly_chart(graphiques.importance(permutation["perte_pr_auc"].clip(lower=0), "Importance par permutation (perte de PR-AUC)", "Perte de PR-AUC"))
    alertes = c.lire("importance_shap_alertes.csv", source="resultats", index_col=0)
    with st.expander("Importance SHAP sur les alertes seules"):
        st.plotly_chart(graphiques.importance(alertes["importance"], "Importance SHAP sur les 3 763 alertes de la validation", "Contribution moyenne absolue"))
    st.markdown("**Nuage d'abeilles SHAP** : chaque point est une transaction ; la couleur est la valeur de la variable, la position horizontale son effet.")
    c.image("nuage_abeilles.png")
    st.info(
        "Mélanger le montant ou la catégorie fait tomber la PR-AUC de 0,917 à environ 0,01 et 0,19 : le signal est l'interaction du montant et de la catégorie. "
        "Ce n'est pas une fuite de données (le montant et la catégorie sont connus à la décision et la performance tient sur la période de test), "
        "mais c'est une signature du simulateur."
    )

with onglets[1]:
    st.markdown("Chaque graphique montre l'effet d'une variable sur la prédiction en fonction de sa valeur, la couleur indiquant la variable qui interagit le plus.")
    choix = st.selectbox(
        "Variable",
        ["amt", "ratio_moy_30j", "z_montant_30j", "delai_precedent"],
        format_func=lambda v: {
            "amt": "Montant",
            "ratio_moy_30j": "Montant / moyenne 30 jours",
            "z_montant_30j": "Écart au montant habituel",
            "delai_precedent": "Délai depuis la précédente",
        }[v],
        key="variable_dependance",
    )
    c.image(f"dependance_{choix}.png")
    lectures = {
        "amt": "Effet en cloche : maximal autour de 900 à 1 100, plus faible au-delà de 1 400. Le modèle a appris le plafond de 1 376 des fraudes du simulateur ; en situation réelle, la suspicion croîtrait avec le montant.",
        "ratio_moy_30j": "Deux régimes : un ratio très faible avec de petits montants pousse fortement vers l'alerte (fraudes à petit montant) ; pour les gros montants l'effet est proche de zéro.",
        "z_montant_30j": "Contributions négatives au-delà de 5, uniquement avec des montants élevés. Contraire à l'intuition : variable très liée au montant, effectifs rares au-delà de 10, explication non établie. À lire avec le montant.",
        "delai_precedent": "Positif pour des délais d'environ 14 à 25 heures, négatif au-delà. Effet non monotone, non expliqué à ce stade.",
    }
    st.caption(lectures[choix])

with onglets[2]:
    echantillon = c.echantillon().copy()
    alerte = echantillon["probabilite"] >= seuil
    fraude = echantillon["is_fraud"] == 1
    echantillon["statut"] = np.select(
        [alerte & fraude, alerte & ~fraude, ~alerte & fraude],
        ["Fraude détectée", "Fausse alerte", "Fraude manquée"],
        default="Transaction normale non signalée",
    )
    statut = st.radio(
        "Type de transaction",
        ["Fraude détectée", "Fausse alerte", "Fraude manquée"],
        horizontal=True,
        key="statut_explication",
    )
    sous = echantillon[echantillon["statut"] == statut].sort_values("probabilite", ascending=False).head(300)
    st.caption(f"Échantillon de la période de test : {application.nombre(len(sous))} transactions affichées sur {application.nombre((echantillon['statut'] == statut).sum())} disponibles (les plus probables d'abord).")
    options = sous.index.tolist()
    choix = st.selectbox(
        "Transaction",
        options,
        format_func=lambda i: f"{echantillon.loc[i, 'trans_date_trans_time']:%d/%m/%Y %H:%M} · montant {application.nombre(echantillon.loc[i, 'amt'], 2)} · {echantillon.loc[i, 'category']} · probabilité {application.pourcentage(echantillon.loc[i, 'probabilite'], 1)}",
        key=f"transaction_expliquee_{statut}",
    )
    ligne = echantillon.loc[[choix]].reset_index(drop=True)
    tableau = explicabilite.cascade(modele, ligne, 0)
    presenter(tableau, float(ligne.loc[0, "probabilite"]), int(ligne.loc[0, "is_fraud"]))

with onglets[3]:
    st.markdown("Saisir les caractéristiques d'une transaction fictive pour voir son score et son explication. Les variables d'historique sont à renseigner à la main.")
    gauche, milieu, droite = st.columns(3)
    jour = gauche.date_input("Date", pd.Timestamp("2020-10-15"), key="date_saisie")
    heure = gauche.slider("Heure", 0, 23, 23, key="heure_saisie")
    montant = milieu.number_input("Montant", min_value=1.0, max_value=30000.0, value=950.0, step=10.0, key="montant_saisi")
    categorie = milieu.selectbox("Catégorie", application.CATEGORIES, index=application.CATEGORIES.index("shopping_net"), key="categorie_saisie")
    distance = droite.number_input("Distance porteur-marchand (km)", min_value=0.0, max_value=200.0, value=78.0, step=1.0, key="distance_saisie")
    sans_historique = st.checkbox("Aucun historique pour cette carte (première transaction)", key="sans_historique")
    historique = st.columns(5)
    nb_tx_1h = historique[0].number_input("Transactions sur 1 h", min_value=0, max_value=20, value=0, key="tx_1h_saisie", disabled=sans_historique)
    nb_tx_24h = historique[1].number_input("Transactions sur 24 h", min_value=0, max_value=60, value=3, key="tx_24h_saisie", disabled=sans_historique)
    ratio = historique[2].number_input("Montant / moyenne 30 jours", min_value=0.0, max_value=200.0, value=1.0, step=0.1, key="ratio_saisi", disabled=sans_historique)
    delai = historique[3].number_input("Délai depuis la précédente (heures)", min_value=0.0, max_value=1000.0, value=12.0, step=1.0, key="delai_saisi", disabled=sans_historique)
    ecart = historique[4].number_input("Écart au montant habituel (écarts-types)", min_value=-50.0, max_value=200.0, value=0.0, step=0.5, key="ecart_saisi", disabled=sans_historique)
    transaction = application.transaction_personnalisee(
        pd.Timestamp(jour) + pd.Timedelta(hours=int(heure)),
        montant,
        categorie,
        0 if sans_historique else nb_tx_1h,
        0 if sans_historique else nb_tx_24h,
        np.nan if sans_historique else ratio,
        np.nan if sans_historique else delai * 3600.0,
        np.nan if sans_historique else ecart,
        distance,
    )
    probabilite = float(modele.probabilites(transaction)[0])
    presenter(explicabilite.cascade(modele, transaction, 0), probabilite)

with onglets[4]:
    st.markdown(
        "**Âge et genre ne sont pas utilisés.** Comparés avec et sans, ils améliorent nettement le modèle (PR-AUC de test 0,900 à 0,936), mais l'âge, constant pour une carte à un an près, "
        "identifie presque chaque porteur avec le genre : le gain vient probablement de la reconnaissance de cartes déjà vues (hypothèse non démontrée), et ce sont des caractéristiques protégées."
    )
    comparaison = c.lire("equite_comparaison_modeles.csv", source="resultats")
    tableau = pd.DataFrame(
        {
            "Modèle": comparaison["modele"].map({"xgboost": "Sans âge ni genre (retenu)", "xgboost_avec_age_genre": "Avec âge et genre"}),
            "Période": comparaison["jeu"].str.capitalize(),
            "PR-AUC": comparaison["pr_auc"].map(lambda v: application.nombre(v, 3)),
            "Rappel": comparaison["rappel"].map(lambda v: application.nombre(v, 3)),
            "Précision": comparaison["precision"].map(lambda v: application.nombre(v, 3)),
            "Coût": comparaison["cout"].map(application.nombre),
        }
    )
    st.dataframe(tableau, hide_index=True)
    st.subheader("Équité des alertes du modèle retenu")
    groupes = c.lire("equite_groupes.csv", source="resultats")
    colonnes = st.columns(2)
    jeu = colonnes[0].radio("Période", ["test", "validation"], format_func=str.capitalize, horizontal=True, key="jeu_equite")
    type_groupe = colonnes[1].radio("Groupes", ["genre", "age"], format_func=lambda t: {"genre": "Genre", "age": "Tranche d'âge"}[t], horizontal=True, key="type_equite")
    sous = groupes[(groupes["jeu"] == jeu) & (groupes["type"] == type_groupe)]
    tableau = pd.DataFrame(
        {
            "Groupe": sous["groupe"],
            "Cartes": sous["cartes"].map(application.nombre),
            "Taux de fraude": sous["taux_fraude"].map(lambda v: application.pourcentage(v, 2)),
            "Taux d'alerte": sous["taux_alerte"].map(lambda v: application.pourcentage(v, 2)),
            "Rappel": sous["rappel"].map(application.pourcentage),
            "Précision": sous["precision"].map(application.pourcentage),
            "Fausses alertes": sous["taux_fausses_alertes"].map(lambda v: application.pourcentage(v, 2)),
        }
    )
    st.dataframe(tableau, hide_index=True)
    st.caption(
        "Seuil choisi sur la validation. Les taux de fraude des groupes sont voisins, mais les femmes reçoivent plus d'alertes (1,2 fois celles des hommes sur le test, 1,3 fois sur la validation) et plus de fausses alertes (1,6 et 1,7 fois). "
        "Le modèle n'utilise ni l'âge ni le genre : l'écart passe par des variables corrélées, mécanisme non établi. Écarts indicatifs : de 62 à 468 cartes par groupe, observations groupées par carte, pas d'intervalle de confiance."
    )
