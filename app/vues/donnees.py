import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import composants as c

c.configurer("Données")

import pandas as pd
import streamlit as st

from src import application, graphiques

c.entete(
    "1. Données",
    "Jeu Sparkov (Kaggle, licence CC0) : 1 852 394 transactions par carte bancaire du 01/01/2019 au 31/12/2020. "
    "Le fichier complet est téléchargé automatiquement avec kagglehub lors de la préparation ; l'application utilise les tables précalculées.",
)

resume = c.lire("exploration_resume.csv").iloc[0]
colonnes = st.columns(4)
colonnes[0].metric("Transactions", application.nombre(resume["transactions"]))
colonnes[1].metric("Fraudes", application.nombre(resume["fraudes"]))
colonnes[2].metric("Taux de fraude", application.pourcentage(resume["taux_fraude"], 2))
colonnes[3].metric(
    "Montant des fraudes",
    application.nombre(resume["montant_fraudes"]),
    help=f"Soit {application.pourcentage(resume['part_montant_fraudes'], 1)} du montant total des transactions.",
)

onglets = st.tabs(["Évolution dans le temps", "Heure et jour", "Catégorie et canal", "Montants", "États", "Importer un fichier"])

with onglets[0]:
    par_mois = c.lire("exploration_par_mois.csv")
    st.plotly_chart(graphiques.ligne(par_mois, "mois", "taux_fraude", "Taux de fraude par mois", "Taux de fraude"))
    figure = graphiques.barres(par_mois, "mois", "transactions", "Nombre de transactions par mois", "Transactions", format_texte=",.0f", hauteur=300)
    figure.update_traces(texttemplate=None, text=None)
    st.plotly_chart(figure)
    st.caption(
        "Le taux de fraude baisse au fil des périodes. Les volumes doublent en décembre sans hausse comparable des fraudes : "
        "le taux mensuel de décembre est mécaniquement plus bas."
    )
    with st.expander("Table des valeurs"):
        st.dataframe(par_mois, hide_index=True)

with onglets[1]:
    par_heure = c.lire("exploration_par_heure.csv")
    nuit = par_heure[par_heure["heure"].isin([22, 23, 0, 1, 2, 3])]
    part_nuit = nuit["fraudes"].sum() / par_heure["fraudes"].sum()
    figure = graphiques.barres(par_heure, "heure", "taux_fraude", "Taux de fraude selon l'heure de la journée", "Taux de fraude", format_texte=".1%")
    figure.update_xaxes(dtick=1, title="Heure")
    st.plotly_chart(figure)
    st.info(f"{application.pourcentage(part_nuit)} des fraudes ont lieu entre 22 h et 3 h 59. Signature typique du simulateur.")
    par_jour = c.lire("exploration_par_jour.csv")
    st.plotly_chart(graphiques.barres(par_jour, "jour", "taux_fraude", "Taux de fraude selon le jour de la semaine", "Taux de fraude", format_texte=".2%", hauteur=320))
    st.caption("Le taux varie surtout avec le volume de transactions normales, qui change beaucoup d'un jour à l'autre, et peu avec le nombre de fraudes.")

with onglets[2]:
    gauche, droite = st.columns([3, 2])
    par_categorie = c.lire("exploration_par_categorie.csv")
    gauche.plotly_chart(graphiques.barres(par_categorie, "categorie", "taux_fraude", "Taux de fraude par catégorie", "Taux de fraude", horizontal=True, format_texte=".2%", hauteur=480))
    par_canal = c.lire("exploration_par_canal.csv")
    droite.plotly_chart(graphiques.barres(par_canal, "canal", "taux_fraude", "Taux de fraude par canal", "Taux de fraude", format_texte=".2%", hauteur=480))
    st.caption(
        "Le jeu n'a pas de colonne de canal : il est déduit du suffixe de la catégorie (_pos : en magasin, _net : en ligne). "
        "Les autres catégories sont classées « non précisé »."
    )

with onglets[3]:
    st.plotly_chart(graphiques.histogramme_montants(c.lire("exploration_histogramme_montants.csv")))
    montants = c.lire("exploration_montants_par_classe.csv")
    montants["is_fraud"] = montants["is_fraud"].map({0: "Normales", 1: "Fraudes"})
    for colonne in montants.columns[1:]:
        montants[colonne] = montants[colonne].map(lambda v: application.nombre(v, 2) if colonne != "transactions" else application.nombre(v))
    montants.columns = ["Classe", "Transactions", "Moyenne", "Médiane", "Écart-type", "Minimum", "1er quartile", "3e quartile", "Maximum"]
    st.dataframe(montants, hide_index=True)
    st.caption("Les fraudes ont des montants nettement plus élevés (médiane autour de 390 contre 47) mais ne dépassent jamais 1 376, alors que les transactions normales vont jusqu'à 28 949.")

with onglets[4]:
    par_etat = c.lire("exploration_par_etat.csv")
    minimum = st.slider("Nombre minimal de transactions par État", 0, 20000, 1000, step=500, key="effectif_minimal_etat")
    filtre = par_etat[par_etat["transactions"] >= minimum].head(15)
    st.plotly_chart(graphiques.barres(filtre, "etat", "taux_fraude", "États au plus fort taux de fraude", "Taux de fraude", format_texte=".2%", hauteur=360))
    st.caption("Les effectifs de certains États sont très petits (l'État DE n'a que 9 transactions, toutes frauduleuses) : cette variable n'est pas utilisée par le modèle.")

with onglets[5]:
    st.markdown(
        "Importer un fichier CSV de transactions **au format du jeu Kaggle** (colonnes `trans_date_trans_time`, `cc_num`, `category`, `amt`, "
        "`lat`, `long`, `merch_lat`, `merch_long`, `dob`, `gender`, et `is_fraud` si elle est connue). Maximum "
        f"{application.nombre(application.LIGNES_MAXIMUM_IMPORT)} lignes. Les variables sont calculées à partir de l'historique du fichier. Le fichier n'est pas conservé."
    )
    fichier = st.file_uploader("Fichier CSV", type="csv", key="import_csv")
    if fichier is not None:
        try:
            with st.spinner("Calcul des variables et des scores..."):
                importe = application.analyser_import(fichier, c.modele("xgboost"))
        except ValueError as erreur:
            st.error(str(erreur))
        else:
            seuil = c.parametres()["seuils"]["xgboost"]
            alertes = importe[importe["probabilite"] >= seuil]
            colonnes = st.columns(4)
            colonnes[0].metric("Transactions", application.nombre(len(importe)))
            colonnes[1].metric("Cartes", application.nombre(importe["cc_num"].nunique()))
            colonnes[2].metric("Alertes au seuil retenu", application.nombre(len(alertes)))
            colonnes[3].metric("Période", f"{importe['trans_date_trans_time'].min():%d/%m/%Y} - {importe['trans_date_trans_time'].max():%d/%m/%Y}")
            if "is_fraud" in importe.columns:
                fraudes = int(importe["is_fraud"].sum())
                detectees = int(alertes["is_fraud"].sum()) if len(alertes) else 0
                st.write(
                    f"Fraudes déclarées dans le fichier : {application.nombre(fraudes)} (taux de fraude {application.pourcentage(fraudes / len(importe), 2)}) ; retrouvées dans les alertes : {application.nombre(detectees)}."
                )
            affichage = alertes.sort_values("probabilite", ascending=False).head(20)[["trans_date_trans_time", "amt", "category", "probabilite"]].copy()
            affichage.columns = ["Date", "Montant", "Catégorie", "Probabilité de fraude"]
            affichage["Probabilité de fraude"] = affichage["Probabilité de fraude"].map(lambda v: application.pourcentage(v))
            st.subheader("Alertes les plus probables")
            st.dataframe(affichage, hide_index=True)
