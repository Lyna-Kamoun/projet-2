import pandas as pd

from src.donnees import (
    COLONNE_CATEGORIE,
    COLONNE_CIBLE,
    COLONNE_DATE,
    COLONNE_ETAT,
    COLONNE_MONTANT,
)

JOURS = {
    0: "lundi",
    1: "mardi",
    2: "mercredi",
    3: "jeudi",
    4: "vendredi",
    5: "samedi",
    6: "dimanche",
}


def canal(categorie):
    if categorie.endswith("_pos"):
        return "magasin"
    if categorie.endswith("_net"):
        return "en ligne"
    return "non precise"


def agreger(df, cle, nom):
    base = pd.DataFrame(
        {
            nom: cle,
            "fraude": df[COLONNE_CIBLE],
            "montant_fraude": df[COLONNE_MONTANT] * df[COLONNE_CIBLE],
        }
    )
    groupes = base.groupby(nom, observed=True)
    tableau = pd.DataFrame(
        {
            "transactions": groupes["fraude"].size(),
            "fraudes": groupes["fraude"].sum(),
            "montant_fraudes": groupes["montant_fraude"].sum(),
        }
    )
    tableau["taux_fraude"] = tableau["fraudes"] / tableau["transactions"]
    return tableau[["transactions", "fraudes", "taux_fraude", "montant_fraudes"]]


def resume_global(df):
    fraudes = int(df[COLONNE_CIBLE].sum())
    montant_total = float(df[COLONNE_MONTANT].sum())
    montant_fraudes = float((df[COLONNE_MONTANT] * df[COLONNE_CIBLE]).sum())
    return pd.DataFrame(
        {
            "transactions": [len(df)],
            "fraudes": [fraudes],
            "taux_fraude": [fraudes / len(df)],
            "montant_total": [montant_total],
            "montant_fraudes": [montant_fraudes],
            "part_montant_fraudes": [montant_fraudes / montant_total],
        }
    )


def montants_par_classe(df):
    groupes = df.groupby(COLONNE_CIBLE)[COLONNE_MONTANT]
    return pd.DataFrame(
        {
            "transactions": groupes.size(),
            "moyenne": groupes.mean(),
            "mediane": groupes.median(),
            "ecart_type": groupes.std(),
            "minimum": groupes.min(),
            "q1": groupes.quantile(0.25),
            "q3": groupes.quantile(0.75),
            "maximum": groupes.max(),
        }
    )


def taux_par_mois(df):
    mois = df[COLONNE_DATE].dt.to_period("M").astype(str)
    return agreger(df, mois, "mois")


def taux_par_heure(df):
    return agreger(df, df[COLONNE_DATE].dt.hour, "heure")


def taux_par_jour_semaine(df):
    tableau = agreger(df, df[COLONNE_DATE].dt.day_of_week, "jour")
    tableau.index = tableau.index.map(JOURS)
    return tableau


def taux_par_categorie(df):
    return agreger(df, df[COLONNE_CATEGORIE], "categorie").sort_values(
        "taux_fraude", ascending=False
    )


def taux_par_canal(df):
    return agreger(df, df[COLONNE_CATEGORIE].map(canal), "canal")


def taux_par_etat(df):
    return agreger(df, df[COLONNE_ETAT], "etat").sort_values(
        "taux_fraude", ascending=False
    )
