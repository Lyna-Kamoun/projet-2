import numpy as np
import pandas as pd
import shap

from src import desequilibre, metriques, preparation
from src.donnees import COLONNE_CATEGORIE, COLONNE_DATE, COLONNE_MONTANT
from src.exploration import JOURS

GROUPES_FIXES = {
    "heure": ["heure_sin", "heure_cos", "nuit"],
    "jour": ["jour_sin", "jour_cos"],
}
CODES = {
    "heure": "HEURE",
    "jour": "JOUR_SEMAINE",
    "amt": "MONTANT",
    "nb_tx_1h": "VELOCITE_1H",
    "nb_tx_24h": "VELOCITE_24H",
    "ratio_moy_30j": "RATIO_MOYENNE_30J",
    "delai_precedent": "DELAI_PRECEDENTE",
    "z_montant_30j": "ECART_MONTANT_30J",
    "distance_km": "DISTANCE",
    "category": "CATEGORIE",
}
COLONNES_PERMUTATION = {
    "heure": ["heure_sin", "heure_cos", "nuit"],
    "jour": ["jour_sin", "jour_cos"],
    "amt": ["amt"],
    "nb_tx_1h": ["nb_tx_1h"],
    "nb_tx_24h": ["nb_tx_24h"],
    "ratio_moy_30j": ["ratio_moy_30j"],
    "delai_precedent": ["delai_precedent"],
    "z_montant_30j": ["z_montant_30j"],
    "distance_km": ["distance_km"],
    "category": [COLONNE_CATEGORIE],
}


class ResultatShap:
    def __init__(self, matrice, valeurs, base):
        self.matrice = matrice
        self.valeurs = valeurs
        self.base = base

    def groupees(self, groupes=None):
        return grouper(self.valeurs, groupes or groupes_colonnes(self.valeurs.columns))


def groupes_colonnes(colonnes):
    colonnes = list(colonnes)
    groupes = {}
    deja = set()
    for nom, membres in GROUPES_FIXES.items():
        presents = [c for c in membres if c in colonnes]
        if presents:
            groupes[nom] = presents
            deja.update(presents)
    categories = [c for c in colonnes if c.startswith(f"{COLONNE_CATEGORIE}_")]
    if categories:
        groupes[COLONNE_CATEGORIE] = categories
        deja.update(categories)
    for colonne in colonnes:
        if colonne not in deja:
            groupes[colonne] = [colonne]
    return groupes


def grouper(valeurs, groupes):
    return pd.DataFrame(
        {nom: valeurs[membres].sum(axis=1) for nom, membres in groupes.items()},
        index=valeurs.index,
    )


def calculer_shap(modele, jeu):
    matrice = modele.matrice(jeu)
    explicateur = shap.TreeExplainer(modele.estimateur)
    valeurs = pd.DataFrame(
        explicateur.shap_values(matrice), index=matrice.index, columns=matrice.columns
    )
    base = float(np.ravel(explicateur.expected_value)[0])
    return ResultatShap(matrice, valeurs, base)


def marge(modele, jeu):
    return modele.estimateur.predict(modele.matrice(jeu), output_margin=True)


def importance_shap(groupees):
    moyenne = groupees.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({"importance": moyenne, "part": moyenne / moyenne.sum()})


def importance_permutation(modele, jeu, repetitions=3, graine=0):
    y = preparation.cible(jeu)
    reference = metriques.pr_auc(y, modele.scores(jeu))
    hasard = np.random.default_rng(graine)
    lignes = []
    for nom, colonnes in COLONNES_PERMUTATION.items():
        pertes = []
        for _ in range(repetitions):
            copie = jeu.copy(deep=False)
            ordre = hasard.permutation(len(jeu))
            for colonne in colonnes:
                copie[colonne] = jeu[colonne].to_numpy()[ordre]
            pertes.append(reference - metriques.pr_auc(y, modele.scores(copie)))
        lignes.append(
            {
                "variable": nom,
                "perte_pr_auc": float(np.mean(pertes)),
                "ecart_type": float(np.std(pertes, ddof=1)) if repetitions > 1 else 0.0,
            }
        )
    resultat = pd.DataFrame(lignes).set_index("variable")
    resultat["pr_auc_reference"] = reference
    return resultat.sort_values("perte_pr_auc", ascending=False)


def nombre(valeur, decimales=1, signe=False):
    format_ = f"{{:{'+' if signe else ''}.{decimales}f}}"
    return format_.format(valeur).replace(".", ",")


def duree_lisible(secondes):
    if secondes < 3600:
        return f"{nombre(secondes / 60, 0)} min"
    if secondes < 172800:
        return f"{nombre(secondes / 3600)} h"
    return f"{nombre(secondes / 86400)} jours"


def libelle(groupe, ligne):
    if groupe == "heure":
        return f"Heure de la transaction : {ligne[COLONNE_DATE].hour} h"
    if groupe == "jour":
        return f"Jour de la semaine : {JOURS[ligne[COLONNE_DATE].day_of_week]}"
    if groupe == "amt":
        return f"Montant : {nombre(ligne[COLONNE_MONTANT], 2)}"
    if groupe == "nb_tx_1h":
        return f"{int(ligne['nb_tx_1h'])} transaction(s) de la carte sur la dernière heure"
    if groupe == "nb_tx_24h":
        return f"{int(ligne['nb_tx_24h'])} transaction(s) de la carte sur 24 heures"
    if groupe == "ratio_moy_30j":
        if pd.isna(ligne["ratio_moy_30j"]):
            return "Aucun historique de montants sur 30 jours"
        return f"Montant égal à {nombre(ligne['ratio_moy_30j'])} fois la moyenne des 30 derniers jours"
    if groupe == "delai_precedent":
        if pd.isna(ligne["delai_precedent"]):
            return "Première transaction de la carte"
        return f"Délai depuis la transaction précédente : {duree_lisible(ligne['delai_precedent'])}"
    if groupe == "z_montant_30j":
        if pd.isna(ligne["z_montant_30j"]):
            return "Écart au montant habituel non calculable"
        return f"Écart au montant habituel : {nombre(ligne['z_montant_30j'], 1, True)} écart-type"
    if groupe == "distance_km":
        return f"Distance porteur-marchand : {nombre(ligne['distance_km'], 0)} km"
    if groupe == COLONNE_CATEGORIE:
        return f"Catégorie d'achat : {ligne[COLONNE_CATEGORIE]}"
    return f"{groupe} : {ligne[groupe]}"


def cascade(modele, jeu, position, resultat=None):
    resultat = resultat or calculer_shap(modele, jeu.iloc[[position]])
    ligne = jeu.iloc[position]
    groupees = resultat.groupees().iloc[0]
    tableau = pd.DataFrame(
        {
            "variable": groupees.index,
            "code": [CODES.get(g, g.upper()) for g in groupees.index],
            "libelle": [libelle(g, ligne) for g in groupees.index],
            "contribution": groupees.to_numpy(),
        }
    )
    tableau = tableau.reindex(tableau["contribution"].abs().sort_values(ascending=False).index)
    tableau = tableau.reset_index(drop=True)
    tableau.attrs["base"] = resultat.base
    tableau.attrs["marge"] = resultat.base + float(groupees.sum())
    probabilite_brute = 1.0 / (1.0 + np.exp(-tableau.attrs["marge"]))
    tableau.attrs["probabilite_corrigee"] = float(
        desequilibre.corriger_probabilites(probabilite_brute, modele.beta)
    )
    return tableau


def codes_motifs(modele, jeu, position, k=5, resultat=None):
    tableau = cascade(modele, jeu, position, resultat)
    positifs = tableau[tableau["contribution"] > 0].head(k)
    return positifs[["code", "libelle", "contribution"]].to_dict("records")
