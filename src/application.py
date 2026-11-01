import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import donnees, metriques, modeles_supervises, variables
from src.donnees import COLONNE_CARTE, COLONNE_DATE

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_RESSOURCES = RACINE / "app" / "ressources"
DOSSIER_RESULTATS = RACINE / "resultats"
DOSSIER_FIGURES = DOSSIER_RESULTATS / "figures"
LIGNES_MAXIMUM_IMPORT = 300000
COLONNES_IMPORT = [
    "trans_date_trans_time",
    COLONNE_CARTE,
    "category",
    "amt",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "dob",
    "gender",
]
CATEGORIES = [
    "entertainment",
    "food_dining",
    "gas_transport",
    "grocery_net",
    "grocery_pos",
    "health_fitness",
    "home",
    "kids_pets",
    "misc_net",
    "misc_pos",
    "personal_care",
    "shopping_net",
    "shopping_pos",
    "travel",
]
LIBELLES_STRATEGIES = {
    "aucune": "Aucune",
    "ponderation": "Pondération des classes",
    "sous_1_1": "Sous-échantillonnage 1:1",
    "sous_1_5": "Sous-échantillonnage 1:5",
    "sous_1_10": "Sous-échantillonnage 1:10",
    "sous_1_20": "Sous-échantillonnage 1:20",
    "smote_1_10": "SMOTE 1:10",
}


def nombre(valeur, decimales=0):
    if valeur is None or (isinstance(valeur, float) and np.isnan(valeur)):
        return "n.d."
    texte = f"{valeur:,.{decimales}f}"
    return texte.replace(",", " ").replace(".", ",")


def pourcentage(valeur, decimales=1):
    if valeur is None or (isinstance(valeur, float) and np.isnan(valeur)):
        return "n.d."
    return nombre(valeur * 100.0, decimales) + " %"


def lire_csv(nom, dossier=None, **options):
    return pd.read_csv((dossier or DOSSIER_RESSOURCES) / nom, **options)


def lire_resultat(nom, **options):
    return pd.read_csv(DOSSIER_RESULTATS / nom, **options)


def lire_parametres(dossier=None):
    with open((dossier or DOSSIER_RESSOURCES) / "parametres.json", encoding="utf-8") as f:
        return json.load(f)


def lire_scores(jeu, dossier=None):
    with np.load((dossier or DOSSIER_RESSOURCES) / f"scores_{jeu}.npz") as contenu:
        return {cle: contenu[cle] for cle in contenu.files}


def indicateurs(scores, fraude, montant, seuil, cout_alerte, jours):
    resultat = metriques.indicateurs_au_seuil(fraude, scores, seuil)
    cout = metriques.cout_au_seuil(fraude, scores, montant, seuil, cout_alerte)
    sans_modele = metriques.cout_sans_modele(fraude, montant)
    resultat.update(
        {
            "alertes_par_jour": resultat["alertes"] / jours,
            "cout": cout,
            "cout_sans_modele": sans_modele,
            "economie": 1.0 - cout / sans_modele if sans_modele > 0 else 0.0,
            "seuil": float(seuil),
        }
    )
    return resultat


def indicateurs_classement(scores, fraude, alertes_par_jour, jours):
    k = int(round(alertes_par_jour * jours))
    k = min(max(k, 1), len(scores))
    precision, rappel = metriques.precision_rappel_au_rang(fraude, scores, k)
    return {
        "pr_auc": metriques.pr_auc(fraude, scores),
        "roc_auc": metriques.roc_auc(fraude, scores),
        "precision_a_k": precision,
        "rappel_a_k": rappel,
        "k": k,
    }


def seuil_budget(scores, alertes_par_jour, jours):
    k = int(round(alertes_par_jour * jours))
    k = min(max(k, 1), len(scores))
    return float(np.sort(scores)[-k])


def seuil_optimal(scores, fraude, montant, cout_alerte):
    return metriques.seuil_optimal(fraude, scores, montant, cout_alerte)


def courbe_cout(scores, fraude, montant, cout_alerte, jours, maximum_par_jour=100.0, points=700):
    courbe = metriques.courbe_cout(fraude, scores, montant, cout_alerte)
    courbe = courbe.assign(alertes_par_jour=courbe["alertes"] / jours)
    courbe = courbe[courbe["alertes_par_jour"] <= maximum_par_jour]
    if len(courbe) > points:
        positions = np.unique(np.linspace(0, len(courbe) - 1, points).astype(int))
        courbe = courbe.iloc[positions]
    return courbe.reset_index(drop=True)


def bornes_curseur_log(seuil_defaut, minimum=-4.0, maximum=0.0):
    return minimum, maximum, round(float(np.log10(seuil_defaut)), 2)


def transaction_personnalisee(
    horodatage,
    montant,
    categorie,
    nb_tx_1h=0,
    nb_tx_24h=0,
    ratio_moy_30j=np.nan,
    delai_precedent=np.nan,
    z_montant_30j=np.nan,
    distance_km=0.0,
):
    ligne = pd.DataFrame(
        {
            COLONNE_DATE: [pd.Timestamp(horodatage)],
            "amt": [float(montant)],
            "category": [categorie],
            "nb_tx_1h": [int(nb_tx_1h)],
            "nb_tx_24h": [int(nb_tx_24h)],
            "ratio_moy_30j": [float(ratio_moy_30j)],
            "delai_precedent": [float(delai_precedent)],
            "z_montant_30j": [float(z_montant_30j)],
            "distance_km": [float(distance_km)],
        }
    )
    temporelles = variables.variables_temporelles(ligne)
    return pd.concat([ligne, temporelles], axis=1)


def verifier_import(df):
    manquantes = [c for c in COLONNES_IMPORT if c not in df.columns]
    if manquantes:
        raise ValueError("Colonnes manquantes : " + ", ".join(manquantes))
    if len(df) == 0:
        raise ValueError("Le fichier ne contient aucune transaction.")
    if len(df) > LIGNES_MAXIMUM_IMPORT:
        raise ValueError(
            f"Le fichier contient {nombre(len(df))} lignes ; la limite est de {nombre(LIGNES_MAXIMUM_IMPORT)}."
        )


def analyser_import(fichier, modele=None):
    df = pd.read_csv(fichier)
    verifier_import(df)
    prepare = donnees.preparer(df)
    avec_variables = variables.construire_variables(prepare)
    if modele is not None:
        avec_variables["probabilite"] = modele.probabilites(avec_variables)
    return avec_variables


def charger_modele(nom="xgboost"):
    return modeles_supervises.charger(nom)
