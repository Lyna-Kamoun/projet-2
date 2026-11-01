import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import donnees, evaluation, exploration
from src.donnees import COLONNE_CIBLE, COLONNE_DATE, COLONNE_MONTANT

METHODES_CURSEUR = ("xgboost", "logistique", "isolation_forest")
COLONNES_EXPLICATION = [
    "heure_sin",
    "heure_cos",
    "jour_sin",
    "jour_cos",
    "nuit",
    "amt",
    "nb_tx_1h",
    "nb_tx_24h",
    "ratio_moy_30j",
    "delai_precedent",
    "z_montant_30j",
    "distance_km",
    "category",
]
POINTS_COURBE = 1500
CLASSES_HISTOGRAMME = 40


def tables_exploration(df):
    return {
        "exploration_resume.csv": exploration.resume_global(df),
        "exploration_par_mois.csv": exploration.taux_par_mois(df).reset_index(),
        "exploration_par_heure.csv": exploration.taux_par_heure(df).reset_index(),
        "exploration_par_jour.csv": exploration.taux_par_jour_semaine(df).reset_index(),
        "exploration_par_categorie.csv": exploration.taux_par_categorie(df).reset_index(),
        "exploration_par_canal.csv": exploration.taux_par_canal(df).reset_index(),
        "exploration_par_etat.csv": exploration.taux_par_etat(df).reset_index(),
        "exploration_montants_par_classe.csv": exploration.montants_par_classe(df).reset_index(),
    }


def histogramme_montants(df, classes=CLASSES_HISTOGRAMME):
    montants = df[COLONNE_MONTANT].to_numpy()
    bornes = np.logspace(0.0, np.log10(montants.max()) + 1e-9, classes + 1)
    bornes[0] = min(bornes[0], montants.min())
    y = df[COLONNE_CIBLE].to_numpy()
    normales, _ = np.histogram(montants[y == 0], bins=bornes)
    fraudes, _ = np.histogram(montants[y == 1], bins=bornes)
    return pd.DataFrame(
        {
            "borne_basse": bornes[:-1],
            "borne_haute": bornes[1:],
            "normales": normales,
            "fraudes": fraudes,
        }
    )


def decoupage(entrainement, validation, test):
    return donnees.resumer(
        {"entrainement": entrainement, "validation": validation, "test": test}
    ).reset_index(names="jeu")


def courbes_pr(jeux, methodes=evaluation.METHODES, points=POINTS_COURBE):
    morceaux = []
    for nom, jeu in jeux.items():
        for methode in methodes:
            rappel, precision = evaluation.courbe_precision_rappel(jeu, methode, points)
            morceaux.append(
                pd.DataFrame(
                    {"jeu": nom, "methode": methode, "rappel": rappel, "precision": precision}
                )
            )
    return pd.concat(morceaux, ignore_index=True)


def scores_compacts(jeu, methodes=METHODES_CURSEUR):
    tableaux = {
        "montant": jeu[COLONNE_MONTANT].to_numpy(dtype="float32"),
        "fraude": jeu[COLONNE_CIBLE].to_numpy(dtype="int8"),
    }
    for methode in methodes:
        valeurs = jeu[methode].to_numpy(dtype="float64")
        arrondi = valeurs if methode == "xgboost" else np.round(valeurs, 5)
        tableaux[methode] = arrondi.astype("float32")
    return tableaux


def echantillon_explication(test, probabilites, seuil, normales=3000, graine=0):
    alertes = probabilites >= seuil
    fraudes = test[COLONNE_CIBLE].to_numpy() == 1
    obligatoires = alertes | fraudes
    hasard = np.random.default_rng(graine)
    restantes = np.flatnonzero(~obligatoires)
    tirees = hasard.choice(restantes, size=min(normales, len(restantes)), replace=False)
    positions = np.sort(np.r_[np.flatnonzero(obligatoires), tirees])
    colonnes = [COLONNE_DATE, COLONNE_CIBLE] + COLONNES_EXPLICATION
    echantillon = test.iloc[positions][colonnes].copy()
    echantillon["probabilite"] = probabilites[positions].astype("float64")
    echantillon.insert(0, "identifiant", test.index[positions].to_numpy())
    return echantillon.reset_index(drop=True)


def parametres(jeux, modele, seuils, cout_alerte, alertes_par_jour):
    contenu = {
        "cout_alerte": cout_alerte,
        "alertes_par_jour": alertes_par_jour,
        "beta": float(modele.beta),
        "seuils": {m: float(s) for m, s in seuils.items()},
        "jeux": {},
    }
    for nom, jeu in jeux.items():
        jours = (jeu[COLONNE_DATE].max() - jeu[COLONNE_DATE].min()).total_seconds() / 86400.0
        contenu["jeux"][nom] = {
            "transactions": int(len(jeu)),
            "fraudes": int(jeu[COLONNE_CIBLE].sum()),
            "jours": float(jours),
            "debut": str(jeu[COLONNE_DATE].min()),
            "fin": str(jeu[COLONNE_DATE].max()),
            "cout_sans_modele": float((jeu[COLONNE_MONTANT] * jeu[COLONNE_CIBLE]).sum()),
        }
    return contenu


def ecrire(dossier, df, entrainement, validation, test, jeux_scores, modele, probabilites_test, seuils, cout_alerte, alertes_par_jour):
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    for nom, tableau in tables_exploration(df).items():
        tableau.to_csv(dossier / nom, index=False)
    histogramme_montants(df).to_csv(dossier / "exploration_histogramme_montants.csv", index=False)
    decoupage(entrainement, validation, test).to_csv(dossier / "decoupage.csv", index=False)
    courbes_pr(jeux_scores).to_csv(dossier / "courbes_precision_rappel.csv", index=False)
    for nom, jeu in jeux_scores.items():
        np.savez_compressed(dossier / f"scores_{nom}.npz", **scores_compacts(jeu))
    echantillon = echantillon_explication(test, probabilites_test, seuils["xgboost"])
    echantillon.to_parquet(dossier / "echantillon_explication.parquet", index=False)
    contenu = parametres(
        {"validation": validation, "test": test}, modele, seuils, cout_alerte, alertes_par_jour
    )
    with open(dossier / "parametres.json", "w", encoding="utf-8") as f:
        json.dump(contenu, f, ensure_ascii=False, indent=2)
    return sorted(p.name for p in dossier.iterdir())
