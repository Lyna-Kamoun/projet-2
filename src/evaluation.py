from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src import metriques
from src.donnees import COLONNE_CIBLE, COLONNE_DATE, COLONNE_MONTANT

METHODES_SUPERVISEES = ("xgboost", "logistique")
METHODES_NON_SUPERVISEES = ("isolation_forest", "one_class_svm", "autoencodeur", "lof")
METHODES = METHODES_SUPERVISEES + METHODES_NON_SUPERVISEES
LIBELLES = {
    "xgboost": "XGBoost",
    "logistique": "Régression logistique",
    "isolation_forest": "Isolation Forest",
    "one_class_svm": "One-Class SVM",
    "autoencodeur": "Autoencodeur",
    "lof": "LOF",
    "dbscan": "DBSCAN",
}
COUT_ALERTE = 5.0
ALERTES_PAR_JOUR = 20.0
VALEURS_COUT_ALERTE = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
BUDGETS_PAR_JOUR = (5, 10, 15, 20, 30, 50, 100)


def charger_scores(dossier):
    dossier = Path(dossier)
    jeux = {}
    for nom in ("validation", "test"):
        supervise = pd.read_parquet(dossier / f"scores_{nom}.parquet")
        autre = pd.read_parquet(dossier / f"scores_non_supervise_{nom}.parquet")
        if not supervise.index.equals(autre.index):
            raise ValueError("les index des fichiers de scores ne correspondent pas")
        jeu = supervise[[COLONNE_DATE, COLONNE_MONTANT, COLONNE_CIBLE]].copy()
        jeu["xgboost"] = supervise["proba_xgboost"]
        jeu["logistique"] = supervise["proba_logistique"]
        for methode in METHODES_NON_SUPERVISEES:
            jeu[methode] = autre[methode]
        jeux[nom] = jeu
    return jeux


def budget(jeu, alertes_par_jour):
    return metriques.budget_total(
        alertes_par_jour, jeu[COLONNE_DATE].min(), jeu[COLONNE_DATE].max()
    )


def evaluer_au_seuil(jeu, methode, seuil, cout_alerte, alertes_par_jour):
    y = jeu[COLONNE_CIBLE].to_numpy()
    scores = jeu[methode].to_numpy()
    montants = jeu[COLONNE_MONTANT].to_numpy()
    k = budget(jeu, alertes_par_jour)
    precision_k, rappel_k = metriques.precision_rappel_au_rang(y, scores, k)
    ligne = {
        "pr_auc": metriques.pr_auc(y, scores),
        "roc_auc": metriques.roc_auc(y, scores),
        "k": k,
        "precision_a_k": precision_k,
        "rappel_a_k": rappel_k,
        "seuil": float(seuil),
    }
    ligne.update(metriques.indicateurs_au_seuil(y, scores, seuil))
    cout = metriques.cout_au_seuil(y, scores, montants, seuil, cout_alerte)
    sans_modele = metriques.cout_sans_modele(y, montants)
    ligne.update(
        {
            "cout": cout,
            "cout_sans_modele": sans_modele,
            "economie": 1.0 - cout / sans_modele if sans_modele > 0 else 0.0,
        }
    )
    return ligne


def seuil_optimal_validation(jeux, methode, cout_alerte=COUT_ALERTE):
    validation = jeux["validation"]
    return metriques.seuil_optimal(
        validation[COLONNE_CIBLE].to_numpy(),
        validation[methode].to_numpy(),
        validation[COLONNE_MONTANT].to_numpy(),
        cout_alerte,
    )


def tableau_comparatif(
    jeux, methodes=METHODES, cout_alerte=COUT_ALERTE, alertes_par_jour=ALERTES_PAR_JOUR
):
    lignes = []
    for methode in methodes:
        seuil = seuil_optimal_validation(jeux, methode, cout_alerte)["seuil"]
        for nom, jeu in jeux.items():
            ligne = evaluer_au_seuil(jeu, methode, seuil, cout_alerte, alertes_par_jour)
            ligne.update({"methode": methode, "jeu": nom})
            if nom == "test":
                optimal_test = metriques.seuil_optimal(
                    jeu[COLONNE_CIBLE].to_numpy(),
                    jeu[methode].to_numpy(),
                    jeu[COLONNE_MONTANT].to_numpy(),
                    cout_alerte,
                )
                ligne["cout_optimal_du_test"] = optimal_test["cout"]
            lignes.append(ligne)
    tableau = pd.DataFrame(lignes)
    premieres = ["methode", "jeu"]
    return tableau[premieres + [c for c in tableau.columns if c not in premieres]]


def sensibilite_cout(jeux, methode, valeurs=VALEURS_COUT_ALERTE, alertes_par_jour=ALERTES_PAR_JOUR):
    lignes = []
    for cout_alerte in valeurs:
        optimum = seuil_optimal_validation(jeux, methode, cout_alerte)
        validation = evaluer_au_seuil(
            jeux["validation"], methode, optimum["seuil"], cout_alerte, alertes_par_jour
        )
        test = evaluer_au_seuil(
            jeux["test"], methode, optimum["seuil"], cout_alerte, alertes_par_jour
        )
        lignes.append(
            {
                "methode": methode,
                "cout_alerte": cout_alerte,
                "seuil": optimum["seuil"],
                "alertes_validation": validation["alertes"],
                "alertes_test": test["alertes"],
                "rappel_validation": validation["rappel"],
                "rappel_test": test["rappel"],
                "precision_validation": validation["precision"],
                "precision_test": test["precision"],
                "cout_validation": validation["cout"],
                "cout_test": test["cout"],
                "economie_validation": validation["economie"],
                "economie_test": test["economie"],
            }
        )
    return pd.DataFrame(lignes)


def selon_budget(jeux, methodes=METHODES, budgets=BUDGETS_PAR_JOUR):
    lignes = []
    for nom, jeu in jeux.items():
        y = jeu[COLONNE_CIBLE].to_numpy()
        for par_jour in budgets:
            k = budget(jeu, par_jour)
            for methode in methodes:
                precision, rappel = metriques.precision_rappel_au_rang(
                    y, jeu[methode].to_numpy(), k
                )
                lignes.append(
                    {
                        "jeu": nom,
                        "alertes_par_jour": par_jour,
                        "k": k,
                        "methode": methode,
                        "precision_a_k": precision,
                        "rappel_a_k": rappel,
                    }
                )
    return pd.DataFrame(lignes)


def complementarite(jeu, base, appoint, alertes_par_jour=ALERTES_PAR_JOUR):
    y = jeu[COLONNE_CIBLE].to_numpy().astype(bool)
    k = budget(jeu, alertes_par_jour)

    def haut(methode, taille):
        ordre = metriques.ordre_decroissant(jeu[methode].to_numpy())[:taille]
        retenu = np.zeros(len(jeu), dtype=bool)
        retenu[ordre] = True
        return retenu

    alertes_base = haut(base, k)
    alertes_appoint = haut(appoint, k)
    fraudes = int(y.sum())
    manquees = y & ~alertes_base
    recuperees = manquees & alertes_appoint
    union = alertes_base | alertes_appoint
    base_double = haut(base, int(union.sum()))
    return {
        "base": base,
        "appoint": appoint,
        "jeu_k": k,
        "fraudes": fraudes,
        "detectees_par_base": int((y & alertes_base).sum()),
        "manquees_par_base": int(manquees.sum()),
        "recuperees_par_appoint": int(recuperees.sum()),
        "alertes_union": int(union.sum()),
        "rappel_base": float((y & alertes_base).sum() / fraudes),
        "rappel_union": float((y & union).sum() / fraudes),
        "rappel_base_au_meme_budget": float((y & base_double).sum() / fraudes),
    }


def courbe_precision_rappel(jeu, methode, points=1500):
    precision, rappel, _ = precision_recall_curve(
        jeu[COLONNE_CIBLE].to_numpy(), jeu[methode].to_numpy()
    )
    if len(precision) > points:
        positions = np.unique(np.linspace(0, len(precision) - 1, points).astype(int))
        precision, rappel = precision[positions], rappel[positions]
    return rappel, precision


def courbe_cout_selon_alertes(jeu, methode, cout_alerte=COUT_ALERTE):
    courbe = metriques.courbe_cout(
        jeu[COLONNE_CIBLE].to_numpy(),
        jeu[methode].to_numpy(),
        jeu[COLONNE_MONTANT].to_numpy(),
        cout_alerte,
    )
    jours = (jeu[COLONNE_DATE].max() - jeu[COLONNE_DATE].min()).total_seconds() / 86400.0
    courbe = courbe.assign(alertes_par_jour=courbe["alertes"] / jours)
    return courbe
