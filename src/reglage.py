import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler

from src import desequilibre, metriques, preparation
from src.donnees import COLONNE_DATE

NOMBRE_BLOCS = 4
RATIO_SOUS_ECHANTILLONNAGE = 20
VALEURS_C = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
TOLERANCE_PR_AUC = 0.001
DISTRIBUTIONS_XGBOOST = {
    "n_estimators": randint(150, 601),
    "max_depth": randint(3, 11),
    "learning_rate": loguniform(0.03, 0.3),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
    "min_child_weight": randint(1, 11),
    "gamma": uniform(0.0, 2.0),
    "reg_lambda": loguniform(1.0, 20.0),
}


def plis_temporels(df, blocs=NOMBRE_BLOCS):
    df = df.sort_values(COLONNE_DATE, kind="stable")
    bornes = np.linspace(0, len(df), blocs + 1).astype(int)
    return [
        (df.iloc[: bornes[i]], df.iloc[bornes[i] : bornes[i + 1]])
        for i in range(1, blocs)
    ]


def preparer_plis(df, fabrique_pipeline, sous_echantillonner, graine):
    plis = []
    for appris, evalue in plis_temporels(df):
        pipeline = fabrique_pipeline()
        x_appris = pipeline.fit_transform(appris)
        x_evalue = pipeline.transform(evalue)
        y_appris = preparation.cible(appris)
        if sous_echantillonner:
            x_appris, y_appris, _ = desequilibre.sous_echantillonner(
                x_appris, y_appris, RATIO_SOUS_ECHANTILLONNAGE, graine
            )
        plis.append((x_appris, y_appris, x_evalue, preparation.cible(evalue)))
    return plis


def noter(fabrique_modele, plis):
    notes = []
    debut = time.time()
    for x_appris, y_appris, x_evalue, y_evalue in plis:
        modele = fabrique_modele()
        modele.fit(x_appris, y_appris)
        notes.append(metriques.pr_auc(y_evalue, modele.predict_proba(x_evalue)[:, 1]))
    return notes, time.time() - debut


def convertir(parametres):
    sortie = {}
    for cle, valeur in parametres.items():
        if isinstance(valeur, (str, bool)):
            sortie[cle] = valeur
        elif isinstance(valeur, (int, np.integer)):
            sortie[cle] = int(valeur)
        else:
            sortie[cle] = float(valeur)
    return sortie


def ligne_resultat(parametres, notes, duree, reference=False):
    ligne = dict(parametres)
    ligne.update(
        {
            "reference": reference,
            "pr_auc_moyenne": float(np.mean(notes)),
            "pr_auc_ecart_type": float(np.std(notes, ddof=1)),
            "duree": duree,
        }
    )
    for i, note in enumerate(notes, start=1):
        ligne[f"pr_auc_pli_{i}"] = note
    return ligne


def rechercher_xgboost(
    entrainement, configurations=30, graine=0, n_jobs=-1, verbeux=False
):
    plis = preparer_plis(entrainement, preparation.pipeline_arbres, True, graine)
    tirages = [
        convertir(p)
        for p in ParameterSampler(
            DISTRIBUTIONS_XGBOOST, n_iter=configurations, random_state=graine
        )
    ]
    candidats = [(dict(desequilibre.PARAMETRES_XGBOOST), True)] + [
        (p, False) for p in tirages
    ]
    lignes = []
    for parametres, reference in candidats:
        notes, duree = noter(
            lambda: desequilibre.modele_xgboost(
                graine, 1.0, n_jobs, parametres=parametres
            ),
            plis,
        )
        lignes.append(ligne_resultat(parametres, notes, duree, reference))
        if verbeux:
            print(len(lignes) - 1, round(float(np.mean(notes)), 4), round(duree, 1), flush=True)
    resultats = pd.DataFrame(lignes)
    return resultats.sort_values("pr_auc_moyenne", ascending=False).reset_index(drop=True)


def rechercher_logistique(entrainement, valeurs=VALEURS_C, graine=0, verbeux=False):
    plis = preparer_plis(entrainement, preparation.pipeline_lineaire, False, graine)
    lignes = []
    for c in valeurs:
        notes, duree = noter(
            lambda: LogisticRegression(C=c, max_iter=2000, random_state=graine), plis
        )
        lignes.append(ligne_resultat({"C": float(c)}, notes, duree))
        if verbeux:
            print("C", c, round(float(np.mean(notes)), 4), round(duree, 1), flush=True)
    resultats = pd.DataFrame(lignes)
    return resultats.sort_values("pr_auc_moyenne", ascending=False).reset_index(drop=True)


def choisir_c(resultats_logistique):
    meilleure = resultats_logistique["pr_auc_moyenne"].max()
    proches = resultats_logistique[
        resultats_logistique["pr_auc_moyenne"].round(9) >= round(meilleure - TOLERANCE_PR_AUC, 9)
    ]
    return float(proches["C"].min())


def choisir_xgboost(resultats_xgboost):
    reference = resultats_xgboost[resultats_xgboost["reference"]].iloc[0]
    tirages = resultats_xgboost[~resultats_xgboost["reference"]]
    if len(tirages):
        meilleur = tirages.iloc[0]
        gain = round(float(meilleur["pr_auc_moyenne"] - reference["pr_auc_moyenne"]), 9)
        if gain > TOLERANCE_PR_AUC:
            return convertir({c: meilleur[c] for c in DISTRIBUTIONS_XGBOOST})
    return convertir(dict(desequilibre.PARAMETRES_XGBOOST))


def meilleurs_parametres(resultats_xgboost, resultats_logistique):
    return {
        "xgboost": choisir_xgboost(resultats_xgboost),
        "logistique": {"C": choisir_c(resultats_logistique)},
    }
