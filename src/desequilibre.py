import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src import metriques, preparation
from src.donnees import COLONNE_DATE, COLONNE_MONTANT

PARAMETRES_XGBOOST = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "tree_method": "hist",
}
RATIOS_SOUS_ECHANTILLONNAGE = (1, 5, 10, 20)
RATIO_SMOTE = 10
VOISINS_SMOTE = 5
MODELES = ("logistique", "xgboost")


def corriger_probabilites(ps, beta):
    ps = np.asarray(ps, dtype="float64")
    return beta * ps / (beta * ps - ps + 1.0)


def sous_echantillonner(x, y, normales_par_fraude, graine):
    echantillonneur = RandomUnderSampler(
        sampling_strategy=1.0 / normales_par_fraude, random_state=graine
    )
    x_res, y_res = echantillonneur.fit_resample(x, y)
    normales = int((np.asarray(y) == 0).sum())
    gardees = int((np.asarray(y_res) == 0).sum())
    return x_res, y_res, gardees / normales


def sur_echantillonner_smote(x, y, normales_par_fraude, graine, voisins=VOISINS_SMOTE):
    generateur = SMOTE(
        sampling_strategy=1.0 / normales_par_fraude,
        k_neighbors=voisins,
        random_state=graine,
    )
    return generateur.fit_resample(x, y)


def modele_logistique(graine, ponderee=False):
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced" if ponderee else None,
        random_state=graine,
    )


def modele_xgboost(graine, poids_positif=1.0, n_jobs=-1, parametres=None):
    reglages = dict(PARAMETRES_XGBOOST)
    if parametres:
        reglages.update(parametres)
    return XGBClassifier(
        **reglages,
        scale_pos_weight=poids_positif,
        random_state=graine,
        n_jobs=n_jobs,
        eval_metric="aucpr",
    )


def comparer_strategies(
    entrainement,
    validation,
    cout_alerte=5.0,
    alertes_par_jour=20.0,
    graines=(0, 1, 2),
    n_jobs=-1,
    parametres_xgboost=None,
    verbeux=False,
    modeles=MODELES,
    strategies=None,
):
    lineaire = preparation.pipeline_lineaire()
    arbres = preparation.pipeline_arbres()
    matrices = {
        "logistique": (
            lineaire.fit_transform(entrainement),
            lineaire.transform(validation),
        ),
        "xgboost": (arbres.fit_transform(entrainement), arbres.transform(validation)),
    }
    x_lineaire_entrainement = matrices["logistique"][0]
    y_entrainement = preparation.cible(entrainement)
    y_validation = preparation.cible(validation)
    montants = validation[COLONNE_MONTANT].to_numpy()
    k = metriques.budget_total(
        alertes_par_jour,
        validation[COLONNE_DATE].min(),
        validation[COLONNE_DATE].max(),
    )
    positifs = int(y_entrainement.sum())
    negatifs = len(y_entrainement) - positifs
    lignes = []

    def construire(modele, graine, poids_positif=1.0, ponderee=False):
        if modele == "logistique":
            return modele_logistique(graine, ponderee)
        return modele_xgboost(graine, poids_positif, n_jobs, parametres_xgboost)

    def essayer(
        modele, strategie, ratio, graine, x_entrainement, y, x_validation=None, **options
    ):
        debut = time.time()
        estimateur = construire(modele, graine, **options)
        estimateur.fit(x_entrainement, y)
        if x_validation is None:
            x_validation = matrices[modele][1]
        scores = estimateur.predict_proba(x_validation)[:, 1]
        evaluation = metriques.evaluer(y_validation, scores, montants, cout_alerte, k)
        lignes.append(
            {
                "modele": modele,
                "strategie": strategie,
                "ratio": ratio,
                "graine": graine,
                "pr_auc": evaluation["pr_auc"],
                "roc_auc": evaluation["roc_auc"],
                "precision_a_k": evaluation["precision_a_k"],
                "rappel_a_k": evaluation["rappel_a_k"],
                "cout": evaluation["cout"],
                "alertes": evaluation["alertes"],
                "duree": time.time() - debut,
            }
        )
        if verbeux:
            print(modele, strategie, graine, round(evaluation["pr_auc"], 4), flush=True)

    def retenue(nom):
        return strategies is None or nom in strategies

    for modele in modeles:
        x_entrainement = matrices[modele][0]
        graines_directes = graines if modele == "xgboost" else graines[:1]
        if retenue("aucune"):
            for graine in graines_directes:
                essayer(modele, "aucune", None, graine, x_entrainement, y_entrainement)
        if retenue("ponderation"):
            for graine in graines_directes:
                essayer(
                    modele,
                    "ponderation",
                    None,
                    graine,
                    x_entrainement,
                    y_entrainement,
                    poids_positif=negatifs / positifs,
                    ponderee=True,
                )
        for ratio in RATIOS_SOUS_ECHANTILLONNAGE:
            if not retenue(f"sous_1_{ratio}"):
                continue
            for graine in graines:
                x_res, y_res, _ = sous_echantillonner(
                    x_entrainement, y_entrainement, ratio, graine
                )
                essayer(modele, f"sous_1_{ratio}", ratio, graine, x_res, y_res)
        if retenue(f"smote_1_{RATIO_SMOTE}"):
            for graine in graines:
                x_res, y_res = sur_echantillonner_smote(
                    x_lineaire_entrainement, y_entrainement, RATIO_SMOTE, graine
                )
                essayer(
                    modele,
                    f"smote_1_{RATIO_SMOTE}",
                    RATIO_SMOTE,
                    graine,
                    x_res,
                    y_res,
                    x_validation=matrices["logistique"][1],
                )
    return pd.DataFrame(lignes)


def resumer_strategies(detail):
    colonnes = ["pr_auc", "roc_auc", "precision_a_k", "rappel_a_k", "cout"]
    groupes = detail.groupby(["modele", "strategie"], sort=False)
    moyennes = groupes[colonnes].mean()
    ecarts = groupes["pr_auc"].std().fillna(0.0).rename("pr_auc_ecart_type")
    effectifs = groupes.size().rename("tirages")
    resume = pd.concat([moyennes, ecarts, effectifs], axis=1)
    return resume.sort_values(["modele", "pr_auc"], ascending=[True, False])
