import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression

from src import desequilibre, preparation, reglage

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_MODELES = RACINE / "modeles"
DOSSIER_RESULTATS = RACINE / "resultats"
FICHIER_PARAMETRES = DOSSIER_RESULTATS / "meilleurs_parametres.json"
NOMS_FICHIERS = {
    "xgboost": "xgboost_fraude.joblib",
    "logistique": "logistique_fraude.joblib",
}


class ModeleFraude:
    def __init__(self, nom, pipeline, estimateur, beta=1.0, parametres=None):
        self.nom = nom
        self.pipeline = pipeline
        self.estimateur = estimateur
        self.beta = beta
        self.parametres = parametres or {}

    def matrice(self, df):
        return self.pipeline.transform(df)

    def scores(self, df):
        return self.estimateur.predict_proba(self.matrice(df))[:, 1]

    def probabilites(self, df):
        return desequilibre.corriger_probabilites(self.scores(df), self.beta)


def charger_parametres(chemin=FICHIER_PARAMETRES):
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def entrainer_xgboost(
    entrainement, parametres=None, graine=0, n_jobs=-1, optionnelles=False, nom="xgboost"
):
    parametres = dict(parametres or desequilibre.PARAMETRES_XGBOOST)
    pipeline = preparation.pipeline_arbres(optionnelles)
    x = pipeline.fit_transform(entrainement)
    y = preparation.cible(entrainement)
    x_res, y_res, beta = desequilibre.sous_echantillonner(
        x, y, reglage.RATIO_SOUS_ECHANTILLONNAGE, graine
    )
    estimateur = desequilibre.modele_xgboost(graine, 1.0, n_jobs, parametres)
    estimateur.fit(x_res, y_res)
    return ModeleFraude(nom, pipeline, estimateur, beta, parametres)


def entrainer_logistique(entrainement, c=1.0, graine=0):
    pipeline = preparation.pipeline_lineaire()
    x = pipeline.fit_transform(entrainement)
    y = preparation.cible(entrainement)
    estimateur = LogisticRegression(C=c, max_iter=2000, random_state=graine)
    estimateur.fit(x, y)
    return ModeleFraude("logistique", pipeline, estimateur, 1.0, {"C": c})


def sauvegarder(modele, dossier=DOSSIER_MODELES):
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / NOMS_FICHIERS[modele.nom]
    joblib.dump(modele, chemin, compress=3)
    return chemin


def charger(nom, dossier=DOSSIER_MODELES):
    return joblib.load(Path(dossier) / NOMS_FICHIERS[nom])
