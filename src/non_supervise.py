import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.svm import OneClassSVM

from src import metriques, preparation

ARBRES = 100
ECHANTILLON_ARBRE = 256
VOISINS_LOF = 20
NU_SVM = 0.01
COUCHES_AUTOENCODEUR = (16, 8, 16)
METHODES = ("isolation_forest", "lof", "one_class_svm", "autoencodeur")


class ModeleAnomalie:
    def __init__(self, nom, pipeline, estimateur, parametres=None):
        self.nom = nom
        self.pipeline = pipeline
        self.estimateur = estimateur
        self.parametres = parametres or {}

    def matrice(self, df):
        return self.pipeline.transform(df)

    def scores_matrice(self, x):
        x = np.asarray(x)
        if self.nom == "isolation_forest":
            return -self.estimateur.score_samples(x)
        if self.nom == "lof":
            return -self.estimateur.score_samples(x)
        if self.nom == "one_class_svm":
            return -self.estimateur.decision_function(x)
        if self.nom == "autoencodeur":
            reconstruit = self.estimateur.predict(x)
            return np.mean((x - reconstruit) ** 2, axis=1)
        raise ValueError(self.nom)

    def scores(self, df):
        return self.scores_matrice(self.matrice(df))


def echantillonner(x, taille, graine):
    if taille is None or taille >= len(x):
        return x
    rng = np.random.default_rng(graine)
    positions = np.sort(rng.choice(len(x), size=taille, replace=False))
    return x.iloc[positions] if hasattr(x, "iloc") else x[positions]


def preparer(entrainement):
    pipeline = preparation.pipeline_lineaire()
    return pipeline, pipeline.fit_transform(entrainement)


def entrainer_isolation_forest(entrainement, graine=0, n_jobs=-1):
    pipeline, x = preparer(entrainement)
    estimateur = IsolationForest(
        n_estimators=ARBRES,
        max_samples=ECHANTILLON_ARBRE,
        random_state=graine,
        n_jobs=n_jobs,
    ).fit(np.asarray(x))
    return ModeleAnomalie("isolation_forest", pipeline, estimateur)


def entrainer_lof(entrainement, taille=50000, graine=0, n_jobs=-1):
    pipeline, x = preparer(entrainement)
    reference = echantillonner(x, taille, graine)
    estimateur = LocalOutlierFactor(
        n_neighbors=VOISINS_LOF, novelty=True, n_jobs=n_jobs
    ).fit(np.asarray(reference))
    return ModeleAnomalie("lof", pipeline, estimateur, {"taille": len(reference)})


def entrainer_one_class_svm(entrainement, colonne_cible, taille=20000, graine=0):
    pipeline, x = preparer(entrainement)
    normales = x[entrainement[colonne_cible].to_numpy() == 0]
    reference = echantillonner(normales, taille, graine)
    estimateur = OneClassSVM(kernel="rbf", gamma="scale", nu=NU_SVM).fit(
        np.asarray(reference)
    )
    return ModeleAnomalie(
        "one_class_svm", pipeline, estimateur, {"taille": len(reference), "nu": NU_SVM}
    )


def entrainer_autoencodeur(entrainement, taille=200000, graine=0):
    pipeline, x = preparer(entrainement)
    reference = echantillonner(x, taille, graine)
    reseau = MLPRegressor(
        hidden_layer_sizes=COUCHES_AUTOENCODEUR,
        activation="relu",
        solver="adam",
        batch_size=1024,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
        max_iter=100,
        random_state=graine,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        reseau.fit(np.asarray(reference), np.asarray(reference))
    return ModeleAnomalie(
        "autoencodeur", pipeline, reseau, {"taille": len(reference)}
    )


def distances_au_kieme_voisin(x, k):
    x = np.asarray(x)
    voisins = NearestNeighbors(n_neighbors=k + 1).fit(x)
    distances, _ = voisins.kneighbors(x)
    return np.sort(distances[:, k])


def epsilon_au_coude(distances_triees):
    d = np.asarray(distances_triees, dtype="float64")
    abscisse = np.linspace(0.0, 1.0, len(d))
    amplitude = d[-1] - d[0]
    if amplitude <= 0:
        return float(d[0]), 0
    ordonnee = (d - d[0]) / amplitude
    position = int(np.argmax(abscisse - ordonnee))
    return float(d[position]), position


def parametres_dbscan(x, graine=0, taille=20000):
    dimension = x.shape[1]
    min_points = 2 * dimension
    echantillon = echantillonner(x, taille, graine)
    distances = distances_au_kieme_voisin(echantillon, min_points - 1)
    epsilon, position = epsilon_au_coude(distances)
    return {
        "epsilon": epsilon,
        "min_points": min_points,
        "position_du_coude": position,
        "echantillon": len(echantillon),
    }


def bruit_dbscan(x, epsilon, min_points, n_jobs=-1):
    etiquettes = DBSCAN(eps=epsilon, min_samples=min_points, n_jobs=n_jobs).fit_predict(
        np.asarray(x)
    )
    return etiquettes == -1


def seuil_top(scores, taux):
    scores = np.asarray(scores, dtype="float64")
    n = max(int(round(len(scores) * taux)), 1)
    return float(np.sort(scores)[-n])


def evaluer_anomalie(y, scores, montants, cout_alerte, k, seuil):
    resultat = metriques.evaluer(y, scores, montants, cout_alerte, k, seuil=seuil)
    resultat["cout_minimal"] = metriques.seuil_optimal(
        y, scores, montants, cout_alerte
    )["cout"]
    return resultat


def sauvegarder(modele, dossier):
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / f"{modele.nom}.joblib"
    joblib.dump(modele, chemin, compress=3)
    return chemin
