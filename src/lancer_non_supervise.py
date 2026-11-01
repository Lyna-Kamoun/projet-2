import time

import numpy as np
import pandas as pd

from src import donnees, metriques, modeles_supervises, non_supervise, variables
from src.donnees import COLONNE_CIBLE, COLONNE_DATE, COLONNE_MONTANT

DOSSIER_DONNEES = modeles_supervises.RACINE / "donnees"
DOSSIER_RESULTATS = modeles_supervises.RACINE / "resultats"
ALERTES_PAR_JOUR = 20.0
COUT_ALERTE = 5.0
TAUX_ALERTE = 0.01
TAILLE_DBSCAN = 40000
REGLES = {
    "isolation_forest": "1 % les plus atypiques",
    "lof": "1 % les plus atypiques",
    "autoencodeur": "1 % les plus atypiques",
    "one_class_svm": "hors de la frontière",
}


def entrainer_tout(entrainement, verbeux=True):
    modeles = {}
    etapes = [
        ("isolation_forest", lambda: non_supervise.entrainer_isolation_forest(entrainement)),
        ("lof", lambda: non_supervise.entrainer_lof(entrainement)),
        ("one_class_svm", lambda: non_supervise.entrainer_one_class_svm(entrainement, COLONNE_CIBLE)),
        ("autoencodeur", lambda: non_supervise.entrainer_autoencodeur(entrainement)),
    ]
    for nom, fabrique in etapes:
        debut = time.time()
        modeles[nom] = fabrique()
        if verbeux:
            print("entrainement", nom, round(time.time() - debut, 1), "s", flush=True)
    return modeles


def evaluer_jeu(nom_jeu, jeu, modeles, verbeux=True):
    y = jeu[COLONNE_CIBLE].to_numpy()
    montants = jeu[COLONNE_MONTANT].to_numpy()
    k = metriques.budget_total(
        ALERTES_PAR_JOUR, jeu[COLONNE_DATE].min(), jeu[COLONNE_DATE].max()
    )
    scores = pd.DataFrame(index=jeu.index)
    lignes = []
    for nom, modele in modeles.items():
        debut = time.time()
        valeurs = modele.scores(jeu)
        scores[nom] = valeurs
        if nom == "one_class_svm":
            seuil = float(np.nextafter(0.0, 1.0))
        else:
            seuil = non_supervise.seuil_top(valeurs, TAUX_ALERTE)
        resultat = non_supervise.evaluer_anomalie(y, valeurs, montants, COUT_ALERTE, k, seuil)
        resultat.update({"jeu": nom_jeu, "methode": nom, "regle": REGLES[nom]})
        lignes.append(resultat)
        if verbeux:
            print("evaluation", nom_jeu, nom, round(time.time() - debut, 1), "s", flush=True)
    return pd.DataFrame(lignes), scores, k


def evaluer_dbscan(jeu, x_jeu, parametres, nom_jeu, graine=0):
    rng = np.random.default_rng(graine)
    positions = np.sort(rng.choice(len(jeu), size=min(TAILLE_DBSCAN, len(jeu)), replace=False))
    echantillon = x_jeu[positions]
    y = jeu[COLONNE_CIBLE].to_numpy()[positions]
    bruit = non_supervise.bruit_dbscan(
        echantillon, parametres["epsilon"], parametres["min_points"]
    )
    indicateurs = metriques.indicateurs_confusion(*metriques.matrice_confusion(y, bruit))
    indicateurs.update(
        {
            "jeu": nom_jeu,
            "methode": "dbscan",
            "regle": "classé en bruit",
            "echantillon": len(positions),
            "epsilon": parametres["epsilon"],
            "min_points": parametres["min_points"],
        }
    )
    return indicateurs


def main():
    df = variables.construire_variables(donnees.charger())
    entrainement, validation, test = donnees.decouper_chronologiquement(df)
    modeles = entrainer_tout(entrainement)
    pipeline = modeles["isolation_forest"].pipeline
    parametres = non_supervise.parametres_dbscan(
        pipeline.transform(entrainement).to_numpy(), taille=TAILLE_DBSCAN
    )
    print("dbscan", parametres, flush=True)
    DOSSIER_DONNEES.mkdir(exist_ok=True)
    DOSSIER_RESULTATS.mkdir(exist_ok=True)
    tableaux = []
    for nom_jeu, jeu in (("validation", validation), ("test", test)):
        tableau, scores, k = evaluer_jeu(nom_jeu, jeu, modeles)
        scores.to_parquet(DOSSIER_DONNEES / f"scores_non_supervise_{nom_jeu}.parquet")
        tableaux.append(tableau)
        debut = time.time()
        ligne = evaluer_dbscan(jeu, pipeline.transform(jeu).to_numpy(), parametres, nom_jeu)
        tableaux.append(pd.DataFrame([ligne]))
        print("evaluation", nom_jeu, "dbscan", round(time.time() - debut, 1), "s", flush=True)
    resultat = pd.concat(tableaux, ignore_index=True)
    resultat.to_csv(DOSSIER_RESULTATS / "comparaison_non_supervise.csv", index=False)
    for nom in ("isolation_forest", "one_class_svm", "autoencodeur"):
        non_supervise.sauvegarder(modeles[nom], modeles_supervises.DOSSIER_MODELES)
    colonnes = ["jeu", "methode", "regle", "alertes", "precision", "rappel", "f2", "mcc"]
    colonnes_seuil = ["pr_auc", "roc_auc", "precision_a_k", "rappel_a_k", "cout", "cout_minimal"]
    print(resultat[colonnes + colonnes_seuil].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
