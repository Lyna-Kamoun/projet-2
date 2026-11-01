import time

import pandas as pd

from src import donnees, metriques, modeles_supervises, variables
from src.donnees import COLONNE_CIBLE, COLONNE_DATE, COLONNE_MONTANT

DOSSIER_DONNEES = modeles_supervises.RACINE / "donnees"
ALERTES_PAR_JOUR = 20.0
COUT_ALERTE = 5.0


def main():
    df = variables.construire_variables(donnees.charger())
    entrainement, validation, test = donnees.decouper_chronologiquement(df)
    parametres = modeles_supervises.charger_parametres()
    debut = time.time()
    modeles = [
        modeles_supervises.entrainer_xgboost(entrainement, parametres["xgboost"]),
        modeles_supervises.entrainer_logistique(entrainement, parametres["logistique"]["C"]),
    ]
    print("entrainement", round(time.time() - debut, 1), "s")
    DOSSIER_DONNEES.mkdir(exist_ok=True)
    lignes = []
    for nom_jeu, jeu in (("validation", validation), ("test", test)):
        sortie = jeu[[COLONNE_DATE, COLONNE_MONTANT, COLONNE_CIBLE]].copy()
        k = metriques.budget_total(
            ALERTES_PAR_JOUR, jeu[COLONNE_DATE].min(), jeu[COLONNE_DATE].max()
        )
        for modele in modeles:
            scores = modele.scores(jeu)
            sortie[f"score_{modele.nom}"] = scores
            sortie[f"proba_{modele.nom}"] = modele.probabilites(jeu)
            y = jeu[COLONNE_CIBLE].to_numpy()
            precision_k, rappel_k = metriques.precision_rappel_au_rang(y, scores, k)
            lignes.append(
                {
                    "jeu": nom_jeu,
                    "modele": modele.nom,
                    "pr_auc": metriques.pr_auc(y, scores),
                    "roc_auc": metriques.roc_auc(y, scores),
                    "precision_a_k": precision_k,
                    "rappel_a_k": rappel_k,
                    "taux_reel": float(y.mean()),
                    "proba_moyenne_corrigee": float(sortie[f"proba_{modele.nom}"].mean()),
                    "k": k,
                }
            )
        sortie.to_parquet(DOSSIER_DONNEES / f"scores_{nom_jeu}.parquet")
    for modele in modeles:
        print(modeles_supervises.sauvegarder(modele))
    print(pd.DataFrame(lignes).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
