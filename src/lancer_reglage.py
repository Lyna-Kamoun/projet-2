import json
import sys
from pathlib import Path

import pandas as pd

from src import donnees, reglage, variables

DOSSIER_RESULTATS = Path(__file__).resolve().parent.parent / "resultats"


def decider(xgboost, logistique):
    meilleurs = reglage.meilleurs_parametres(xgboost, logistique)
    with open(DOSSIER_RESULTATS / "meilleurs_parametres.json", "w", encoding="utf-8") as f:
        json.dump(meilleurs, f, indent=2)
    return meilleurs


def decider_depuis_fichiers():
    xgboost = pd.read_csv(DOSSIER_RESULTATS / "reglage_xgboost.csv")
    logistique = pd.read_csv(DOSSIER_RESULTATS / "reglage_logistique.csv")
    print(json.dumps(decider(xgboost, logistique), indent=2))


def main():
    df = variables.construire_variables(donnees.charger())
    entrainement, _, _ = donnees.decouper_chronologiquement(df)
    DOSSIER_RESULTATS.mkdir(exist_ok=True)
    logistique = reglage.rechercher_logistique(entrainement, verbeux=True)
    logistique.to_csv(DOSSIER_RESULTATS / "reglage_logistique.csv", index=False)
    xgboost = reglage.rechercher_xgboost(entrainement, verbeux=True)
    xgboost.to_csv(DOSSIER_RESULTATS / "reglage_xgboost.csv", index=False)
    meilleurs = decider(xgboost, logistique)
    colonnes = ["reference", "pr_auc_moyenne", "pr_auc_ecart_type", "pr_auc_pli_1", "pr_auc_pli_2", "pr_auc_pli_3"]
    print(logistique[["C"] + colonnes[1:]].round(4).to_string())
    print(xgboost[colonnes + list(reglage.DISTRIBUTIONS_XGBOOST)].head(8).round(4).to_string())
    print(xgboost[xgboost["reference"]][colonnes].round(4).to_string())
    print(json.dumps(meilleurs, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "decider":
        decider_depuis_fichiers()
    else:
        main()
