import sys
from pathlib import Path

import pandas as pd

from src import desequilibre, donnees, variables

DOSSIER_RESULTATS = Path(__file__).resolve().parent.parent / "resultats"
FICHIER_DETAIL = DOSSIER_RESULTATS / "comparaison_desequilibre_detail.csv"
FICHIER_RESUME = DOSSIER_RESULTATS / "comparaison_desequilibre.csv"


def preparer_jeux():
    df = variables.construire_variables(donnees.charger())
    entrainement, validation, _ = donnees.decouper_chronologiquement(df)
    return entrainement, validation


def enregistrer(detail):
    DOSSIER_RESULTATS.mkdir(exist_ok=True)
    detail.to_csv(FICHIER_DETAIL, index=False)
    resume = desequilibre.resumer_strategies(detail)
    resume.to_csv(FICHIER_RESUME)
    print(resume.round(4).to_string())


def lancer_tout():
    entrainement, validation = preparer_jeux()
    detail = desequilibre.comparer_strategies(entrainement, validation, verbeux=True)
    enregistrer(detail)


def completer():
    entrainement, validation = preparer_jeux()
    existant = pd.read_csv(FICHIER_DETAIL)
    nouveau = desequilibre.comparer_strategies(
        entrainement,
        validation,
        graines=(1, 2),
        verbeux=True,
        modeles=("xgboost",),
        strategies=("aucune", "ponderation"),
    )
    detail = pd.concat([existant, nouveau], ignore_index=True)
    detail = detail.drop_duplicates(subset=["modele", "strategie", "graine"], keep="last")
    enregistrer(detail)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "completer":
        completer()
    else:
        lancer_tout()
