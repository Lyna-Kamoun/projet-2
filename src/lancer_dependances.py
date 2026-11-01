from src import donnees, explicabilite, figures, modeles_supervises, variables

DOSSIER_FIGURES = modeles_supervises.RACINE / "resultats" / "figures"
TAILLE_ECHANTILLON = 20000
VARIABLES = ("amt", "ratio_moy_30j", "z_montant_30j", "delai_precedent")


def main():
    df = variables.construire_variables(donnees.charger())
    _, validation, _ = donnees.decouper_chronologiquement(df)
    modele = modeles_supervises.charger("xgboost")
    echantillon = validation.sample(TAILLE_ECHANTILLON, random_state=0).sort_index()
    resultat = explicabilite.calculer_shap(modele, echantillon)
    for variable in VARIABLES:
        chemin = figures.figure_dependance(resultat, variable, DOSSIER_FIGURES / f"dependance_{variable}.png")
        print(chemin)


if __name__ == "__main__":
    main()
