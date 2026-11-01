from src import donnees, evaluation, modeles_supervises, ressources_application, variables
from src.donnees import COLONNE_CIBLE, COLONNE_MONTANT

DOSSIER_DONNEES = modeles_supervises.RACINE / "donnees"
DOSSIER_SORTIE = modeles_supervises.RACINE / "app" / "ressources"
COUT_ALERTE = 5.0
ALERTES_PAR_JOUR = 20.0


def main():
    df = variables.construire_variables(donnees.charger())
    entrainement, validation, test = donnees.decouper_chronologiquement(df)
    modele = modeles_supervises.charger("xgboost")
    jeux_scores = evaluation.charger_scores(DOSSIER_DONNEES)
    seuils = {
        methode: evaluation.seuil_optimal_validation(jeux_scores, methode, COUT_ALERTE)["seuil"]
        for methode in ressources_application.METHODES_CURSEUR
    }
    probabilites_test = modele.probabilites(test)
    fichiers = ressources_application.ecrire(
        DOSSIER_SORTIE,
        df,
        entrainement,
        validation,
        test,
        jeux_scores,
        modele,
        probabilites_test,
        seuils,
        COUT_ALERTE,
        ALERTES_PAR_JOUR,
    )
    total = 0
    for nom in fichiers:
        taille = (DOSSIER_SORTIE / nom).stat().st_size
        total += taille
        print(f"{nom:45s} {taille / 1024:10.1f} Ko")
    print(f"{'total':45s} {total / 1024 / 1024:10.2f} Mo")
    print(seuils)


if __name__ == "__main__":
    main()
