import pandas as pd

from src import donnees, equite, explicabilite, metriques, modeles_supervises, variables
from src.donnees import COLONNE_CIBLE, COLONNE_DATE, COLONNE_MONTANT

DOSSIER_RESULTATS = modeles_supervises.RACINE / "resultats"
COUT_ALERTE = 5.0
ALERTES_PAR_JOUR = 20.0
TAILLE_ECHANTILLON = 20000


def evaluer(modele, nom, jeux):
    validation = jeux["validation"]
    seuil = metriques.seuil_optimal(
        validation[COLONNE_CIBLE].to_numpy(),
        modele.probabilites(validation),
        validation[COLONNE_MONTANT].to_numpy(),
        COUT_ALERTE,
    )["seuil"]
    lignes = []
    probabilites = {}
    for jeu_nom, jeu in jeux.items():
        proba = modele.probabilites(jeu)
        probabilites[jeu_nom] = proba
        y = jeu[COLONNE_CIBLE].to_numpy()
        k = metriques.budget_total(ALERTES_PAR_JOUR, jeu[COLONNE_DATE].min(), jeu[COLONNE_DATE].max())
        precision_k, rappel_k = metriques.precision_rappel_au_rang(y, proba, k)
        ligne = {
            "modele": nom,
            "jeu": jeu_nom,
            "pr_auc": metriques.pr_auc(y, proba),
            "roc_auc": metriques.roc_auc(y, proba),
            "precision_a_k": precision_k,
            "rappel_a_k": rappel_k,
            "seuil": seuil,
        }
        ligne.update(metriques.indicateurs_au_seuil(y, proba, seuil))
        ligne["cout"] = metriques.cout_au_seuil(y, proba, jeu[COLONNE_MONTANT].to_numpy(), seuil, COUT_ALERTE)
        lignes.append(ligne)
    return lignes, probabilites, seuil


def main():
    df = variables.construire_variables(donnees.charger())
    entrainement, validation, test = donnees.decouper_chronologiquement(df)
    jeux = {"validation": validation, "test": test}
    parametres = modeles_supervises.charger_parametres()
    base = modeles_supervises.charger("xgboost")
    variante = modeles_supervises.entrainer_xgboost(
        entrainement,
        parametres["xgboost"],
        optionnelles=True,
        nom="xgboost_avec_age_genre",
    )
    lignes_base, probas_base, seuil_base = evaluer(base, "xgboost", jeux)
    lignes_variante, _, _ = evaluer(variante, "xgboost_avec_age_genre", jeux)
    comparaison = pd.DataFrame(lignes_base + lignes_variante)

    echantillon = validation.sample(min(TAILLE_ECHANTILLON, len(validation)), random_state=0).sort_index()
    shap_variante = explicabilite.calculer_shap(variante, echantillon)
    importance = explicabilite.importance_shap(shap_variante.groupees())

    DOSSIER_RESULTATS.mkdir(exist_ok=True)
    comparaison.to_csv(DOSSIER_RESULTATS / "equite_comparaison_modeles.csv", index=False)
    importance.to_csv(DOSSIER_RESULTATS / "equite_importance_variante.csv")

    tables = []
    for jeu_nom, jeu in jeux.items():
        par_genre, par_age = equite.tables_equite(jeu, probas_base[jeu_nom], seuil_base)
        for type_groupe, tableau in (("genre", par_genre), ("age", par_age)):
            sortie = tableau.reset_index()
            sortie.insert(0, "type", type_groupe)
            sortie.insert(0, "jeu", jeu_nom)
            tables.append(sortie)
            print(jeu_nom, type_groupe, equite.ecarts(tableau))
    groupes = pd.concat(tables, ignore_index=True)
    groupes.to_csv(DOSSIER_RESULTATS / "equite_groupes.csv", index=False)

    colonnes = ["modele", "jeu", "pr_auc", "roc_auc", "precision_a_k", "rappel_a_k", "alertes", "rappel", "precision", "f2", "mcc", "cout"]
    print(comparaison[colonnes].round(4).to_string(index=False))
    print(importance.round(4).to_string())
    print(groupes.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
