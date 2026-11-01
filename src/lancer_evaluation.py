import pandas as pd

from src import evaluation, figures, modeles_supervises

DOSSIER_DONNEES = modeles_supervises.RACINE / "donnees"
DOSSIER_RESULTATS = modeles_supervises.RACINE / "resultats"
DOSSIER_FIGURES = DOSSIER_RESULTATS / "figures"


def main():
    jeux = evaluation.charger_scores(DOSSIER_DONNEES)
    DOSSIER_RESULTATS.mkdir(exist_ok=True)
    tableau = evaluation.tableau_comparatif(jeux)
    tableau.to_csv(DOSSIER_RESULTATS / "evaluation_comparative.csv", index=False)
    sensibilite = pd.concat(
        [evaluation.sensibilite_cout(jeux, m) for m in ("xgboost", "logistique", "isolation_forest")],
        ignore_index=True,
    )
    sensibilite.to_csv(DOSSIER_RESULTATS / "sensibilite_cout.csv", index=False)
    budgets = evaluation.selon_budget(jeux)
    budgets.to_csv(DOSSIER_RESULTATS / "evaluation_selon_budget.csv", index=False)
    complementarite = pd.DataFrame(
        [
            evaluation.complementarite(jeux[nom], "xgboost", appoint)
            for nom in ("validation", "test")
            for appoint in ("isolation_forest", "one_class_svm", "autoencodeur")
        ]
    )
    complementarite.insert(0, "jeu", ["validation"] * 3 + ["test"] * 3)
    complementarite.to_csv(DOSSIER_RESULTATS / "complementarite.csv", index=False)
    figures.figure_courbes_precision_rappel(jeux, DOSSIER_FIGURES / "courbes_precision_rappel.png")
    figures.figure_cout_selon_alertes(jeux, DOSSIER_FIGURES / "cout_selon_alertes.png")
    colonnes = [
        "methode", "jeu", "pr_auc", "roc_auc", "precision_a_k", "rappel_a_k", "alertes",
        "precision", "rappel", "f2", "mcc", "cout", "economie",
    ]
    print(tableau[colonnes].round(4).to_string(index=False))
    print(sensibilite.round(4).to_string(index=False))
    print(budgets[budgets["methode"].isin(["xgboost", "isolation_forest"])].round(4).to_string(index=False))
    print(complementarite.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
