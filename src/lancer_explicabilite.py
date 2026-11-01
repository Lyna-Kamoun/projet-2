import json

import numpy as np
import pandas as pd

from src import donnees, explicabilite, figures, metriques, modeles_supervises, variables
from src.donnees import COLONNE_CIBLE, COLONNE_DATE, COLONNE_MONTANT

DOSSIER_RESULTATS = modeles_supervises.RACINE / "resultats"
DOSSIER_FIGURES = DOSSIER_RESULTATS / "figures"
COUT_ALERTE = 5.0
TAILLE_ECHANTILLON = 20000


def choisir_exemples(test, probabilites, seuil):
    y = test[COLONNE_CIBLE].to_numpy()
    alertes = probabilites >= seuil
    vrais = np.flatnonzero((y == 1) & alertes)
    faux = np.flatnonzero((y == 0) & alertes)
    manques = np.flatnonzero((y == 1) & ~alertes)
    milieu = vrais[np.argsort(probabilites[vrais])][len(vrais) // 2]
    return {
        "vrai_positif": int(milieu),
        "faux_positif": int(faux[np.argmax(probabilites[faux])]),
        "fraude_manquee": int(manques[np.argmax(test[COLONNE_MONTANT].to_numpy()[manques])]),
    }


def main():
    df = variables.construire_variables(donnees.charger())
    _, validation, test = donnees.decouper_chronologiquement(df)
    modele = modeles_supervises.charger("xgboost")
    probabilites_validation = modele.probabilites(validation)
    seuil = metriques.seuil_optimal(
        validation[COLONNE_CIBLE].to_numpy(),
        probabilites_validation,
        validation[COLONNE_MONTANT].to_numpy(),
        COUT_ALERTE,
    )["seuil"]
    print("seuil", seuil)

    echantillon = validation.sample(TAILLE_ECHANTILLON, random_state=0).sort_index()
    resultat = explicabilite.calculer_shap(modele, echantillon)
    population = explicabilite.importance_shap(resultat.groupees())
    alertes = validation[probabilites_validation >= seuil]
    resultat_alertes = explicabilite.calculer_shap(modele, alertes)
    sur_alertes = explicabilite.importance_shap(resultat_alertes.groupees())
    permutation = explicabilite.importance_permutation(modele, validation, repetitions=3)

    DOSSIER_RESULTATS.mkdir(exist_ok=True)
    population.to_csv(DOSSIER_RESULTATS / "importance_shap_population.csv")
    sur_alertes.to_csv(DOSSIER_RESULTATS / "importance_shap_alertes.csv")
    permutation.to_csv(DOSSIER_RESULTATS / "importance_permutation.csv")
    figures.figure_importance_shap(population, DOSSIER_FIGURES / "importance_shap.png")
    figures.figure_importance_permutation(permutation, DOSSIER_FIGURES / "importance_permutation.png")
    figures.figure_nuage_abeilles(resultat, DOSSIER_FIGURES / "nuage_abeilles.png")

    probabilites_test = modele.probabilites(test)
    exemples = choisir_exemples(test, probabilites_test, seuil)
    titres = {
        "vrai_positif": "Fraude détectée",
        "faux_positif": "Fausse alerte",
        "fraude_manquee": "Fraude manquée",
    }
    sortie = {}
    for nom, position in exemples.items():
        resultat_ligne = explicabilite.calculer_shap(modele, test.iloc[[position]])
        tableau = explicabilite.cascade(modele, test, position, resultat_ligne)
        figures.figure_cascade(tableau, DOSSIER_FIGURES / f"cascade_{nom}.png", titres[nom])
        ligne = test.iloc[position]
        sortie[nom] = {
            "date": str(ligne[COLONNE_DATE]),
            "montant": float(ligne[COLONNE_MONTANT]),
            "categorie": ligne["category"],
            "fraude": int(ligne[COLONNE_CIBLE]),
            "probabilite_corrigee": round(float(probabilites_test[position]), 4),
            "alerte": bool(probabilites_test[position] >= seuil),
            "codes_motifs": [
                {"code": m["code"], "libelle": m["libelle"], "contribution": round(m["contribution"], 3)}
                for m in explicabilite.codes_motifs(modele, test, position, 5, resultat_ligne)
            ],
        }
    with open(DOSSIER_RESULTATS / "explications_exemples.json", "w", encoding="utf-8") as f:
        json.dump(sortie, f, ensure_ascii=False, indent=2)

    print(population.round(4).to_string())
    print(sur_alertes.round(4).to_string())
    print(permutation.round(4).to_string())
    print(json.dumps(sortie, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
