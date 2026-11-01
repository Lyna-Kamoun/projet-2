import numpy as np
import pandas as pd

from src import metriques
from src.donnees import COLONNE_CARTE, COLONNE_CIBLE

BORNES_AGE = [0, 25, 35, 45, 55, 65, 200]
LIBELLES_AGE = ["moins de 25 ans", "25 à 34 ans", "35 à 44 ans", "45 à 54 ans", "55 à 64 ans", "65 ans et plus"]
LIBELLES_GENRE = {"F": "femmes", "M": "hommes"}


def tranche_age(age):
    return pd.cut(age, bins=BORNES_AGE, labels=LIBELLES_AGE, right=False)


def taux(numerateur, denominateur):
    return float(numerateur / denominateur) if denominateur > 0 else float("nan")


def indicateurs_par_groupe(jeu, alertes, groupes):
    y = jeu[COLONNE_CIBLE].to_numpy()
    alertes = np.asarray(alertes).astype(bool)
    table = pd.DataFrame({"groupe": np.asarray(groupes), "y": y, "alerte": alertes})
    table[COLONNE_CARTE] = jeu[COLONNE_CARTE].to_numpy()
    lignes = []
    for nom, sous in table.groupby("groupe", observed=True):
        vp, fn, fp, vn = metriques.matrice_confusion(sous["y"], sous["alerte"])
        lignes.append(
            {
                "groupe": nom,
                "transactions": len(sous),
                "cartes": int(sous[COLONNE_CARTE].nunique()),
                "fraudes": vp + fn,
                "taux_fraude": taux(vp + fn, len(sous)),
                "taux_alerte": taux(vp + fp, len(sous)),
                "rappel": taux(vp, vp + fn),
                "precision": taux(vp, vp + fp),
                "taux_fausses_alertes": taux(fp, fp + vn),
            }
        )
    return pd.DataFrame(lignes).set_index("groupe")


def ecarts(tableau):
    def rapport(colonne):
        valeurs = tableau[colonne].dropna()
        return float(valeurs.max() / valeurs.min()) if len(valeurs) > 1 and valeurs.min() > 0 else float("nan")

    def difference(colonne):
        valeurs = tableau[colonne].dropna()
        return float(valeurs.max() - valeurs.min()) if len(valeurs) > 1 else float("nan")

    return {
        "rapport_taux_alerte": rapport("taux_alerte"),
        "ecart_rappel": difference("rappel"),
        "rapport_taux_fausses_alertes": rapport("taux_fausses_alertes"),
    }


def tables_equite(jeu, probabilites, seuil):
    alertes = np.asarray(probabilites) >= seuil
    par_genre = indicateurs_par_groupe(jeu, alertes, jeu["gender"].map(LIBELLES_GENRE))
    par_age = indicateurs_par_groupe(jeu, alertes, tranche_age(jeu["age"]))
    return par_genre, par_age
