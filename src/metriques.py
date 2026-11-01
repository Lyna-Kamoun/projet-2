import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def matrice_confusion(y, alertes):
    y = np.asarray(y).astype(bool)
    alertes = np.asarray(alertes).astype(bool)
    vp = int(np.sum(y & alertes))
    fn = int(np.sum(y & ~alertes))
    fp = int(np.sum(~y & alertes))
    vn = int(np.sum(~y & ~alertes))
    return vp, fn, fp, vn


def diviser(numerateur, denominateur):
    return float(numerateur / denominateur) if denominateur > 0 else 0.0


def fbeta(precision, rappel, beta):
    b2 = beta**2
    return diviser((1 + b2) * precision * rappel, b2 * precision + rappel)


def mcc(vp, fn, fp, vn):
    denominateur = np.sqrt(
        float(vp + fp) * float(vp + fn) * float(vn + fp) * float(vn + fn)
    )
    return diviser(float(vp) * vn - float(fp) * fn, denominateur)


def indicateurs_confusion(vp, fn, fp, vn):
    precision = diviser(vp, vp + fp)
    rappel = diviser(vp, vp + fn)
    return {
        "vp": vp,
        "fn": fn,
        "fp": fp,
        "vn": vn,
        "alertes": vp + fp,
        "accuracy": diviser(vp + vn, vp + fn + fp + vn),
        "precision": precision,
        "rappel": rappel,
        "f1": fbeta(precision, rappel, 1.0),
        "f2": fbeta(precision, rappel, 2.0),
        "mcc": mcc(vp, fn, fp, vn),
    }


def indicateurs_au_seuil(y, scores, seuil):
    alertes = np.asarray(scores) >= seuil
    return indicateurs_confusion(*matrice_confusion(y, alertes))


def ordre_decroissant(scores):
    return np.argsort(-np.asarray(scores, dtype="float64"), kind="stable")


def precision_rappel_au_rang(y, scores, k):
    y = np.asarray(y).astype(bool)
    k = int(min(max(k, 0), len(y)))
    if k == 0:
        return 0.0, 0.0
    retenus = ordre_decroissant(scores)[:k]
    vrais = int(y[retenus].sum())
    return diviser(vrais, k), diviser(vrais, int(y.sum()))


def precision_a_k(y, scores, k):
    return precision_rappel_au_rang(y, scores, k)[0]


def rappel_a_k(y, scores, k):
    return precision_rappel_au_rang(y, scores, k)[1]


def pr_auc(y, scores):
    return float(average_precision_score(y, scores))


def roc_auc(y, scores):
    return float(roc_auc_score(y, scores))


def cout_au_seuil(y, scores, montants, seuil, cout_alerte):
    y = np.asarray(y).astype(bool)
    scores = np.asarray(scores, dtype="float64")
    montants = np.asarray(montants, dtype="float64")
    alertes = scores >= seuil
    manquees = float(montants[y & ~alertes].sum())
    return manquees + float(cout_alerte) * int(alertes.sum())


def cout_sans_modele(y, montants):
    y = np.asarray(y).astype(bool)
    return float(np.asarray(montants, dtype="float64")[y].sum())


def courbe_cout(y, scores, montants, cout_alerte):
    y = np.asarray(y).astype(bool)
    scores = np.asarray(scores, dtype="float64")
    montants = np.asarray(montants, dtype="float64")
    ordre = ordre_decroissant(scores)
    scores_tries = scores[ordre]
    fraudes_tries = np.where(y[ordre], montants[ordre], 0.0)
    total = float(montants[y].sum())
    cumul = np.cumsum(fraudes_tries)
    derniers = np.flatnonzero(np.r_[scores_tries[1:] != scores_tries[:-1], True])
    alertes = derniers + 1
    manquees = total - cumul[derniers]
    cout = manquees + float(cout_alerte) * alertes
    seuils = scores_tries[derniers]
    tableau = pd.DataFrame({"seuil": seuils, "alertes": alertes, "cout": cout})
    sans_alerte = pd.DataFrame(
        {
            "seuil": [np.nextafter(scores_tries[0], np.inf)],
            "alertes": [0],
            "cout": [total],
        }
    )
    return pd.concat([sans_alerte, tableau], ignore_index=True)


def seuil_optimal(y, scores, montants, cout_alerte):
    courbe = courbe_cout(y, scores, montants, cout_alerte)
    meilleur = courbe.loc[courbe["cout"].idxmin()]
    return {
        "seuil": float(meilleur["seuil"]),
        "alertes": int(meilleur["alertes"]),
        "cout": float(meilleur["cout"]),
        "cout_sans_modele": cout_sans_modele(y, montants),
    }


def budget_total(alertes_par_jour, debut, fin):
    jours = (fin - debut).total_seconds() / 86400.0
    return int(round(alertes_par_jour * jours))


def evaluer(y, scores, montants, cout_alerte, k, seuil=None):
    if seuil is None:
        seuil = seuil_optimal(y, scores, montants, cout_alerte)["seuil"]
    precision_k, rappel_k = precision_rappel_au_rang(y, scores, k)
    ligne = {
        "roc_auc": roc_auc(y, scores),
        "pr_auc": pr_auc(y, scores),
        "precision_a_k": precision_k,
        "rappel_a_k": rappel_k,
        "seuil": seuil,
        "cout": cout_au_seuil(y, scores, montants, seuil, cout_alerte),
        "cout_sans_modele": cout_sans_modele(y, montants),
    }
    ligne.update(indicateurs_au_seuil(y, scores, seuil))
    return ligne
