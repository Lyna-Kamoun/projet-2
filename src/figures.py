from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import evaluation
from src.donnees import COLONNE_CIBLE

SURFACE = "#fcfcfb"
TEXTE = "#0b0b0b"
TEXTE_SECONDAIRE = "#52514e"
GRILLE = "#e4e3df"
COULEURS = {
    "xgboost": "#2a78d6",
    "logistique": "#eb6834",
    "isolation_forest": "#1baf7a",
    "one_class_svm": "#eda100",
    "autoencodeur": "#e87ba4",
    "lof": "#008300",
}
EPAISSEUR = 1.44


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        ax.spines[cote].set_color(GRILLE)
    ax.grid(True, color=GRILLE, linewidth=0.72, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXTE_SECONDAIRE, labelsize=8, length=0)


def figure_courbes_precision_rappel(jeux, chemin, methodes=evaluation.METHODES):
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.4), dpi=100, facecolor=SURFACE)
    from sklearn.metrics import average_precision_score

    for ax, (nom, jeu) in zip(axes, jeux.items()):
        style_axes(ax)
        for methode in methodes:
            rappel, precision = evaluation.courbe_precision_rappel(jeu, methode)
            pr_auc = average_precision_score(jeu[COLONNE_CIBLE], jeu[methode])
            supervise = methode in evaluation.METHODES_SUPERVISEES
            ax.plot(
                rappel,
                precision,
                color=COULEURS[methode],
                linewidth=EPAISSEUR,
                linestyle="-" if supervise else (0, (5, 2.5)),
                solid_capstyle="round",
                label=f"{evaluation.LIBELLES[methode]} (PR-AUC {pr_auc:.3f})".replace(".", ","),
            )
        taux = jeu[COLONNE_CIBLE].mean()
        ax.axhline(taux, color=TEXTE_SECONDAIRE, linewidth=0.72)
        ax.text(0.99, taux + 0.012, "hasard", ha="right", va="bottom", fontsize=8, color=TEXTE_SECONDAIRE)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Rappel", color=TEXTE_SECONDAIRE, fontsize=9)
        ax.set_ylabel("Précision", color=TEXTE_SECONDAIRE, fontsize=9)
        ax.set_title(
            "Validation" if nom == "validation" else "Test",
            loc="left",
            color=TEXTE,
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.17),
            ncol=2,
            frameon=False,
            fontsize=8,
            labelcolor=TEXTE,
            handlelength=2.6,
        )
    figure.suptitle(
        "Courbes précision-rappel, toutes méthodes",
        x=0.06,
        ha="left",
        color=TEXTE,
        fontsize=13,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.3, wspace=0.18)
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(chemin, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return chemin


def figure_cout_selon_alertes(jeux, chemin, methode="xgboost", cout_alerte=evaluation.COUT_ALERTE):
    figure, ax = plt.subplots(figsize=(8.4, 4.8), dpi=100, facecolor=SURFACE)
    style_axes(ax)
    couleurs = {"validation": COULEURS["xgboost"], "test": COULEURS["logistique"]}
    for nom, jeu in jeux.items():
        courbe = evaluation.courbe_cout_selon_alertes(jeu, methode, cout_alerte)
        courbe = courbe[courbe["alertes_par_jour"] <= 100]
        ax.plot(
            courbe["alertes_par_jour"],
            courbe["cout"],
            color=couleurs[nom],
            linewidth=EPAISSEUR,
            solid_capstyle="round",
            label="Validation" if nom == "validation" else "Test",
        )
    optimum = evaluation.seuil_optimal_validation(jeux, methode, cout_alerte)
    jours = (
        jeux["validation"].iloc[:, 0].max() - jeux["validation"].iloc[:, 0].min()
    ).total_seconds() / 86400.0
    abscisse = optimum["alertes"] / jours
    ax.scatter(
        [abscisse],
        [optimum["cout"]],
        s=64,
        color=couleurs["validation"],
        edgecolor=SURFACE,
        linewidth=2,
        zorder=5,
    )
    ax.annotate(
        f"coût minimal en validation\n{optimum['cout']:,.0f} pour {abscisse:.0f} alertes par jour".replace(",", " "),
        xy=(abscisse, optimum["cout"]),
        xytext=(abscisse + 14, optimum["cout"] * 2.6),
        fontsize=8,
        color=TEXTE,
        arrowprops={"arrowstyle": "-", "color": TEXTE_SECONDAIRE, "linewidth": 0.72},
    )
    ax.set_yscale("log")
    graduations = [20000, 30000, 50000, 100000, 200000, 500000, 1000000]
    ax.set_yticks(graduations)
    ax.set_yticklabels([f"{g:,}".replace(",", " ") for g in graduations])
    ax.set_ylim(15000, 1300000)
    ax.minorticks_off()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Alertes par jour", color=TEXTE_SECONDAIRE, fontsize=9)
    ax.set_ylabel("Coût total (échelle logarithmique)", color=TEXTE_SECONDAIRE, fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=TEXTE)
    ax.set_title(
        f"Coût selon le nombre d'alertes, {evaluation.LIBELLES[methode]} (coût d'analyse {cout_alerte:g} par alerte)",
        loc="left",
        color=TEXTE,
        fontsize=11,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.14)
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(chemin, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return chemin


NOMS_VARIABLES = {
    "heure": "Heure",
    "jour": "Jour de la semaine",
    "amt": "Montant",
    "nb_tx_1h": "Transactions sur 1 h",
    "nb_tx_24h": "Transactions sur 24 h",
    "ratio_moy_30j": "Montant / moyenne 30 jours",
    "delai_precedent": "Délai depuis la précédente",
    "z_montant_30j": "Écart au montant habituel",
    "distance_km": "Distance porteur-marchand",
    "category": "Catégorie d'achat",
}
POSITIF = "#e34948"
NEGATIF = "#2a78d6"


def enregistrer(figure, chemin):
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(chemin, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return chemin


def figure_barres(serie, titre, etiquette_x, chemin, format_valeur="{:.3f}"):
    serie = serie.sort_values(ascending=True)
    figure, ax = plt.subplots(figsize=(8.4, 4.8), dpi=100, facecolor=SURFACE)
    style_axes(ax)
    ax.grid(False, axis="y")
    noms = [NOMS_VARIABLES.get(n, n) for n in serie.index]
    ax.barh(noms, serie.to_numpy(), height=0.5, color=COULEURS["xgboost"])
    marge = serie.max() * 0.015
    for position, valeur in enumerate(serie.to_numpy()):
        ax.text(
            valeur + marge,
            position,
            format_valeur.format(valeur).replace(".", ","),
            va="center",
            fontsize=8,
            color=TEXTE,
        )
    ax.set_xlim(0, serie.max() * 1.15)
    ax.set_xlabel(etiquette_x, color=TEXTE_SECONDAIRE, fontsize=9)
    ax.tick_params(axis="y", labelcolor=TEXTE, labelsize=9)
    ax.set_title(titre, loc="left", color=TEXTE, fontsize=11, fontweight="bold")
    figure.subplots_adjust(left=0.3, right=0.97, top=0.9, bottom=0.14)
    return enregistrer(figure, chemin)


def figure_importance_shap(importance, chemin):
    return figure_barres(
        importance["importance"],
        "Importance globale SHAP (moyenne des valeurs absolues, en log-odds)",
        "Contribution moyenne absolue",
        chemin,
    )


def figure_importance_permutation(importance, chemin):
    return figure_barres(
        importance["perte_pr_auc"].clip(lower=0),
        "Importance par permutation (perte de PR-AUC)",
        "Perte de PR-AUC",
        chemin,
    )


def figure_nuage_abeilles(resultat, chemin, maximum=None):
    import shap

    explication = shap.Explanation(
        values=resultat.valeurs.to_numpy(),
        base_values=np.full(len(resultat.valeurs), resultat.base),
        data=resultat.matrice.to_numpy(),
        feature_names=list(resultat.valeurs.columns),
    )
    plt.figure(figsize=(9, 7.2), dpi=100, facecolor=SURFACE)
    shap.plots.beeswarm(
        explication, max_display=maximum or resultat.valeurs.shape[1], show=False
    )
    figure = plt.gcf()
    figure.set_facecolor(SURFACE)
    barre = figure.axes[-1]
    barre.set_ylabel("Valeur de la variable", color=TEXTE_SECONDAIRE, fontsize=9)
    barre.set_yticklabels(["Faible", "Élevée"])
    figure.axes[0].set_xlabel("Valeur SHAP (log-odds)", color=TEXTE_SECONDAIRE, fontsize=9)
    figure.tight_layout()
    return enregistrer(figure, chemin)


def figure_cascade(tableau, chemin, titre, maximum=8):
    base = tableau.attrs["base"]
    retenues = tableau.head(maximum)
    reste = tableau.iloc[maximum:]["contribution"].sum() if len(tableau) > maximum else 0.0
    lignes = [(r.libelle, r.contribution) for r in retenues.itertuples()]
    if len(tableau) > maximum:
        lignes.append((f"{len(tableau) - maximum} autres variables", reste))
    lignes = lignes[::-1]
    figure, ax = plt.subplots(figsize=(10, 5.2), dpi=100, facecolor=SURFACE)
    style_axes(ax)
    ax.grid(False, axis="y")
    cumul = base
    for position, (nom, valeur) in enumerate(lignes):
        ax.barh(
            position,
            valeur,
            left=cumul,
            height=0.5,
            color=POSITIF if valeur > 0 else NEGATIF,
        )
        etiquette = f"{valeur:+.2f}".replace(".", ",")
        decalage = 0.04 * (1 if valeur > 0 else -1)
        ax.text(
            cumul + valeur + decalage,
            position,
            etiquette,
            va="center",
            ha="left" if valeur > 0 else "right",
            fontsize=8,
            color=TEXTE,
        )
        cumul += valeur
    haut = len(lignes) - 0.55
    ax.vlines([base, cumul], -0.55, haut, color=TEXTE_SECONDAIRE, linewidth=0.72)
    ax.set_yticks(range(len(lignes)))
    ax.set_yticklabels([n for n, _ in lignes], fontsize=8.5, color=TEXTE)
    basses, hautes = ax.get_xlim()
    largeur = hautes - basses
    ax.set_xlim(basses - 0.14 * largeur, hautes + 0.06 * largeur)
    ax.text(base, haut + 0.1, f"valeur moyenne {base:+.2f}".replace(".", ","), ha="center", va="bottom", fontsize=8, color=TEXTE_SECONDAIRE)
    ax.text(cumul, haut + 0.1, f"prédiction {cumul:+.2f}".replace(".", ","), ha="center", va="bottom", fontsize=8, color=TEXTE_SECONDAIRE)
    ax.set_ylim(-0.7, len(lignes) + 0.4)
    ax.set_xlabel("Log-odds de fraude", color=TEXTE_SECONDAIRE, fontsize=9)
    probabilite = tableau.attrs["probabilite_corrigee"]
    ax.set_title(
        f"{titre} (probabilité corrigée {probabilite * 100:.1f} %)".replace(".", ","),
        loc="left",
        color=TEXTE,
        fontsize=11,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.4, right=0.97, top=0.88, bottom=0.17)
    return enregistrer(figure, chemin)


def figure_dependance(resultat, variable, chemin):
    import shap

    explication = shap.Explanation(
        values=resultat.valeurs.to_numpy(),
        base_values=np.full(len(resultat.valeurs), resultat.base),
        data=resultat.matrice.to_numpy(),
        feature_names=list(resultat.valeurs.columns),
    )
    plt.figure(figsize=(8.4, 5.2), dpi=100, facecolor=SURFACE)
    shap.plots.scatter(explication[:, variable], color=explication, show=False)
    figure = plt.gcf()
    figure.set_facecolor(SURFACE)
    figure.axes[0].set_ylabel(f"Valeur SHAP de {variable} (log-odds)", color=TEXTE_SECONDAIRE, fontsize=9)
    if len(figure.axes) > 1:
        figure.axes[-1].set_ylabel("Variable de couleur", color=TEXTE_SECONDAIRE, fontsize=9)
    figure.tight_layout()
    return enregistrer(figure, chemin)
