import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api.types import is_string_dtype

SURFACE = "#fcfcfb"
TEXTE = "#0b0b0b"
TEXTE_SECONDAIRE = "#52514e"
GRILLE = "#e4e3df"
POSITIF = "#e34948"
NEGATIF = "#2a78d6"
COULEURS = {
    "xgboost": "#2a78d6",
    "logistique": "#eb6834",
    "isolation_forest": "#1baf7a",
    "one_class_svm": "#eda100",
    "autoencodeur": "#e87ba4",
    "lof": "#008300",
}
LIBELLES = {
    "xgboost": "XGBoost",
    "logistique": "Régression logistique",
    "isolation_forest": "Isolation Forest",
    "one_class_svm": "One-Class SVM",
    "autoencodeur": "Autoencodeur",
    "lof": "LOF",
    "dbscan": "DBSCAN",
}
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
BLEU = COULEURS["xgboost"]
ORANGE = COULEURS["logistique"]


def mise_en_page(figure, titre=None, hauteur=400, axe_x=None, axe_y=None, legende=True):
    figure.update_layout(
        title={"text": titre, "x": 0.0, "xanchor": "left", "font": {"size": 15, "color": TEXTE}}
        if titre
        else None,
        height=hauteur,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"color": TEXTE, "size": 13},
        margin={"l": 60, "r": 20, "t": 60 if titre else 20, "b": 50},
        separators=", ",
        showlegend=legende,
        legend={"orientation": "h", "yanchor": "top", "y": -0.2, "x": 0.0, "font": {"size": 12}},
        hoverlabel={"bgcolor": "white", "font": {"size": 12, "color": TEXTE}},
    )
    figure.update_xaxes(
        title=axe_x,
        showgrid=True,
        gridcolor=GRILLE,
        gridwidth=1,
        zeroline=False,
        linecolor=GRILLE,
        tickfont={"color": TEXTE_SECONDAIRE},
        title_font={"color": TEXTE_SECONDAIRE, "size": 12},
    )
    figure.update_yaxes(
        title=axe_y,
        showgrid=True,
        gridcolor=GRILLE,
        gridwidth=1,
        zeroline=False,
        linecolor=GRILLE,
        tickfont={"color": TEXTE_SECONDAIRE},
        title_font={"color": TEXTE_SECONDAIRE, "size": 12},
    )
    return figure


def courbes_precision_rappel(courbes, jeu, methodes, point=None, methode_point=None, taux=None):
    figure = go.Figure()
    for methode in methodes:
        sous = courbes[(courbes["jeu"] == jeu) & (courbes["methode"] == methode)]
        if sous.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=sous["rappel"],
                y=sous["precision"],
                mode="lines",
                name=LIBELLES.get(methode, methode),
                line={
                    "color": COULEURS.get(methode, TEXTE),
                    "width": 2,
                    "dash": "solid" if methode in ("xgboost", "logistique") else "dash",
                },
                hovertemplate="Rappel %{x:.3f}<br>Précision %{y:.3f}<extra>" + LIBELLES.get(methode, methode) + "</extra>",
            )
        )
    if taux is not None:
        figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[taux, taux],
                mode="lines",
                name="Hasard (taux de fraude)",
                line={"color": TEXTE_SECONDAIRE, "width": 1},
                hovertemplate="Taux de fraude %{y:.2%}<extra>Hasard</extra>",
            )
        )
    if point is not None:
        figure.add_trace(
            go.Scatter(
                x=[point[0]],
                y=[point[1]],
                mode="markers",
                name="Seuil choisi",
                marker={
                    "size": 12,
                    "color": COULEURS.get(methode_point, TEXTE),
                    "line": {"color": SURFACE, "width": 2},
                    "symbol": "diamond",
                },
                hovertemplate="Rappel %{x:.3f}<br>Précision %{y:.3f}<extra>Seuil choisi</extra>",
            )
        )
    figure = mise_en_page(figure, "Courbes précision-rappel", 440, "Rappel", "Précision")
    figure.update_xaxes(range=[0, 1])
    figure.update_yaxes(range=[0, 1.02])
    return figure


def courbe_cout(courbe, point=None, titre="Coût selon le nombre d'alertes par jour"):
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=courbe["alertes_par_jour"],
            y=courbe["cout"],
            mode="lines",
            name="Coût total",
            line={"color": BLEU, "width": 2},
            hovertemplate="%{x:.1f} alertes par jour<br>Coût %{y:,.0f}<extra></extra>",
        )
    )
    if point is not None:
        figure.add_trace(
            go.Scatter(
                x=[point[0]],
                y=[point[1]],
                mode="markers",
                name="Seuil choisi",
                marker={"size": 12, "color": ORANGE, "line": {"color": SURFACE, "width": 2}, "symbol": "diamond"},
                hovertemplate="%{x:.1f} alertes par jour<br>Coût %{y:,.0f}<extra>Seuil choisi</extra>",
            )
        )
    figure = mise_en_page(figure, titre, 380, "Alertes par jour", "Coût total (échelle logarithmique)")
    graduations = [g for g in (1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000) if courbe["cout"].min() / 3 <= g <= courbe["cout"].max() * 3]
    figure.update_yaxes(
        type="log",
        tickvals=graduations,
        ticktext=[f"{g:,}".replace(",", "\u202f") for g in graduations],
        minor={"showgrid": False, "ticks": ""},
    )
    return figure


def barres(df, x, y, titre, axe_y, horizontal=False, format_texte=".2%", couleur=BLEU, hauteur=380):
    donnees = df.copy()
    if horizontal:
        donnees = donnees.iloc[::-1]
    figure = go.Figure(
        go.Bar(
            x=donnees[y] if horizontal else donnees[x],
            y=donnees[x] if horizontal else donnees[y],
            orientation="h" if horizontal else "v",
            marker={"color": couleur},
            text=donnees[y],
            texttemplate="%{text:" + format_texte + "}",
            textposition="outside",
            textfont={"size": 11, "color": TEXTE},
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x:" + format_texte + "}<extra></extra>"
            if horizontal
            else "%{x}<br>%{y:" + format_texte + "}<extra></extra>",
        )
    )
    figure = mise_en_page(
        figure,
        titre,
        hauteur,
        axe_y if horizontal else None,
        None if horizontal else axe_y,
        legende=False,
    )
    figure.update_xaxes(showgrid=horizontal)
    figure.update_yaxes(showgrid=not horizontal)
    if not horizontal and is_string_dtype(df[x]):
        figure.update_xaxes(type="category", tickangle=-45)
    return figure


def ligne(df, x, y, titre, axe_y, format_axe=".1%", hauteur=340):
    figure = go.Figure(
        go.Scatter(
            x=df[x],
            y=df[y],
            mode="lines+markers",
            line={"color": BLEU, "width": 2},
            marker={"size": 7, "color": BLEU, "line": {"color": SURFACE, "width": 2}},
            hovertemplate="%{x}<br>%{y:" + format_axe + "}<extra></extra>",
        )
    )
    figure = mise_en_page(figure, titre, hauteur, None, axe_y, legende=False)
    figure.update_yaxes(tickformat=format_axe, rangemode="tozero")
    if is_string_dtype(df[x]):
        figure.update_xaxes(type="category", tickangle=-45)
    return figure


def histogramme_montants(histogramme):
    positions = np.arange(len(histogramme))
    normales = histogramme["normales"] / histogramme["normales"].sum()
    fraudes = histogramme["fraudes"] / histogramme["fraudes"].sum()
    figure = go.Figure()
    for nom, valeurs, couleur, effectif in (
        ("Transactions normales", normales, BLEU, histogramme["normales"]),
        ("Fraudes", fraudes, POSITIF, histogramme["fraudes"]),
    ):
        figure.add_trace(
            go.Bar(
                x=positions,
                y=valeurs,
                name=nom,
                marker={"color": couleur},
                customdata=np.c_[histogramme["borne_basse"], histogramme["borne_haute"], effectif],
                hovertemplate="De %{customdata[0]:,.0f} à %{customdata[1]:,.0f}<br>%{y:.1%} de la classe (%{customdata[2]:,.0f} transactions)<extra>"
                + nom
                + "</extra>",
            )
        )
    graduations = list(range(0, len(histogramme), 4))
    figure = mise_en_page(
        figure,
        "Distribution des montants par classe (part de chaque classe)",
        400,
        "Montant (classes de largeur croissante)",
        "Part de la classe",
    )
    figure.update_layout(barmode="group", bargap=0.2, bargroupgap=0.05)
    figure.update_xaxes(
        tickvals=graduations,
        ticktext=[f"{histogramme['borne_basse'].iloc[i]:,.0f}".replace(",", "\u202f") for i in graduations],
    )
    figure.update_yaxes(tickformat=".0%")
    return figure


def importance(serie, titre, axe_x, format_texte=".3f"):
    donnees = serie.sort_values(ascending=True)
    noms = [NOMS_VARIABLES.get(n, n) for n in donnees.index]
    figure = go.Figure(
        go.Bar(
            x=donnees.to_numpy(),
            y=noms,
            orientation="h",
            marker={"color": BLEU},
            text=donnees.to_numpy(),
            texttemplate="%{text:" + format_texte + "}",
            textposition="outside",
            textfont={"size": 11, "color": TEXTE},
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x:" + format_texte + "}<extra></extra>",
        )
    )
    figure = mise_en_page(figure, titre, 400, axe_x, None, legende=False)
    figure.update_yaxes(showgrid=False)
    figure.update_xaxes(range=[0, float(donnees.max()) * 1.15])
    return figure


def cascade(tableau, maximum=8, titre="Contribution de chaque variable à l'alerte (log-odds)"):
    base = tableau.attrs["base"]
    retenues = tableau.head(maximum)
    lignes = [(r.libelle, r.contribution) for r in retenues.itertuples()]
    if len(tableau) > maximum:
        lignes.append((f"{len(tableau) - maximum} autres variables", float(tableau.iloc[maximum:]["contribution"].sum())))
    lignes = lignes[::-1]
    debuts = []
    cumul = base
    for _, valeur in lignes:
        debuts.append(cumul)
        cumul += valeur
    figure = go.Figure(
        go.Bar(
            x=[valeur for _, valeur in lignes],
            y=[nom for nom, _ in lignes],
            base=debuts,
            orientation="h",
            marker={"color": [POSITIF if v > 0 else NEGATIF for _, v in lignes]},
            text=[f"{v:+.2f}".replace(".", ",") for _, v in lignes],
            textposition="outside",
            textfont={"size": 11, "color": TEXTE},
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x:+.3f} log-odds<extra></extra>",
        )
    )
    figure.add_vline(x=base, line={"color": TEXTE_SECONDAIRE, "width": 1})
    figure.add_vline(x=cumul, line={"color": TEXTE_SECONDAIRE, "width": 1})
    for position, texte in ((base, f"valeur moyenne {base:+.2f}"), (cumul, f"prédiction {cumul:+.2f}")):
        figure.add_annotation(
            x=position,
            y=1.0,
            yref="paper",
            yshift=14,
            text=texte.replace(".", ","),
            showarrow=False,
            font={"size": 12, "color": TEXTE_SECONDAIRE},
        )
    figure = mise_en_page(figure, titre, 460, "Log-odds de fraude", None, legende=False)
    figure.update_layout(margin={"l": 20, "r": 20, "t": 100, "b": 50})
    figure.update_yaxes(showgrid=False, automargin=True)
    bornes = [base, cumul] + [d + v for d, (_, v) in zip(debuts, lignes)] + debuts
    etendue = max(bornes) - min(bornes)
    figure.update_xaxes(range=[min(bornes) - 0.06 * etendue, max(bornes) + 0.14 * etendue])
    return figure


def comparaison_strategies(resume, modele):
    donnees = resume[resume["modele"] == modele].sort_values("pr_auc", ascending=True)
    bas = float((donnees["pr_auc"] - donnees["pr_auc_ecart_type"]).min())
    haut = float((donnees["pr_auc"] + donnees["pr_auc_ecart_type"]).max())
    marge = max((haut - bas) * 0.12, 0.005)
    figure = go.Figure(
        go.Scatter(
            x=donnees["pr_auc"],
            y=donnees["libelle"],
            mode="markers+text",
            marker={"color": BLEU, "size": 11, "line": {"color": SURFACE, "width": 2}},
            error_x={"type": "data", "array": donnees["pr_auc_ecart_type"], "color": TEXTE_SECONDAIRE, "thickness": 1.5, "width": 4},
            text=[f"{v:.3f}".replace(".", ",") for v in donnees["pr_auc"]],
            textposition="top center",
            textfont={"size": 11, "color": TEXTE},
            customdata=donnees["tirages"],
            hovertemplate="%{y}<br>PR-AUC %{x:.4f}<br>%{customdata} tirage(s)<extra></extra>",
        )
    )
    figure = mise_en_page(
        figure,
        f"PR-AUC en validation par stratégie, {LIBELLES.get(modele, modele)}",
        380,
        "PR-AUC moyenne (barres d'erreur : écart-type entre tirages)",
        None,
        legende=False,
    )
    figure.update_yaxes(showgrid=True)
    figure.update_xaxes(range=[bas - marge, haut + marge])
    return figure


def comparaison_methodes(tableau, colonne, titre, axe_x, format_texte=".3f"):
    figure = go.Figure()
    ordre = tableau.sort_values(colonne, ascending=True)
    figure.add_trace(
        go.Bar(
            x=ordre[colonne],
            y=[LIBELLES.get(m, m) for m in ordre["methode"]],
            orientation="h",
            marker={"color": [COULEURS.get(m, TEXTE_SECONDAIRE) for m in ordre["methode"]]},
            text=ordre[colonne],
            texttemplate="%{text:" + format_texte + "}",
            textposition="outside",
            textfont={"size": 11, "color": TEXTE},
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x:" + format_texte + "}<extra></extra>",
        )
    )
    figure = mise_en_page(figure, titre, 360, axe_x, None, legende=False)
    figure.update_yaxes(showgrid=False)
    figure.update_xaxes(range=[0, float(ordre[colonne].max()) * 1.18])
    return figure
