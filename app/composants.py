import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

import pandas as pd
import streamlit as st

from src import application


def configurer(titre):
    st.set_page_config(page_title=f"{titre} | Détection de fraude", layout="wide")


def entete(titre, description=None):
    st.title(titre)
    if description:
        st.caption(description)


@st.cache_data(show_spinner=False)
def lire(nom, source="ressources", **options):
    if source == "ressources":
        return application.lire_csv(nom, **options)
    return application.lire_resultat(nom, **options)


@st.cache_data(show_spinner=False)
def parametres():
    return application.lire_parametres()


@st.cache_data(show_spinner=False)
def scores(jeu):
    return application.lire_scores(jeu)


@st.cache_data(show_spinner=False)
def echantillon():
    return pd.read_parquet(application.DOSSIER_RESSOURCES / "echantillon_explication.parquet")


@st.cache_resource(show_spinner=False)
def modele(nom="xgboost"):
    return application.charger_modele(nom)


def image(nom, legende=None):
    st.image(str(application.DOSSIER_FIGURES / nom), caption=legende, width="stretch")
