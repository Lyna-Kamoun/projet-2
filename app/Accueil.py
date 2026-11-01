import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

pages = [
    st.Page("vues/accueil.py", title="Accueil", url_path="accueil", default=True),
    st.Page("vues/donnees.py", title="1. Données", url_path="donnees"),
    st.Page("vues/preparation.py", title="2. Préparation", url_path="preparation"),
    st.Page("vues/modelisation.py", title="3. Modélisation", url_path="modelisation"),
    st.Page("vues/evaluation_et_seuil.py", title="4. Évaluation et seuil", url_path="evaluation"),
    st.Page("vues/explicabilite.py", title="5. Explicabilité", url_path="explicabilite"),
    st.Page("vues/api.py", title="6. API", url_path="api"),
]

st.navigation(pages).run()
