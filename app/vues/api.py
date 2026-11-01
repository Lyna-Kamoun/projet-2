import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import composants as c

c.configurer("API")

import numpy as np
import pandas as pd
import streamlit as st

from src import application, client_api, explicabilite, graphiques

c.entete("6. API de scoring et d'explication", "Le modèle retenu est exposé par un service FastAPI ; cette page l'interroge comme le ferait un autre système.")

parametres = c.parametres()
seuil_local = parametres["seuils"]["xgboost"]

st.markdown("**Points d'accès**")
st.dataframe(
    pd.DataFrame(
        {
            "Méthode": ["GET", "GET", "POST", "POST"],
            "Chemin": ["/sante", "/modele", "/scorer", "/expliquer"],
            "Rôle": [
                "Vérifier que le service tourne",
                "Seuil, coût d'analyse, catégories et variables du modèle",
                "Probabilité de fraude et décision d'alerte pour une transaction",
                "Probabilité, décision, codes motifs (3 à 5 variables) et contribution SHAP de chaque variable",
            ],
        }
    ),
    hide_index=True,
)
st.caption("La documentation interactive du service est à l'adresse /docs. Les données du service sont synthétiques, comme celles de l'application.")

st.subheader("Connexion")
colonnes = st.columns([3, 2, 1])
adresse = colonnes[0].text_input("Adresse de l'API", value=client_api.adresse_par_defaut(), placeholder="http://localhost:8000", key="adresse_api")
cle = colonnes[1].text_input("Clé d'API (si le service en exige une)", value=client_api.cle_par_defaut(), type="password", key="cle_api")
colonnes[2].write("")
colonnes[2].write("")
tester = colonnes[2].button("Tester", key="tester_api")

connecte = False
if not adresse.strip():
    st.info("Aucune adresse renseignée : les résultats ci-dessous sont calculés localement avec le même modèle. Pour appeler le service, démarrer l'API (`python -m uvicorn api.principal:app --port 8000`) puis saisir http://localhost:8000.")
elif tester:
    try:
        with st.spinner("Appel du service. S'il était endormi, il peut mettre environ une minute à répondre."):
            sante = client_api.appeler(adresse, "GET", "/sante", cle=cle or None)
            description = client_api.appeler(adresse, "GET", "/modele", cle=cle or None)
        st.success(f"Service {sante['modele']} version {sante['version']} joignable. Seuil par défaut : {application.pourcentage(description['seuil_par_defaut'], 2)}, coût d'analyse : {application.nombre(description['cout_alerte'])}.")
    except client_api.ErreurAPI as erreur:
        st.error(str(erreur))

st.subheader("Interroger le service")
gauche, milieu, droite = st.columns(3)
jour = gauche.date_input("Date", pd.Timestamp("2020-10-15"), key="api_date")
heure = gauche.slider("Heure", 0, 23, 23, key="api_heure")
montant = milieu.number_input("Montant", min_value=1.0, max_value=30000.0, value=950.0, step=10.0, key="api_montant")
categorie = milieu.selectbox("Catégorie", application.CATEGORIES, index=application.CATEGORIES.index("shopping_net"), key="api_categorie")
distance = droite.number_input("Distance porteur-marchand (km)", min_value=0.0, max_value=200.0, value=78.0, step=1.0, key="api_distance")
sans_historique = st.checkbox("Aucun historique pour cette carte (première transaction)", key="api_sans_historique")
historique = st.columns(5)
nb_tx_1h = historique[0].number_input("Transactions sur 1 h", min_value=0, max_value=20, value=0, key="api_tx_1h", disabled=sans_historique)
nb_tx_24h = historique[1].number_input("Transactions sur 24 h", min_value=0, max_value=60, value=3, key="api_tx_24h", disabled=sans_historique)
ratio = historique[2].number_input("Montant / moyenne 30 jours", min_value=0.0, max_value=200.0, value=1.0, step=0.1, key="api_ratio", disabled=sans_historique)
delai = historique[3].number_input("Délai depuis la précédente (heures)", min_value=0.0, max_value=1000.0, value=12.0, step=1.0, key="api_delai", disabled=sans_historique)
ecart = historique[4].number_input("Écart au montant habituel (écarts-types)", min_value=-50.0, max_value=200.0, value=0.0, step=0.5, key="api_ecart", disabled=sans_historique)
seuil_choisi = st.number_input(
    "Seuil de probabilité (laisser à 0 pour le seuil par défaut du service)",
    min_value=0.0,
    max_value=0.99,
    value=0.0,
    step=0.01,
    format="%.3f",
    key="api_seuil",
)

horodatage = pd.Timestamp(jour) + pd.Timedelta(hours=int(heure))
charge = client_api.charge_transaction(
    horodatage,
    montant,
    categorie,
    0 if sans_historique else nb_tx_1h,
    0 if sans_historique else nb_tx_24h,
    None if sans_historique else ratio,
    None if sans_historique else delai * 3600.0,
    None if sans_historique else ecart,
    distance,
)

local = application.transaction_personnalisee(
    horodatage,
    montant,
    categorie,
    0 if sans_historique else nb_tx_1h,
    0 if sans_historique else nb_tx_24h,
    np.nan if sans_historique else ratio,
    np.nan if sans_historique else delai * 3600.0,
    np.nan if sans_historique else ecart,
    distance,
)
modele = c.modele("xgboost")

reponse = None
if adresse.strip():
    options = {"motifs": 5}
    if seuil_choisi > 0:
        options["seuil"] = seuil_choisi
    try:
        with st.spinner("Appel du service. S'il était endormi, il peut mettre environ une minute à répondre."):
            reponse = client_api.appeler(adresse, "POST", "/expliquer", charge=charge, parametres=options, cle=cle or None)
        origine = "service"
    except client_api.ErreurAPI as erreur:
        st.error(str(erreur))
else:
    origine = "local"
    tableau_local = explicabilite.cascade(modele, local, 0)
    seuil_applique = seuil_choisi if seuil_choisi > 0 else seuil_local
    probabilite_locale = float(modele.probabilites(local)[0])
    reponse = {
        "probabilite": probabilite_locale,
        "alerte": probabilite_locale >= seuil_applique,
        "seuil": seuil_applique,
        "motifs": tableau_local[tableau_local["contribution"] > 0].head(5)[["code", "libelle", "contribution"]].to_dict("records"),
        "contributions": tableau_local[["code", "libelle", "contribution"]].to_dict("records"),
        "valeur_moyenne": tableau_local.attrs["base"],
        "prediction": tableau_local.attrs["marge"],
    }

if reponse is not None:
    st.caption("Réponse du service." if origine == "service" else "Calcul local (aucune adresse d'API renseignée).")
    metriques = st.columns(3)
    metriques[0].metric("Probabilité de fraude", application.pourcentage(reponse["probabilite"], 1))
    metriques[1].metric("Décision", "Alerte" if reponse["alerte"] else "Pas d'alerte")
    metriques[2].metric("Seuil appliqué", application.pourcentage(reponse["seuil"], 2))
    if origine == "service":
        ecart_local = abs(float(modele.probabilites(local)[0]) - reponse["probabilite"])
        st.caption(f"Écart avec le calcul local du même modèle : {application.nombre(ecart_local, 6)}.")
    if reponse["motifs"]:
        motifs = pd.DataFrame(reponse["motifs"])
        motifs["contribution"] = motifs["contribution"].map(lambda v: application.nombre(v, 2))
        motifs.columns = ["Code motif", "Motif", "Contribution (log-odds)"]
        st.markdown("**Codes motifs** (variables qui poussent le plus vers l'alerte)")
        st.dataframe(motifs, hide_index=True)
    st.plotly_chart(graphiques.cascade(client_api.tableau_contributions(reponse)))
    with st.expander("Réponse brute"):
        st.json(reponse)

st.subheader("Appel depuis un programme")
adresse_exemple = client_api.normaliser_adresse(adresse) or "http://localhost:8000"
exemple_curl = f"""curl -X POST "{adresse_exemple}/expliquer?motifs=5" \\
  -H "Content-Type: application/json" \\
  -d '{{"horodatage": "2020-10-15T23:10:00", "montant": 950, "categorie": "shopping_net", "nb_tx_24h": 3, "ratio_moy_30j": 1.0, "delai_precedent_secondes": 43200, "z_montant_30j": 0.0, "distance_km": 78}}'"""
st.code(exemple_curl, language="bash")
exemple_python = f"""import httpx

transaction = {{
    "horodatage": "2020-10-15T23:10:00",
    "montant": 950,
    "categorie": "shopping_net",
    "nb_tx_24h": 3,
    "ratio_moy_30j": 1.0,
    "delai_precedent_secondes": 43200,
    "z_montant_30j": 0.0,
    "distance_km": 78,
}}
reponse = httpx.post("{adresse_exemple}/expliquer", json=transaction)
print(reponse.json()["motifs"])"""
st.code(exemple_python, language="python")
