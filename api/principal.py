import os
from contextlib import asynccontextmanager
from typing import Annotated, Optional

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query

from api.schemas import Contribution, Decision, Explication, Modele, Sante, Transaction
from src import application, explicabilite, modeles_supervises

VERSION = "1.0.0"
DESCRIPTION = """
Service de scoring et d'explication pour la détection de fraude par carte bancaire.

Le modèle est un XGBoost entraîné sur le simulateur Sparkov. Il renvoie une probabilité de fraude, une décision d'alerte
au seuil retenu (choisi en minimisant un coût en montants) et, pour chaque alerte, les variables qui la motivent (valeurs SHAP).
Les données sont synthétiques : les performances ne se transposent pas telles quelles à des données réelles.
"""


@asynccontextmanager
async def cycle_de_vie(application_web):
    application_web.state.modele = modeles_supervises.charger("xgboost")
    application_web.state.parametres = application.lire_parametres()
    yield


app = FastAPI(title="API de détection de fraude explicable", description=DESCRIPTION, version=VERSION, lifespan=cycle_de_vie)


def verifier_cle(x_api_key: Annotated[Optional[str], Header()] = None):
    attendue = os.environ.get("CLE_API")
    if attendue and x_api_key != attendue:
        raise HTTPException(status_code=401, detail="Clé d'API absente ou invalide (en-tête X-API-Key).")


def vers_tableau(transactions):
    lignes = [
        application.transaction_personnalisee(
            t.horodatage,
            t.montant,
            t.categorie,
            t.nb_tx_1h,
            t.nb_tx_24h,
            float("nan") if t.ratio_moy_30j is None else t.ratio_moy_30j,
            float("nan") if t.delai_precedent_secondes is None else t.delai_precedent_secondes,
            float("nan") if t.z_montant_30j is None else t.z_montant_30j,
            t.distance_km,
        )
        for t in transactions
    ]
    return pd.concat(lignes, ignore_index=True)


def seuil_effectif(seuil, parametres):
    return parametres["seuils"]["xgboost"] if seuil is None else float(seuil)


def decisions(modele, tableau, seuil):
    scores = modele.scores(tableau)
    probabilites = modele.probabilites(tableau)
    return [
        Decision(probabilite=float(p), score_brut=float(s), alerte=bool(p >= seuil), seuil=float(seuil))
        for p, s in zip(probabilites, scores)
    ]


SeuilOptionnel = Annotated[Optional[float], Query(gt=0, lt=1, description="Seuil de probabilité ; par défaut celui qui minimise le coût sur la validation.")]


@app.get("/sante", response_model=Sante, tags=["Service"])
def sante():
    return Sante(statut="ok", modele="xgboost", version=VERSION)


@app.get("/modele", response_model=Modele, tags=["Service"], dependencies=[Depends(verifier_cle)])
def modele():
    parametres = app.state.parametres
    variables = [c for c in app.state.modele.matrice(vers_tableau([Transaction(horodatage="2020-01-01T00:00:00", montant=1.0, categorie=application.CATEGORIES[0])])).columns]
    return Modele(
        nom="xgboost",
        version=VERSION,
        seuil_par_defaut=parametres["seuils"]["xgboost"],
        cout_alerte=parametres["cout_alerte"],
        beta=parametres["beta"],
        categories=application.CATEGORIES,
        variables=variables,
    )


@app.post("/scorer", response_model=Decision, tags=["Scoring"], dependencies=[Depends(verifier_cle)])
def scorer(transaction: Transaction, seuil: SeuilOptionnel = None):
    seuil = seuil_effectif(seuil, app.state.parametres)
    return decisions(app.state.modele, vers_tableau([transaction]), seuil)[0]


@app.post("/expliquer", response_model=Explication, tags=["Explication"], dependencies=[Depends(verifier_cle)])
def expliquer(
    transaction: Transaction,
    seuil: SeuilOptionnel = None,
    motifs: Annotated[int, Query(ge=3, le=5, description="Nombre maximal de codes motifs renvoyés, de 3 à 5.")] = 5,
):
    seuil = seuil_effectif(seuil, app.state.parametres)
    tableau = vers_tableau([transaction])
    modele_charge = app.state.modele
    cascade = explicabilite.cascade(modele_charge, tableau, 0)
    decision = decisions(modele_charge, tableau, seuil)[0]
    contributions = [Contribution(code=r.code, libelle=r.libelle, contribution=float(r.contribution)) for r in cascade.itertuples()]
    positives = [c for c in contributions if c.contribution > 0][:motifs]
    return Explication(
        **decision.model_dump(),
        motifs=positives,
        contributions=contributions,
        valeur_moyenne=float(cascade.attrs["base"]),
        prediction=float(cascade.attrs["marge"]),
    )
