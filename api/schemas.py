from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import application


class Transaction(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "horodatage": "2020-10-15T23:10:00",
                "montant": 950.0,
                "categorie": "shopping_net",
                "nb_tx_1h": 0,
                "nb_tx_24h": 3,
                "ratio_moy_30j": 1.0,
                "delai_precedent_secondes": 43200,
                "z_montant_30j": 0.0,
                "distance_km": 78.0,
            }
        }
    )

    horodatage: datetime = Field(description="Date et heure de la transaction.")
    montant: float = Field(gt=0, le=1_000_000, description="Montant de la transaction.")
    categorie: str = Field(description="Catégorie d'achat, parmi celles listées par GET /modele.")
    nb_tx_1h: int = Field(0, ge=0, le=1000, description="Transactions antérieures de la carte sur la dernière heure.")
    nb_tx_24h: int = Field(0, ge=0, le=10000, description="Transactions antérieures de la carte sur les dernières 24 heures.")
    ratio_moy_30j: Optional[float] = Field(None, ge=0, description="Montant divisé par la moyenne des montants de la carte sur 30 jours. Absent si la carte n'a pas d'historique.")
    delai_precedent_secondes: Optional[float] = Field(None, ge=0, description="Secondes écoulées depuis la transaction précédente de la carte. Absent pour la première transaction.")
    z_montant_30j: Optional[float] = Field(None, description="Écart du montant à la moyenne sur 30 jours, en écarts-types. Absent si l'historique est insuffisant.")
    distance_km: float = Field(0.0, ge=0, le=20000, description="Distance entre la position du porteur et celle du marchand.")

    @field_validator("categorie")
    @classmethod
    def categorie_connue(cls, valeur):
        if valeur not in application.CATEGORIES:
            raise ValueError("Catégorie inconnue : " + valeur + ". Valeurs possibles : " + ", ".join(application.CATEGORIES))
        return valeur


class Decision(BaseModel):
    probabilite: float = Field(description="Probabilité de fraude corrigée du sous-échantillonnage.")
    score_brut: float = Field(description="Sortie brute du modèle, avant correction.")
    alerte: bool
    seuil: float = Field(description="Seuil appliqué à la probabilité.")


class Contribution(BaseModel):
    code: str
    libelle: str
    contribution: float = Field(description="Contribution SHAP en log-odds ; positive si elle pousse vers l'alerte.")


class Explication(Decision):
    motifs: list[Contribution] = Field(description="Variables qui poussent le plus vers l'alerte.")
    contributions: list[Contribution] = Field(description="Contribution de chaque variable, de la plus forte à la plus faible en valeur absolue.")
    valeur_moyenne: float = Field(description="Prédiction moyenne du modèle, en log-odds.")
    prediction: float = Field(description="Prédiction pour cette transaction, en log-odds.")


class Sante(BaseModel):
    statut: str
    modele: str
    version: str


class Modele(BaseModel):
    nom: str
    version: str
    seuil_par_defaut: float
    cout_alerte: float
    beta: float
    categories: list[str]
    variables: list[str]
