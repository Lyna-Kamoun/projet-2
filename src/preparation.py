import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.donnees import COLONNE_CATEGORIE, COLONNE_CIBLE
from src.variables import (
    COLONNES_COMPORTEMENTALES,
    COLONNES_GEOGRAPHIQUES,
    COLONNES_OPTIONNELLES,
    COLONNES_TEMPORELLES,
)

QUANTILE_BAS = 0.001
QUANTILE_HAUT = 0.999
COLONNES_AVEC_ABSENCE = ["ratio_moy_30j", "z_montant_30j"]


class Ecreteur(BaseEstimator, TransformerMixin):
    def __init__(self, quantile_bas=QUANTILE_BAS, quantile_haut=QUANTILE_HAUT):
        self.quantile_bas = quantile_bas
        self.quantile_haut = quantile_haut

    def fit(self, X, y=None):
        valeurs = np.asarray(X, dtype="float64")
        self.noms_ = list(X.columns) if hasattr(X, "columns") else None
        self.bornes_basses_ = np.nanquantile(valeurs, self.quantile_bas, axis=0)
        self.bornes_hautes_ = np.nanquantile(valeurs, self.quantile_haut, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "bornes_basses_")
        valeurs = np.asarray(X, dtype="float64")
        return np.clip(valeurs, self.bornes_basses_, self.bornes_hautes_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray(self.noms_, dtype=object)


def en_flottant(valeurs):
    return valeurs.astype("float64")


def indicateur_absence():
    return Pipeline(
        [
            ("indicateur", MissingIndicator(features="all")),
            ("conversion", FunctionTransformer(en_flottant, feature_names_out="one-to-one")),
        ]
    )


def encodeur_categorie():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def pipeline_lineaire(optionnelles=False):
    bornees = COLONNES_TEMPORELLES + ["log_montant"] + COLONNES_GEOGRAPHIQUES
    if optionnelles:
        bornees = bornees + COLONNES_OPTIONNELLES
    queue_lourde = Pipeline(
        [
            ("ecretage", Ecreteur()),
            ("imputation", SimpleImputer(strategy="median")),
            ("echelle", StandardScaler()),
        ]
    )
    transformation = ColumnTransformer(
        [
            ("bornees", StandardScaler(), bornees),
            ("historique", queue_lourde, COLONNES_COMPORTEMENTALES),
            ("absence", indicateur_absence(), COLONNES_AVEC_ABSENCE),
            ("categorie", encodeur_categorie(), [COLONNE_CATEGORIE]),
        ],
        verbose_feature_names_out=False,
    )
    transformation.set_output(transform="pandas")
    return transformation


def pipeline_arbres(optionnelles=False):
    numeriques = (
        COLONNES_TEMPORELLES
        + ["amt"]
        + COLONNES_COMPORTEMENTALES
        + COLONNES_GEOGRAPHIQUES
    )
    if optionnelles:
        numeriques = numeriques + COLONNES_OPTIONNELLES
    transformation = ColumnTransformer(
        [
            ("numeriques", "passthrough", numeriques),
            ("categorie", encodeur_categorie(), [COLONNE_CATEGORIE]),
        ],
        verbose_feature_names_out=False,
    )
    transformation.set_output(transform="pandas")
    return transformation


def cible(df):
    return df[COLONNE_CIBLE].to_numpy()


def ajuster_et_transformer(pipeline, entrainement, validation, test):
    x_entrainement = pipeline.fit_transform(entrainement)
    return x_entrainement, pipeline.transform(validation), pipeline.transform(test)
