import numpy as np
import pandas as pd

from src.donnees import (
    COLONNE_CARTE,
    COLONNE_DATE,
    COLONNE_MONTANT,
)

SECONDES_1H = 3600
SECONDES_24H = 24 * 3600
SECONDES_30J = 30 * 24 * 3600
RAYON_TERRE_KM = 6371.0088
HEURES_NUIT = (22, 23, 0, 1, 2, 3)
TOLERANCE_VARIANCE = 1e-8

COLONNES_TEMPORELLES = ["heure_sin", "heure_cos", "jour_sin", "jour_cos", "nuit"]
COLONNES_MONTANT = ["amt", "log_montant"]
COLONNES_COMPORTEMENTALES = [
    "nb_tx_1h",
    "nb_tx_24h",
    "ratio_moy_30j",
    "delai_precedent",
    "z_montant_30j",
]
COLONNES_GEOGRAPHIQUES = ["distance_km"]
COLONNES_OPTIONNELLES = ["age", "genre_f"]
COLONNES_NUMERIQUES = (
    COLONNES_TEMPORELLES
    + COLONNES_MONTANT
    + COLONNES_COMPORTEMENTALES
    + COLONNES_GEOGRAPHIQUES
)


def variables_temporelles(df):
    dates = df[COLONNE_DATE]
    heure = dates.dt.hour + dates.dt.minute / 60.0 + dates.dt.second / 3600.0
    jour = dates.dt.day_of_week
    resultat = pd.DataFrame(index=df.index)
    resultat["heure_sin"] = np.sin(2 * np.pi * heure / 24.0)
    resultat["heure_cos"] = np.cos(2 * np.pi * heure / 24.0)
    resultat["jour_sin"] = np.sin(2 * np.pi * jour / 7.0)
    resultat["jour_cos"] = np.cos(2 * np.pi * jour / 7.0)
    resultat["nuit"] = dates.dt.hour.isin(HEURES_NUIT).astype(int)
    return resultat


def distance_haversine(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * RAYON_TERRE_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def calculer_historique_carte(temps, montants):
    n = len(temps)
    positions = np.arange(n)
    gauche_1h = np.searchsorted(temps, temps - SECONDES_1H, side="left")
    gauche_24h = np.searchsorted(temps, temps - SECONDES_24H, side="left")
    gauche_30j = np.searchsorted(temps, temps - SECONDES_30J, side="left")
    somme = np.concatenate([[0.0], np.cumsum(montants)])
    somme_carres = np.concatenate([[0.0], np.cumsum(montants**2)])
    effectif = positions - gauche_30j
    total = somme[positions] - somme[gauche_30j]
    total_carres = somme_carres[positions] - somme_carres[gauche_30j]
    avec_historique = effectif > 0
    moyenne = np.full(n, np.nan)
    moyenne[avec_historique] = total[avec_historique] / effectif[avec_historique]
    ratio = np.full(n, np.nan)
    valide = avec_historique & (moyenne > 0)
    ratio[valide] = montants[valide] / moyenne[valide]
    z = np.full(n, np.nan)
    deux = effectif >= 2
    variance = np.full(n, np.nan)
    variance[deux] = (
        total_carres[deux] - total[deux] ** 2 / effectif[deux]
    ) / (effectif[deux] - 1)
    seuil = TOLERANCE_VARIANCE * (1.0 + np.where(deux, moyenne, 0.0) ** 2)
    ecart_type_positif = deux & (variance > seuil)
    z[ecart_type_positif] = (
        montants[ecart_type_positif] - moyenne[ecart_type_positif]
    ) / np.sqrt(variance[ecart_type_positif])
    delai = np.full(n, np.nan)
    delai[1:] = np.diff(temps)
    return {
        "nb_tx_1h": positions - gauche_1h,
        "nb_tx_24h": positions - gauche_24h,
        "ratio_moy_30j": ratio,
        "delai_precedent": delai,
        "z_montant_30j": z,
    }


def variables_comportementales(df):
    if not df[COLONNE_DATE].is_monotonic_increasing:
        raise ValueError("les transactions doivent etre triees par date croissante")
    temps = df[COLONNE_DATE].to_numpy().astype("datetime64[s]").astype("int64")
    montants = df[COLONNE_MONTANT].to_numpy(dtype="float64")
    n = len(df)
    sorties = {
        "nb_tx_1h": np.zeros(n, dtype="int64"),
        "nb_tx_24h": np.zeros(n, dtype="int64"),
        "ratio_moy_30j": np.full(n, np.nan),
        "delai_precedent": np.full(n, np.nan),
        "z_montant_30j": np.full(n, np.nan),
    }
    groupes = df.groupby(COLONNE_CARTE, sort=False).indices
    for positions in groupes.values():
        positions = np.sort(positions)
        resultat = calculer_historique_carte(temps[positions], montants[positions])
        for nom, valeurs in resultat.items():
            sorties[nom][positions] = valeurs
    return pd.DataFrame(sorties, index=df.index)


def construire_variables(df):
    resultat = df.copy()
    temporelles = variables_temporelles(df)
    comportementales = variables_comportementales(df)
    resultat = pd.concat([resultat, temporelles, comportementales], axis=1)
    resultat["log_montant"] = np.log1p(df[COLONNE_MONTANT])
    resultat["distance_km"] = distance_haversine(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )
    naissance = pd.to_datetime(df["dob"], format="%Y-%m-%d")
    resultat["age"] = (df[COLONNE_DATE] - naissance).dt.days / 365.25
    resultat["genre_f"] = (df["gender"] == "F").astype(int)
    return resultat
