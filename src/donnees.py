from pathlib import Path

import pandas as pd

IDENTIFIANT = "kartik2112/fraud-detection"
FICHIERS = ("fraudTrain.csv", "fraudTest.csv")
COLONNES_ECARTEES = ["Unnamed: 0", "unix_time"]
COLONNE_DATE = "trans_date_trans_time"
COLONNE_CIBLE = "is_fraud"
COLONNE_CARTE = "cc_num"
COLONNE_MONTANT = "amt"
COLONNE_CATEGORIE = "category"
COLONNE_ETAT = "state"
FORMAT_DATE = "%Y-%m-%d %H:%M:%S"
DOSSIER_CACHE = Path.home() / ".cache" / "detection-fraude"
FICHIER_CACHE = DOSSIER_CACHE / "transactions.parquet"
PARTS = (0.6, 0.2, 0.2)


def telecharger():
    import kagglehub

    return Path(kagglehub.dataset_download(IDENTIFIANT))


def preparer(df):
    df = df.drop(columns=COLONNES_ECARTEES, errors="ignore")
    df[COLONNE_DATE] = pd.to_datetime(df[COLONNE_DATE], format=FORMAT_DATE)
    return df.sort_values(COLONNE_DATE, kind="stable").reset_index(drop=True)


def lire_fichiers(dossier):
    morceaux = [pd.read_csv(Path(dossier) / nom) for nom in FICHIERS]
    return preparer(pd.concat(morceaux, ignore_index=True))


def charger(dossier=None, utiliser_cache=True):
    if dossier is None and utiliser_cache and FICHIER_CACHE.exists():
        return pd.read_parquet(FICHIER_CACHE)
    source = dossier if dossier is not None else telecharger()
    df = lire_fichiers(source)
    if dossier is None and utiliser_cache:
        FICHIER_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(FICHIER_CACHE, index=False)
    return df


def decouper_chronologiquement(df, parts=PARTS):
    if len(parts) != 3 or abs(sum(parts) - 1.0) > 1e-9 or min(parts) <= 0:
        raise ValueError("les parts doivent etre trois valeurs positives de somme 1")
    df = df.sort_values(COLONNE_DATE, kind="stable")
    n = len(df)
    premiere = int(round(n * parts[0]))
    seconde = int(round(n * (parts[0] + parts[1])))
    entrainement = df.iloc[:premiere].copy()
    validation = df.iloc[premiere:seconde].copy()
    test = df.iloc[seconde:].copy()
    if entrainement[COLONNE_DATE].max() > validation[COLONNE_DATE].min():
        raise ValueError("entrainement et validation se chevauchent dans le temps")
    if validation[COLONNE_DATE].max() > test[COLONNE_DATE].min():
        raise ValueError("validation et test se chevauchent dans le temps")
    return entrainement, validation, test


def resumer(jeux):
    lignes = {}
    for nom, df in jeux.items():
        lignes[nom] = {
            "transactions": len(df),
            "fraudes": int(df[COLONNE_CIBLE].sum()),
            "taux_fraude": float(df[COLONNE_CIBLE].mean()),
            "debut": df[COLONNE_DATE].min(),
            "fin": df[COLONNE_DATE].max(),
            "cartes": int(df[COLONNE_CARTE].nunique()),
        }
    return pd.DataFrame.from_dict(lignes, orient="index")
