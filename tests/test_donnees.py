import numpy as np
import pandas as pd
import pytest

from src import donnees


def fabriquer(debut, n, graine):
    rng = np.random.default_rng(graine)
    dates = pd.date_range(debut, periods=n, freq="10min")
    df = pd.DataFrame(
        {
            "Unnamed: 0": np.arange(n),
            "trans_date_trans_time": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "cc_num": rng.integers(1, 40, n),
            "category": rng.choice(["grocery_pos", "misc_net", "travel"], n),
            "amt": rng.gamma(2.0, 30.0, n).round(2),
            "unix_time": np.arange(n) + 1325376018,
            "is_fraud": (rng.random(n) < 0.02).astype(int),
        }
    )
    return df.sample(frac=1.0, random_state=graine).reset_index(drop=True)


@pytest.fixture
def dossier_jeu(tmp_path):
    train = fabriquer("2019-01-01 00:00:00", 600, 1)
    test = fabriquer("2019-01-05 04:00:00", 400, 2)
    train.to_csv(tmp_path / "fraudTrain.csv", index=False)
    test.to_csv(tmp_path / "fraudTest.csv", index=False)
    return tmp_path


def test_lecture_concatene_et_trie(dossier_jeu):
    df = donnees.lire_fichiers(dossier_jeu)
    assert len(df) == 1000
    assert df[donnees.COLONNE_DATE].is_monotonic_increasing
    assert pd.api.types.is_datetime64_any_dtype(df[donnees.COLONNE_DATE])
    assert "Unnamed: 0" not in df.columns
    assert "unix_time" not in df.columns


def test_decoupage_proportions(dossier_jeu):
    df = donnees.lire_fichiers(dossier_jeu)
    entrainement, validation, test = donnees.decouper_chronologiquement(df)
    assert (len(entrainement), len(validation), len(test)) == (600, 200, 200)


def test_decoupage_chronologique(dossier_jeu):
    df = donnees.lire_fichiers(dossier_jeu)
    entrainement, validation, test = donnees.decouper_chronologiquement(df)
    col = donnees.COLONNE_DATE
    assert entrainement[col].max() <= validation[col].min()
    assert validation[col].max() <= test[col].min()


def test_decoupage_sans_recouvrement(dossier_jeu):
    df = donnees.lire_fichiers(dossier_jeu)
    jeux = donnees.decouper_chronologiquement(df)
    indices = [set(j.index) for j in jeux]
    assert not (indices[0] & indices[1])
    assert not (indices[1] & indices[2])
    assert not (indices[0] & indices[2])
    assert set().union(*indices) == set(df.index)


def test_parts_invalides(dossier_jeu):
    df = donnees.lire_fichiers(dossier_jeu)
    with pytest.raises(ValueError):
        donnees.decouper_chronologiquement(df, parts=(0.5, 0.2, 0.2))
    with pytest.raises(ValueError):
        donnees.decouper_chronologiquement(df, parts=(0.8, 0.2))


def test_resume_coherent(dossier_jeu):
    df = donnees.lire_fichiers(dossier_jeu)
    entrainement, validation, test = donnees.decouper_chronologiquement(df)
    resume = donnees.resumer(
        {"entrainement": entrainement, "validation": validation, "test": test}
    )
    assert resume["transactions"].sum() == len(df)
    assert resume["fraudes"].sum() == int(df[donnees.COLONNE_CIBLE].sum())
    assert list(resume.index) == ["entrainement", "validation", "test"]


def test_cache_reutilise(dossier_jeu, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setattr(donnees, "DOSSIER_CACHE", cache)
    monkeypatch.setattr(donnees, "FICHIER_CACHE", cache / "transactions.parquet")
    monkeypatch.setattr(donnees, "telecharger", lambda: dossier_jeu)
    premier = donnees.charger()
    assert (cache / "transactions.parquet").exists()

    def interdit():
        raise AssertionError("telechargement inattendu")

    monkeypatch.setattr(donnees, "telecharger", interdit)
    second = donnees.charger()
    pd.testing.assert_frame_equal(premier, second)
