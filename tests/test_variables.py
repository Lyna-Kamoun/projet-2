import numpy as np
import pandas as pd
import pytest

from src import variables


def transactions(lignes):
    df = pd.DataFrame(lignes, columns=["date", "cc_num", "amt"])
    df["trans_date_trans_time"] = pd.to_datetime(df.pop("date"))
    df["lat"] = 40.0
    df["long"] = -75.0
    df["merch_lat"] = 40.5
    df["merch_long"] = -75.5
    df["dob"] = "1980-06-15"
    df["gender"] = "F"
    df["category"] = "misc_net"
    return df.sort_values("trans_date_trans_time", kind="stable").reset_index(drop=True)


def reference(df, i, fenetre_secondes):
    ligne = df.iloc[i]
    passe = df.iloc[:i]
    passe = passe[passe["cc_num"] == ligne["cc_num"]]
    limite = ligne["trans_date_trans_time"] - pd.Timedelta(seconds=fenetre_secondes)
    return passe[passe["trans_date_trans_time"] >= limite]


def aleatoire(n=400, cartes=5, graine=3):
    rng = np.random.default_rng(graine)
    secondes = np.sort(rng.integers(0, 40 * 86400, n))
    dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(secondes, unit="s")
    df = transactions(
        list(zip(dates, rng.integers(1, cartes + 1, n), rng.gamma(2.0, 40.0, n).round(2)))
    )
    return df


def test_distance_haversine():
    assert variables.distance_haversine(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)
    assert variables.distance_haversine(0.0, 0.0, 0.0, 1.0) == pytest.approx(111.195, abs=0.05)
    paris_londres = variables.distance_haversine(48.8566, 2.3522, 51.5074, -0.1278)
    assert paris_londres == pytest.approx(343.5, abs=2.0)


def test_codage_cyclique_et_nuit():
    dates = [
        "2020-01-06 00:00:00",
        "2020-01-06 06:00:00",
        "2020-01-06 12:00:00",
        "2020-01-06 03:59:59",
        "2020-01-06 04:00:00",
        "2020-01-06 21:59:59",
        "2020-01-06 22:00:00",
    ]
    df = pd.DataFrame({"trans_date_trans_time": pd.to_datetime(dates)})
    t = variables.variables_temporelles(df)
    assert t.loc[0, "heure_sin"] == pytest.approx(0.0, abs=1e-12)
    assert t.loc[0, "heure_cos"] == pytest.approx(1.0)
    assert t.loc[1, "heure_sin"] == pytest.approx(1.0)
    assert t.loc[2, "heure_cos"] == pytest.approx(-1.0)
    assert t.loc[0, "jour_sin"] == pytest.approx(0.0, abs=1e-12)
    assert t.loc[0, "jour_cos"] == pytest.approx(1.0)
    assert list(t["nuit"]) == [1, 0, 0, 1, 0, 0, 1]


def test_exemple_du_cours():
    lignes = [(f"2020-01-0{j} 12:00:00", 1, m) for j, m in enumerate([42, 38, 45, 40, 47, 43], start=1)]
    lignes.append(("2020-01-07 03:00:00", 1, 260.0))
    df = transactions(lignes)
    h = variables.variables_comportementales(df).iloc[-1]
    assert h["ratio_moy_30j"] == pytest.approx(260.0 / 42.5)
    assert h["z_montant_30j"] == pytest.approx((260.0 - 42.5) / np.std([42, 38, 45, 40, 47, 43], ddof=1))
    assert h["z_montant_30j"] == pytest.approx(66.5, abs=0.1)
    assert h["nb_tx_1h"] == 0
    assert h["nb_tx_24h"] == 1
    assert h["delai_precedent"] == pytest.approx(15 * 3600)


def test_debut_d_historique():
    df = transactions(
        [
            ("2020-01-01 10:00:00", 1, 50.0),
            ("2020-01-02 10:00:00", 1, 70.0),
            ("2020-01-03 10:00:00", 1, 60.0),
        ]
    )
    h = variables.variables_comportementales(df)
    assert h.loc[0, "nb_tx_24h"] == 0
    assert np.isnan(h.loc[0, "ratio_moy_30j"])
    assert np.isnan(h.loc[0, "z_montant_30j"])
    assert np.isnan(h.loc[0, "delai_precedent"])
    assert h.loc[1, "ratio_moy_30j"] == pytest.approx(70.0 / 50.0)
    assert np.isnan(h.loc[1, "z_montant_30j"])
    assert not np.isnan(h.loc[2, "z_montant_30j"])


def test_ecart_type_nul():
    df = transactions(
        [
            ("2020-01-01 10:00:00", 1, 25.0),
            ("2020-01-02 10:00:00", 1, 25.0),
            ("2020-01-03 10:00:00", 1, 25.0),
            ("2020-01-04 10:00:00", 1, 90.0),
        ]
    )
    h = variables.variables_comportementales(df)
    assert np.isnan(h.loc[3, "z_montant_30j"])
    assert h.loc[3, "ratio_moy_30j"] == pytest.approx(90.0 / 25.0)


def test_bord_de_fenetre_inclus():
    df = transactions(
        [
            ("2020-01-01 10:00:00", 1, 10.0),
            ("2020-01-01 11:00:00", 1, 10.0),
            ("2020-01-01 12:00:01", 1, 10.0),
        ]
    )
    h = variables.variables_comportementales(df)
    assert list(h["nb_tx_1h"]) == [0, 1, 0]
    assert list(h["nb_tx_24h"]) == [0, 1, 2]


def test_cartes_independantes():
    df = transactions(
        [
            ("2020-01-01 10:00:00", 1, 10.0),
            ("2020-01-01 10:05:00", 2, 10.0),
            ("2020-01-01 10:10:00", 2, 10.0),
        ]
    )
    h = variables.variables_comportementales(df)
    assert list(h["nb_tx_1h"]) == [0, 0, 1]


def test_comparaison_calcul_de_reference():
    df = aleatoire()
    h = variables.variables_comportementales(df)
    for i in range(0, len(df), 7):
        ligne = df.iloc[i]
        un_h = reference(df, i, 3600)
        un_j = reference(df, i, 86400)
        trente = reference(df, i, 30 * 86400)
        assert h.loc[i, "nb_tx_1h"] == len(un_h)
        assert h.loc[i, "nb_tx_24h"] == len(un_j)
        if len(trente) >= 1:
            assert h.loc[i, "ratio_moy_30j"] == pytest.approx(ligne["amt"] / trente["amt"].mean(), rel=1e-6)
        else:
            assert np.isnan(h.loc[i, "ratio_moy_30j"])
        if len(trente) >= 2:
            attendu = (ligne["amt"] - trente["amt"].mean()) / trente["amt"].std(ddof=1)
            assert h.loc[i, "z_montant_30j"] == pytest.approx(attendu, rel=1e-6, abs=1e-8)
        else:
            assert np.isnan(h.loc[i, "z_montant_30j"])
        precedentes = df.iloc[:i][df.iloc[:i]["cc_num"] == ligne["cc_num"]]
        if len(precedentes):
            ecart = (ligne["trans_date_trans_time"] - precedentes["trans_date_trans_time"].iloc[-1]).total_seconds()
            assert h.loc[i, "delai_precedent"] == pytest.approx(ecart)
        else:
            assert np.isnan(h.loc[i, "delai_precedent"])


def test_aucune_dependance_au_futur():
    df = aleatoire()
    complet = variables.variables_comportementales(df)
    for coupure in (50, 123, 300):
        partiel = variables.variables_comportementales(df.iloc[:coupure].copy())
        pd.testing.assert_frame_equal(
            complet.iloc[:coupure].reset_index(drop=True),
            partiel.reset_index(drop=True),
        )


def test_ordre_requis():
    df = aleatoire(50).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError):
        variables.variables_comportementales(df)


def test_construire_variables_colonnes():
    df = aleatoire(80)
    r = variables.construire_variables(df)
    for colonne in variables.COLONNES_NUMERIQUES + variables.COLONNES_OPTIONNELLES:
        assert colonne in r.columns
    assert np.allclose(r["log_montant"], np.log1p(df["amt"]))
    assert (r["distance_km"] > 0).all()
    assert np.isfinite(r[["age", "distance_km", "log_montant"]]).all().all()
    assert set(r["genre_f"]) == {1}
    assert r["age"].between(39, 41).all()
