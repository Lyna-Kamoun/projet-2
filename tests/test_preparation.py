import numpy as np
import pandas as pd
import pytest

from src import donnees, preparation, variables

CATEGORIES = ["grocery_pos", "misc_net", "travel", "shopping_net"]


def jeu(n=1500, graine=0, categories=CATEGORIES):
    rng = np.random.default_rng(graine)
    secondes = np.sort(rng.integers(0, 60 * 86400, n))
    df = pd.DataFrame(
        {
            "trans_date_trans_time": pd.Timestamp("2020-01-01")
            + pd.to_timedelta(secondes, unit="s"),
            "cc_num": rng.integers(1, 40, n),
            "category": rng.choice(categories, n),
            "amt": rng.gamma(2.0, 40.0, n).round(2) + 1.0,
            "lat": rng.uniform(30, 45, n),
            "long": rng.uniform(-120, -75, n),
            "merch_lat": rng.uniform(30, 45, n),
            "merch_long": rng.uniform(-120, -75, n),
            "dob": "1975-03-02",
            "gender": rng.choice(["F", "M"], n),
            "is_fraud": (rng.random(n) < 0.03).astype(int),
        }
    )
    return variables.construire_variables(df)


def decoupage(df):
    return donnees.decouper_chronologiquement(df)


def test_ecreteur_borne_aux_quantiles_de_l_entrainement():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (5000, 2))
    e = preparation.Ecreteur().fit(base)
    nouveau = np.array([[100.0, -100.0], [0.0, 0.0]])
    sortie = e.transform(nouveau)
    assert sortie[0, 0] == pytest.approx(e.bornes_hautes_[0])
    assert sortie[0, 1] == pytest.approx(e.bornes_basses_[1])
    assert sortie[1, 0] == pytest.approx(0.0)


def test_ecreteur_conserve_les_valeurs_manquantes():
    base = np.array([[1.0], [2.0], [np.nan], [3.0]])
    sortie = preparation.Ecreteur().fit(base).transform(base)
    assert np.isnan(sortie[2, 0])


def test_ecreteur_independant_des_nouvelles_donnees():
    base = np.random.default_rng(1).normal(size=(1000, 1))
    e = preparation.Ecreteur().fit(base)
    bornes = (e.bornes_basses_.copy(), e.bornes_hautes_.copy())
    e.transform(np.array([[1e6]]))
    assert np.array_equal(e.bornes_basses_, bornes[0])
    assert np.array_equal(e.bornes_hautes_, bornes[1])


def test_pipeline_lineaire_sans_valeurs_manquantes_et_centre():
    e, v, t = decoupage(jeu())
    p = preparation.pipeline_lineaire()
    xe, xv, xt = preparation.ajuster_et_transformer(p, e, v, t)
    for x in (xe, xv, xt):
        assert not x.isna().any().any()
        assert list(x.columns) == list(xe.columns)
    numeriques = [
        c for c in xe.columns if not c.startswith(("category_", "missingindicator_"))
    ]
    assert xe[numeriques].mean().abs().max() < 1e-8
    assert "log_montant" in xe.columns
    assert "amt" not in xe.columns
    assert [c for c in xe.columns if c.startswith("category_")] == [
        f"category_{c}" for c in sorted(CATEGORIES)
    ]


def test_pipeline_arbres_garde_les_valeurs_manquantes():
    e, v, t = decoupage(jeu())
    p = preparation.pipeline_arbres()
    xe, xv, xt = preparation.ajuster_et_transformer(p, e, v, t)
    assert xe["ratio_moy_30j"].isna().any()
    assert "amt" in xe.columns
    assert "log_montant" not in xe.columns
    assert list(xv.columns) == list(xe.columns)
    assert len(xe) == len(e)


def test_categorie_inconnue_ignoree():
    e, v, t = decoupage(jeu())
    v = v.copy()
    v["category"] = "categorie_nouvelle"
    for p in (preparation.pipeline_lineaire(), preparation.pipeline_arbres()):
        p.fit(e)
        x = p.transform(v)
        colonnes = [c for c in x.columns if c.startswith("category_")]
        assert (x[colonnes].to_numpy() == 0).all()


def test_aucune_fuite_de_la_validation_dans_l_ajustement():
    e, v, t = decoupage(jeu())
    p1 = preparation.pipeline_lineaire().fit(e)
    v_modifie = v.copy()
    v_modifie["amt"] = v_modifie["amt"] * 1000
    v_modifie["log_montant"] = np.log1p(v_modifie["amt"])
    p2 = preparation.pipeline_lineaire().fit(e)
    p2.transform(v_modifie)
    pd.testing.assert_frame_equal(p1.transform(e), p2.transform(e))
    echelle = p1.named_transformers_["bornees"]
    assert echelle.mean_[list(p1.transformers_[0][2]).index("log_montant")] == pytest.approx(
        e["log_montant"].mean()
    )


def test_optionnelles():
    e, v, t = decoupage(jeu())
    sans = preparation.pipeline_lineaire().fit(e).transform(e)
    avec = preparation.pipeline_lineaire(optionnelles=True).fit(e).transform(e)
    assert {"age", "genre_f"} <= set(avec.columns)
    assert not ({"age", "genre_f"} & set(sans.columns))
    arbres = preparation.pipeline_arbres(optionnelles=True).fit(e).transform(e)
    assert {"age", "genre_f"} <= set(arbres.columns)


def test_cible():
    e, v, t = decoupage(jeu())
    assert (preparation.cible(e) == e["is_fraud"].to_numpy()).all()


def test_indicateurs_d_absence_non_standardises():
    e, v, t = decoupage(jeu())
    xe, xv, xt = preparation.ajuster_et_transformer(preparation.pipeline_lineaire(), e, v, t)
    for colonne in ["ratio_moy_30j", "z_montant_30j"]:
        indicateur = xe[f"missingindicator_{colonne}"]
        assert set(indicateur.unique()) <= {0.0, 1.0}
        assert (indicateur.to_numpy() == e[colonne].isna().to_numpy().astype(float)).all()
    assert "missingindicator_delai_precedent" not in xe.columns
    assert (xe.dtypes == "float64").all()
    assert xe.shape[1] == 14 + len(CATEGORIES)
