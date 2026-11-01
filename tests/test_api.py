import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.principal import app, vers_tableau
from api.schemas import Transaction
from src import application, modeles_supervises

EXEMPLE = {
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
PETIT_MONTANT = {**EXEMPLE, "horodatage": "2020-10-15T12:00:00", "montant": 20.0, "categorie": "travel"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_sante(client):
    r = client.get("/sante")
    assert r.status_code == 200
    assert r.json() == {"statut": "ok", "modele": "xgboost", "version": "1.0.0"}


def test_description_du_modele(client):
    r = client.get("/modele")
    assert r.status_code == 200
    corps = r.json()
    parametres = application.lire_parametres()
    assert corps["seuil_par_defaut"] == pytest.approx(parametres["seuils"]["xgboost"])
    assert corps["cout_alerte"] == parametres["cout_alerte"]
    assert corps["categories"] == application.CATEGORIES
    assert "amt" in corps["variables"] and "category_travel" in corps["variables"]


def test_scorer_correspond_au_calcul_local(client):
    modele = modeles_supervises.charger("xgboost")
    attendu = modele.probabilites(vers_tableau([Transaction(**EXEMPLE)]))[0]
    r = client.post("/scorer", json=EXEMPLE)
    assert r.status_code == 200
    corps = r.json()
    assert corps["probabilite"] == pytest.approx(float(attendu))
    assert 0 <= corps["score_brut"] <= 1
    assert corps["seuil"] == pytest.approx(application.lire_parametres()["seuils"]["xgboost"])
    assert corps["alerte"] is True
    assert client.post("/scorer", json=PETIT_MONTANT).json()["alerte"] is False


def test_le_seuil_modifie_la_decision(client):
    assert client.post("/scorer", json=EXEMPLE, params={"seuil": 0.9999}).json()["alerte"] is False
    assert client.post("/scorer", json=PETIT_MONTANT, params={"seuil": 0.000001}).json()["alerte"] is True
    assert client.post("/scorer", json=EXEMPLE, params={"seuil": 1.5}).status_code == 422
    assert client.post("/scorer", json=EXEMPLE, params={"seuil": 0}).status_code == 422


def test_expliquer(client):
    r = client.post("/expliquer", json=EXEMPLE)
    assert r.status_code == 200
    corps = r.json()
    assert corps["alerte"] is True
    motifs = corps["motifs"]
    assert len(motifs) == 5
    assert all(m["contribution"] > 0 for m in motifs)
    assert [m["contribution"] for m in motifs] == sorted([m["contribution"] for m in motifs], reverse=True)
    assert all(m["code"] and m["libelle"] for m in motifs)
    total = corps["valeur_moyenne"] + sum(c["contribution"] for c in corps["contributions"])
    assert total == pytest.approx(corps["prediction"], abs=1e-4)
    assert len(corps["contributions"]) == 10
    assert corps["probabilite"] == pytest.approx(client.post("/scorer", json=EXEMPLE).json()["probabilite"])


def test_nombre_de_motifs(client):
    assert len(client.post("/expliquer", json=EXEMPLE, params={"motifs": 3}).json()["motifs"]) == 3
    assert client.post("/expliquer", json=EXEMPLE, params={"motifs": 2}).status_code == 422
    assert client.post("/expliquer", json=EXEMPLE, params={"motifs": 6}).status_code == 422


def test_sans_historique(client):
    sans = {**EXEMPLE, "ratio_moy_30j": None, "delai_precedent_secondes": None, "z_montant_30j": None, "nb_tx_24h": 0}
    r = client.post("/expliquer", json=sans)
    assert r.status_code == 200
    libelles = [c["libelle"] for c in r.json()["contributions"]]
    assert "Première transaction de la carte" in libelles
    minimal = {"horodatage": "2020-10-15T10:00:00", "montant": 12.5, "categorie": "home"}
    assert client.post("/scorer", json=minimal).status_code == 200


@pytest.mark.parametrize(
    "modification",
    [
        {"categorie": "inconnue"},
        {"montant": 0},
        {"montant": -5},
        {"montant": 2_000_000},
        {"nb_tx_1h": -1},
        {"distance_km": -3},
        {"horodatage": "pas une date"},
        {"ratio_moy_30j": -1},
    ],
)
def test_entrees_invalides(client, modification):
    r = client.post("/scorer", json={**EXEMPLE, **modification})
    assert r.status_code == 422


def test_champ_obligatoire_manquant(client):
    incomplet = {k: v for k, v in EXEMPLE.items() if k != "montant"}
    assert client.post("/scorer", json=incomplet).status_code == 422


def test_message_de_categorie_inconnue(client):
    r = client.post("/scorer", json={**EXEMPLE, "categorie": "yacht"})
    assert "Catégorie inconnue" in r.text and "shopping_net" in r.text


def test_cle_d_api(client, monkeypatch):
    monkeypatch.setenv("CLE_API", "secret")
    assert client.get("/sante").status_code == 200
    assert client.post("/scorer", json=EXEMPLE).status_code == 401
    assert client.post("/scorer", json=EXEMPLE, headers={"X-API-Key": "mauvaise"}).status_code == 401
    assert client.post("/scorer", json=EXEMPLE, headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/modele").status_code == 401
    assert client.get("/modele", headers={"X-API-Key": "secret"}).status_code == 200


def test_documentation_interactive(client):
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert {"/sante", "/modele", "/scorer", "/expliquer"} == set(schema["paths"])
    assert schema["info"]["version"] == "1.0.0"
