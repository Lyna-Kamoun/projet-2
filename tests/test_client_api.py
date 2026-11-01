import math
import socket

import httpx
import pytest
from fastapi.testclient import TestClient

from api.principal import app
from src import client_api


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def adresse_fermee():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def exemple(**changements):
    base = client_api.charge_transaction("2020-10-15 23:10:00", 950, "shopping_net", 0, 3, 1.0, 43200.0, 0.0, 78.0)
    base.update(changements)
    return base


def test_normaliser_adresse():
    assert client_api.normaliser_adresse("localhost:8000/") == "http://localhost:8000"
    assert client_api.normaliser_adresse(" https://exemple.org// ") == "https://exemple.org"
    assert client_api.normaliser_adresse("") == ""
    assert client_api.normaliser_adresse(None) == ""


def test_charge_transaction_convertit_les_valeurs_absentes():
    charge = client_api.charge_transaction("2020-10-15 23:10:00", 12, "home", 0, 0, float("nan"), None, float("nan"), 3)
    assert charge["horodatage"] == "2020-10-15T23:10:00"
    assert charge["ratio_moy_30j"] is None and charge["delai_precedent_secondes"] is None and charge["z_montant_30j"] is None
    assert charge["montant"] == 12.0 and charge["distance_km"] == 3.0


def test_sante_et_modele(client):
    assert client_api.appeler("", "GET", "/sante", client=client)["statut"] == "ok"
    assert "xgboost" == client_api.appeler("", "GET", "/modele", client=client)["nom"]


def test_expliquer_et_tableau(client):
    reponse = client_api.appeler("", "POST", "/expliquer", charge=exemple(), parametres={"motifs": 3}, client=client)
    assert len(reponse["motifs"]) == 3
    tableau = client_api.tableau_contributions(reponse)
    assert list(tableau.columns) == ["code", "libelle", "contribution"]
    assert tableau.attrs["base"] == reponse["valeur_moyenne"]
    assert math.isclose(tableau.attrs["base"] + tableau["contribution"].sum(), tableau.attrs["marge"], abs_tol=1e-4)


def test_transaction_refusee(client):
    with pytest.raises(client_api.ErreurAPI, match="Catégorie inconnue"):
        client_api.appeler("", "POST", "/scorer", charge=exemple(categorie="yacht"), client=client)


def test_cle_refusee(client, monkeypatch):
    monkeypatch.setenv("CLE_API", "secret")
    with pytest.raises(client_api.ErreurAPI, match="Clé d'API refusée"):
        client_api.appeler("", "POST", "/scorer", charge=exemple(), client=client)
    assert "probabilite" in client_api.appeler("", "POST", "/scorer", charge=exemple(), cle="secret", client=client)


def test_adresse_absente():
    with pytest.raises(client_api.ErreurAPI, match="Aucune adresse"):
        client_api.appeler("", "GET", "/sante")


def test_service_injoignable():
    with pytest.raises(client_api.ErreurAPI, match="Connexion impossible"):
        client_api.appeler(adresse_fermee(), "GET", "/sante", delai=5)


def test_reponse_qui_n_est_pas_du_json(monkeypatch):
    transport = httpx.MockTransport(lambda requete: httpx.Response(200, text="<html></html>"))
    with httpx.Client(transport=transport, base_url="http://exemple.test") as faux:
        with pytest.raises(client_api.ErreurAPI, match="pas du JSON"):
            client_api.appeler("", "GET", "/sante", client=faux)


def test_erreur_serveur():
    transport = httpx.MockTransport(lambda requete: httpx.Response(503))
    with httpx.Client(transport=transport, base_url="http://exemple.test") as faux:
        with pytest.raises(client_api.ErreurAPI, match="503"):
            client_api.appeler("", "GET", "/sante", client=faux)
