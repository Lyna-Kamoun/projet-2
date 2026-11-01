import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

RACINE = Path(__file__).resolve().parent.parent
ENTREE = str(RACINE / "app" / "Accueil.py")


def port_libre():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def service():
    port = port_libre()
    processus = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.principal:app", "--port", str(port), "--log-level", "warning"],
        cwd=RACINE,
    )
    adresse = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if httpx.get(adresse + "/sante", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            pytest.fail("Le service ne démarre pas.")
        yield adresse
    finally:
        processus.terminate()
        processus.wait(timeout=20)


def ouvrir():
    at = AppTest.from_file(ENTREE, default_timeout=180)
    at.run()
    at.switch_page("vues/api.py")
    at.run()
    return at


def metriques(at):
    resultat = {}
    for m in at.metric:
        resultat.setdefault(m.label, m.value)
    return resultat


def test_sans_adresse_calcul_local():
    at = ouvrir()
    assert not at.exception, [e.value for e in at.exception]
    valeurs = metriques(at)
    assert valeurs["Décision"] == "Alerte"
    assert any("Calcul local" in c.value for c in at.caption)


def test_avec_service_les_resultats_concordent(service):
    local = metriques(ouvrir())
    at = ouvrir()
    at.text_input(key="adresse_api").set_value(service).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error
    distant = metriques(at)
    assert distant["Probabilité de fraude"] == local["Probabilité de fraude"]
    assert distant["Décision"] == local["Décision"]
    assert any("Réponse du service" in c.value for c in at.caption)


def test_bouton_tester(service):
    at = ouvrir()
    at.text_input(key="adresse_api").set_value(service)
    at.button(key="tester_api").click().run()
    assert not at.exception
    assert any("joignable" in s.value for s in at.success)


def test_adresse_erronee_affiche_une_erreur():
    at = ouvrir()
    at.text_input(key="adresse_api").set_value(f"http://127.0.0.1:{port_libre()}").run()
    assert not at.exception
    assert any("Connexion impossible" in e.value or "délai" in e.value for e in at.error)


def test_seuil_personnalise(service):
    at = ouvrir()
    at.text_input(key="adresse_api").set_value(service)
    at.number_input(key="api_seuil").set_value(0.99).run()
    assert not at.exception
    assert metriques(at)["Seuil appliqué"].startswith("99")
