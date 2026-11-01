import math
import os

import httpx
import pandas as pd

DELAI_SECONDES = 20.0


class ErreurAPI(Exception):
    pass


def adresse_par_defaut():
    return os.environ.get("URL_API", "")


def cle_par_defaut():
    return os.environ.get("CLE_API", "")


def normaliser_adresse(adresse):
    adresse = (adresse or "").strip().rstrip("/")
    if adresse and not adresse.startswith(("http://", "https://")):
        adresse = "http://" + adresse
    return adresse


def appeler(adresse, methode, chemin, charge=None, parametres=None, cle=None, client=None, delai=DELAI_SECONDES):
    adresse = normaliser_adresse(adresse)
    if not adresse and client is None:
        raise ErreurAPI("Aucune adresse d'API renseignée.")
    en_tetes = {"X-API-Key": cle} if cle else {}
    cible = chemin if client is not None else adresse + chemin
    try:
        if client is not None:
            reponse = client.request(methode, cible, json=charge, params=parametres, headers=en_tetes)
        else:
            reponse = httpx.request(methode, cible, json=charge, params=parametres, headers=en_tetes, timeout=delai)
    except httpx.TimeoutException:
        raise ErreurAPI("L'API n'a pas répondu dans le délai imparti. Un service hébergé gratuitement peut avoir besoin d'une minute pour se réveiller : réessayer.")
    except httpx.HTTPError as erreur:
        raise ErreurAPI("Connexion impossible à l'API (" + erreur.__class__.__name__ + "). Vérifier l'adresse et que le service est démarré.")
    if reponse.status_code == 401:
        raise ErreurAPI("Clé d'API refusée.")
    if reponse.status_code == 422:
        details = reponse.json().get("detail", [])
        messages = [str(d.get("msg", d)) if isinstance(d, dict) else str(d) for d in details]
        raise ErreurAPI("Transaction refusée par l'API : " + " ; ".join(messages))
    if reponse.status_code >= 400:
        raise ErreurAPI("Réponse inattendue de l'API (code " + str(reponse.status_code) + ").")
    try:
        return reponse.json()
    except ValueError:
        raise ErreurAPI("La réponse de l'API n'est pas du JSON : l'adresse ne désigne probablement pas ce service.")


def valeur_json(valeur):
    if valeur is None:
        return None
    if isinstance(valeur, float) and math.isnan(valeur):
        return None
    return valeur


def charge_transaction(horodatage, montant, categorie, nb_tx_1h=0, nb_tx_24h=0, ratio_moy_30j=None, delai_precedent_secondes=None, z_montant_30j=None, distance_km=0.0):
    return {
        "horodatage": pd.Timestamp(horodatage).isoformat(),
        "montant": float(montant),
        "categorie": categorie,
        "nb_tx_1h": int(nb_tx_1h),
        "nb_tx_24h": int(nb_tx_24h),
        "ratio_moy_30j": valeur_json(ratio_moy_30j),
        "delai_precedent_secondes": valeur_json(delai_precedent_secondes),
        "z_montant_30j": valeur_json(z_montant_30j),
        "distance_km": float(distance_km),
    }


def tableau_contributions(reponse):
    tableau = pd.DataFrame(reponse["contributions"]).rename(columns={"code": "code", "libelle": "libelle", "contribution": "contribution"})
    tableau.attrs["base"] = reponse["valeur_moyenne"]
    tableau.attrs["marge"] = reponse["prediction"]
    tableau.attrs["probabilite_corrigee"] = reponse["probabilite"]
    return tableau
