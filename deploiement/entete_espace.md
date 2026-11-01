---
title: API de détection de fraude explicable
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# API de détection de fraude explicable

Service FastAPI de scoring et d'explication (XGBoost, SHAP) pour la détection de fraude par carte bancaire.
Documentation interactive : `/docs`. Santé du service : `/sante`.

Une clé d'API peut être exigée en définissant le secret `CLE_API` de l'espace ; elle est alors à fournir dans l'en-tête `X-API-Key`.
