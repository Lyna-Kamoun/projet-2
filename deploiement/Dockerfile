FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 service
WORKDIR /home/service/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=service:service api ./api
COPY --chown=service:service src ./src
COPY --chown=service:service app/ressources/parametres.json ./app/ressources/parametres.json
COPY --chown=service:service modeles/xgboost_fraude.joblib ./modeles/xgboost_fraude.joblib

USER service
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "api.principal:app", "--host", "0.0.0.0", "--port", "7860"]
