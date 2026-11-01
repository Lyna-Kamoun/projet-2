import shutil
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
SORTIE = RACINE / "deploiement" / "espace_api"
EXCLUS_SRC = ("lancer_*.py", "executer_carnets.py", "figures.py", "graphiques.py")


def copier_src(destination):
    destination.mkdir(parents=True)
    exclus = {chemin for motif in EXCLUS_SRC for chemin in (RACINE / "src").glob(motif)}
    for chemin in (RACINE / "src").glob("*.py"):
        if chemin not in exclus:
            shutil.copy2(chemin, destination / chemin.name)


def main():
    if SORTIE.exists():
        shutil.rmtree(SORTIE)
    SORTIE.mkdir(parents=True)
    shutil.copytree(RACINE / "api", SORTIE / "api", ignore=shutil.ignore_patterns("__pycache__"))
    copier_src(SORTIE / "src")
    (SORTIE / "app" / "ressources").mkdir(parents=True)
    shutil.copy2(RACINE / "app" / "ressources" / "parametres.json", SORTIE / "app" / "ressources" / "parametres.json")
    (SORTIE / "modeles").mkdir()
    shutil.copy2(RACINE / "modeles" / "xgboost_fraude.joblib", SORTIE / "modeles" / "xgboost_fraude.joblib")
    shutil.copy2(RACINE / "deploiement" / "requirements_api.txt", SORTIE / "requirements.txt")
    shutil.copy2(RACINE / "deploiement" / "Dockerfile", SORTIE / "Dockerfile")
    shutil.copy2(RACINE / "deploiement" / "entete_espace.md", SORTIE / "README.md")
    fichiers = sorted(str(p.relative_to(SORTIE)) for p in SORTIE.rglob("*") if p.is_file())
    print(len(fichiers), "fichiers dans", SORTIE)
    for nom in fichiers:
        print(" ", nom)


if __name__ == "__main__":
    main()
