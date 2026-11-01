import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

DOSSIER = Path(__file__).resolve().parent.parent / "notebooks"


def executer(chemin, delai=1800):
    carnet = nbformat.read(chemin, as_version=4)
    client = NotebookClient(carnet, timeout=delai, kernel_name="python3", resources={"metadata": {"path": str(chemin.parent)}})
    client.execute()
    nbformat.write(carnet, chemin)


def main():
    noms = sys.argv[1:]
    chemins = [DOSSIER / nom for nom in noms] if noms else sorted(DOSSIER.glob("*.ipynb"))
    for chemin in chemins:
        print("execution de", chemin.name, flush=True)
        executer(chemin)
    print("termine")


if __name__ == "__main__":
    main()
