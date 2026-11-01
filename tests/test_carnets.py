import ast
import io
import tokenize
from pathlib import Path

import nbformat
import pytest

DOSSIER = Path(__file__).resolve().parent.parent / "notebooks"
CARNETS = sorted(DOSSIER.glob("*.ipynb"))


def test_six_carnets():
    assert [c.name[:2] for c in CARNETS] == ["01", "02", "03", "04", "05", "06"]


@pytest.mark.parametrize("chemin", CARNETS, ids=lambda c: c.name)
def test_code_valide_et_sans_commentaire(chemin):
    carnet = nbformat.read(chemin, as_version=4)
    nbformat.validate(carnet)
    for cellule in carnet.cells:
        if cellule.cell_type != "code":
            continue
        ast.parse(cellule.source)
        jetons = tokenize.generate_tokens(io.StringIO(cellule.source).readline)
        assert not any(j.type == tokenize.COMMENT for j in jetons)
