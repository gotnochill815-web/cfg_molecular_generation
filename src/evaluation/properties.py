import os
import sys

from rdkit import Chem, RDConfig
from rdkit.Chem import QED
from rdkit.Chem import Descriptors

# sascorer ships with RDKit's SA_Score contrib module, which is not on the
# import path by default.
_SA_DIR = os.path.join(RDConfig.RDContribDir, "SA_Score")
if _SA_DIR not in sys.path:
    sys.path.insert(0, _SA_DIR)

import sascorer


def compute_properties(smiles):

    # Whitespace would make RDKit parse only the part before it.
    smiles = "".join(str(smiles).split())

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return {
        "QED": QED.qed(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "SAS": sascorer.calculateScore(mol),
    }
