from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Descriptors

import sascorer


def compute_properties(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return {
        "QED": QED.qed(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "SAS": sascorer.calculateScore(mol),
    }