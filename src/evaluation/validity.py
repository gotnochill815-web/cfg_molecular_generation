from rdkit import Chem


def is_valid_smiles(smiles):

    mol = Chem.MolFromSmiles(smiles)

    return mol is not None