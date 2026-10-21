from rdkit import Chem


def is_valid_smiles(smiles):

    # Whitespace would make RDKit parse only the part before it.
    smiles = "".join(str(smiles).split())

    if not smiles:
        return False

    mol = Chem.MolFromSmiles(smiles)

    return mol is not None
