import pandas as pd

from .properties import compute_properties
from .validity import is_valid_smiles


def evaluate_smiles(smiles_list):

    rows = []

    valid = 0

    for smi in smiles_list:

        if not is_valid_smiles(smi):
            continue

        valid += 1

        props = compute_properties(smi)

        rows.append({
            "SMILES": smi,
            **props
        })

    df = pd.DataFrame(rows)

    summary = {
        "Validity": valid / len(smiles_list),
        "Avg_QED": df["QED"].mean(),
        "Avg_LogP": df["LogP"].mean(),
        "Avg_TPSA": df["TPSA"].mean(),
        "Avg_SAS": df["SAS"].mean(),
    }

    return df, summary