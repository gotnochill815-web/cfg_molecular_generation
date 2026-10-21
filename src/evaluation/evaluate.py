import pandas as pd

from .properties import compute_properties
from .validity import is_valid_smiles

COLUMNS = ["SMILES", "QED", "LogP", "TPSA", "SAS"]


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

    df = pd.DataFrame(rows, columns=COLUMNS)

    summary = {
        "Validity": valid / len(smiles_list) if smiles_list else 0.0,
        "Avg_QED": df["QED"].mean() if valid else None,
        "Avg_LogP": df["LogP"].mean() if valid else None,
        "Avg_TPSA": df["TPSA"].mean() if valid else None,
        "Avg_SAS": df["SAS"].mean() if valid else None,
    }

    return df, summary
