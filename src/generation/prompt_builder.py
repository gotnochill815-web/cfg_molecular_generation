"""
Prompt builder.
"""


def build_prompt(
    qed=None,
    logp=None,
    tpsa=None,
    sas=None,
):

    prompt = "<pstart>\n"

    if qed is not None:
        prompt += f"<QED> {qed:.2f}\n"

    if logp is not None:
        prompt += f"<LOGP> {logp:.2f}\n"

    if tpsa is not None:
        prompt += f"<TPSA> {tpsa:.2f}\n"

    if sas is not None:
        prompt += f"<SAS> {sas:.2f}\n"

    prompt += "<molstart>"

    return prompt