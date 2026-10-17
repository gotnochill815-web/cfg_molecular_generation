"""
Generation utilities for property-conditioned molecular generation.

Uses classifier-free guidance (CFG) with the same unconditional
prompt format used during training dropout:

    <NULL>
    <molstart>
"""

from src.cfg.decoder import cfg_decode
from src.generation.prompt_builder import build_prompt


def generate_molecule(
    model,
    tokenizer,
    qed=None,
    logp=None,
    tpsa=None,
    sas=None,
    guidance_scale=2.0,
    temperature=1.0,
    top_p=0.95,
    max_new_tokens=128,
    stop_newline=True,
    verbose=False,
    targets_are_raw=False,
):
    """
    Generate one molecule using classifier-free guidance.
    """

    # Convert raw physical targets to the scale used during training.
    # Default False preserves existing calls that already pass normalized values.
    if targets_are_raw:
        from src.utils.property_scaling import normalize_targets

        scaled = normalize_targets(
            qed=qed,
            logp=logp,
            tpsa=tpsa,
            sas=sas,
        )
        qed = scaled["qed"]
        logp = scaled["logp"]
        tpsa = scaled["tpsa"]
        sas = scaled["sas"]

    conditional_prompt = build_prompt(
        qed=qed,
        logp=logp,
        tpsa=tpsa,
        sas=sas,
    )

    # IMPORTANT:
    # Training CFG dropout uses:
    #
    # <NULL>
    # <molstart>
    #
    # Therefore inference must use the same unconditional prompt.
    unconditional_prompt = "<NULL>\n<molstart>"

    if verbose:
        print("Conditional prompt:")
        print(repr(conditional_prompt))
        print("Unconditional prompt:")
        print(repr(unconditional_prompt))

    smiles = cfg_decode(
        model=model,
        tokenizer=tokenizer,
        conditional_prompt=conditional_prompt,
        unconditional_prompt=unconditional_prompt,
        guidance_scale=guidance_scale,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        stop_newline=stop_newline,
        verbose=verbose,
    )

    return smiles


def generate_multiple(
    model,
    tokenizer,
    n_samples=10,
    qed=None,
    logp=None,
    tpsa=None,
    sas=None,
    guidance_scale=2.0,
    temperature=1.0,
    top_p=0.95,
    max_new_tokens=128,
    stop_newline=True,
    verbose=False,
    targets_are_raw=False,
):
    """
    Generate multiple molecules under the same property condition.
    """

    molecules = []

    print("=" * 70)
    print("CFG MULTIPLE GENERATION")
    print("=" * 70)

    print(f"Samples         : {n_samples}")
    print(f"Guidance Scale  : {guidance_scale}")
    print(f"Temperature     : {temperature}")
    print(f"Top-p           : {top_p}")
    print(f"Max New Tokens  : {max_new_tokens}")
    print(f"Stop Newline    : {stop_newline}")

    print("\nTarget properties:")
    print(f"QED             : {qed}")
    print(f"LogP            : {logp}")
    print(f"TPSA            : {tpsa}")
    print(f"SAS             : {sas}")

    print("\nTraining-aligned unconditional prompt:")
    print(repr("<NULL>\n<molstart>"))

    print("=" * 70)

    for i in range(n_samples):

        try:
            smiles = generate_molecule(
                model=model,
                tokenizer=tokenizer,
                qed=qed,
                logp=logp,
                tpsa=tpsa,
                sas=sas,
                guidance_scale=guidance_scale,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                stop_newline=stop_newline,
                verbose=verbose,
                targets_are_raw=targets_are_raw,
            )

            molecules.append(smiles)

            if verbose:
                print(
                    f"[{i + 1:03d}/{n_samples:03d}] "
                    f"{smiles}"
                )

        except Exception as e:

            print(
                f"[{i + 1:03d}/{n_samples:03d}] "
                f"GENERATION ERROR: "
                f"{type(e).__name__}: {e}"
            )

            molecules.append(None)

    successful = sum(
        x is not None
        for x in molecules
    )

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Successful generations: "
        f"{successful}/{n_samples}"
    )

    print("=" * 70)

    return molecules
