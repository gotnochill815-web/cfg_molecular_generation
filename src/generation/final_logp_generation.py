from src.cfg.decoder import cfg_decode
from src.generation.prompt_builder import build_prompt


def generate_logp_molecule(
    model,
    tokenizer,
    target_logp=2.0,
    guidance_scale=2.0,
    temperature=1.0,
    top_p=0.95,
    max_new_tokens=128,
    verbose=False,
):
    """
    LogP-conditioned CFG generation using the repository's
    actual cfg_decode() interface.
    """

    conditional_prompt = build_prompt(
        qed=None,
        logp=target_logp,
        tpsa=None,
        sas=None,
    )

    unconditional_prompt = "<NULL>\n<molstart>"

    return cfg_decode(
        model=model,
        tokenizer=tokenizer,
        conditional_prompt=conditional_prompt,
        unconditional_prompt=unconditional_prompt,
        guidance_scale=guidance_scale,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_newline=True,
        verbose=verbose,
    )
