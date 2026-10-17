"""
Classifier-Free Guidance (CFG)

This module implements the guidance equation used during
conditional molecule generation.

guided_logits =
    uncond_logits +
    guidance_scale * (cond_logits - uncond_logits)
"""

import torch


def classifier_free_guidance(
    cond_logits: torch.Tensor,
    uncond_logits: torch.Tensor,
    guidance_scale: float = 2.0,
) -> torch.Tensor:
    """
    Apply Classifier-Free Guidance.

    Parameters
    ----------
    cond_logits : torch.Tensor
        Logits produced using the conditional prompt.

    uncond_logits : torch.Tensor
        Logits produced using the unconditional prompt.

    guidance_scale : float, default=2.0
        Guidance strength.
        1.0 -> No additional guidance
        >1.0 -> Stronger conditioning
        0.0 -> Pure unconditional generation

    Returns
    -------
    torch.Tensor
        Guided logits having the same shape as the inputs.
    """

    if not isinstance(cond_logits, torch.Tensor):
        raise TypeError(
            f"cond_logits must be torch.Tensor, got {type(cond_logits)}"
        )

    if not isinstance(uncond_logits, torch.Tensor):
        raise TypeError(
            f"uncond_logits must be torch.Tensor, got {type(uncond_logits)}"
        )

    if cond_logits.shape != uncond_logits.shape:
        raise ValueError(
            "Conditional and unconditional logits must have "
            f"the same shape.\n"
            f"cond_logits   : {cond_logits.shape}\n"
            f"uncond_logits : {uncond_logits.shape}"
        )

    if guidance_scale < 0:
        raise ValueError(
            "guidance_scale must be >= 0."
        )

    guided_logits = (
        uncond_logits
        + guidance_scale * (cond_logits - uncond_logits)
    )

    return guided_logits