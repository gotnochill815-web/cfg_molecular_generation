"""
CFG Decoder

Classifier-Free Guidance decoding for SMILES.

This version is compatible with the current repository interface
and adds tokenizer-aware SMILES handling:

    - merged-token structural checking
    - merged-token separator detection
    - repeated n-gram blocking
    - parenthesis/bracket balance protection
    - EOS/newline stopping
"""

import torch
import torch.nn.functional as F

from src.cfg.guidance import classifier_free_guidance


# ============================================================
# TOKENIZER STRUCTURE CACHE
# ============================================================

_STRUCTURE_CACHE = {}


def _build_structure_cache(tokenizer):
    """
    Build tokenizer-aware structural metadata.

    Important:
    GPT-NeoX may tokenize strings such as:

        ")C"
        "C("
        "[N+]"
        "CC;"
        "C,F"

    as single vocabulary pieces.

    Therefore structural handling must inspect token text rather
    than relying only on exact token IDs.
    """

    key = id(tokenizer)

    if key in _STRUCTURE_CACHE:
        return _STRUCTURE_CACHE[key]

    vocab_size = len(tokenizer)

    token_texts = tokenizer.convert_ids_to_tokens(
        list(range(vocab_size))
    )

    paren_delta = [0] * vocab_size
    bracket_delta = [0] * vocab_size

    min_paren_prefix = [0] * vocab_size
    min_bracket_prefix = [0] * vocab_size

    has_separator = [False] * vocab_size

    # Separators used by the dataset/generation format.
    stop_chars = ["\n", ";", ","]

    for tid, text in enumerate(token_texts):

        if text is None:
            continue

        text = str(text)

        # We only care about the portion that can belong to the
        # first molecule. Anything after a separator/newline is
        # not part of the first molecule.
        cut_positions = [
            text.find(ch)
            for ch in stop_chars
            if text.find(ch) >= 0
        ]

        if cut_positions:

            cut = min(cut_positions)
            molecule_text = text[:cut]
            has_separator[tid] = True

        else:

            molecule_text = text

        p = 0
        b = 0
        min_p = 0
        min_b = 0

        for ch in molecule_text:

            if ch == "(":
                p += 1

            elif ch == ")":
                p -= 1
                min_p = min(min_p, p)

            elif ch == "[":
                b += 1

            elif ch == "]":
                b -= 1
                min_b = min(min_b, b)

        paren_delta[tid] = p
        bracket_delta[tid] = b

        min_paren_prefix[tid] = min_p
        min_bracket_prefix[tid] = min_b

    cache = {
        "paren_delta": paren_delta,
        "bracket_delta": bracket_delta,
        "min_paren_prefix": min_paren_prefix,
        "min_bracket_prefix": min_bracket_prefix,
        "has_separator": has_separator,
    }

    _STRUCTURE_CACHE[key] = cache

    return cache


# ============================================================
# REPEATED N-GRAM BLOCKING
# ============================================================

def block_repeated_ngrams(
    logits,
    generated_ids,
    n=4,
):
    """
    Prevent exact repeated n-gram loops.
    """

    if len(generated_ids) < n - 1:
        return logits

    current_prefix = tuple(
        generated_ids[-(n - 1):]
    )

    for i in range(
        len(generated_ids) - (n - 1)
    ):

        if tuple(
            generated_ids[i:i + n - 1]
        ) == current_prefix:

            banned_token = (
                generated_ids[i + n - 1]
            )

            logits[:, banned_token] = float(
                "-inf"
            )

    return logits


# ============================================================
# CFG DECODER
# ============================================================

@torch.no_grad()
def cfg_decode(
    model,
    tokenizer,
    conditional_prompt: str,
    unconditional_prompt: str,
    guidance_scale: float = 2.0,
    max_new_tokens: int = 128,
    min_new_tokens: int = 8,
    temperature: float = 1.0,
    top_p: float = 0.95,
    stop_newline: bool = True,
    stop_separators: bool = True,
    block_repeats: bool = True,
    repeat_ngram_size: int = 4,
    balance_parens: bool = True,
    verbose: bool = True,
):
    """
    Generate one molecule using classifier-free guidance.
    """

    model.eval()

    device = next(
        model.parameters()
    ).device

    # --------------------------------------------------------
    # Prompts
    # --------------------------------------------------------

    if verbose:

        print("\n" + "=" * 80)
        print("Conditional Prompt")
        print(repr(conditional_prompt))

        print("\n" + "=" * 80)
        print("Unconditional Prompt")
        print(repr(unconditional_prompt))

        print("\n" + "=" * 80)
        print(f"CFG scale      : {guidance_scale}")
        print(f"Temperature    : {temperature}")
        print(f"Top-p          : {top_p}")
        print(f"Max new tokens : {max_new_tokens}")
        print(f"Stop newline   : {stop_newline}")
        print(f"Stop separators: {stop_separators}")
        print(f"Block repeats  : {block_repeats}")
        print(f"Balance parens : {balance_parens}")
        print("=" * 80)

    # IMPORTANT:
    # Do not pass clean_up_tokenization_spaces because the local
    # GPT-NeoX tokenizer does not accept it in __call__.

    cond_inputs = tokenizer(
        conditional_prompt,
        return_tensors="pt",
    ).to(device)

    uncond_inputs = tokenizer(
        unconditional_prompt,
        return_tensors="pt",
    ).to(device)

    cond_ids = cond_inputs["input_ids"]
    cond_mask = cond_inputs["attention_mask"]

    uncond_ids = uncond_inputs["input_ids"]
    uncond_mask = uncond_inputs["attention_mask"]

    eos_id = tokenizer.eos_token_id

    newline_ids = set(
        tokenizer.encode(
            "\n",
            add_special_tokens=False,
        )
    )

    # --------------------------------------------------------
    # Token-aware structural metadata
    # --------------------------------------------------------

    structure = _build_structure_cache(
        tokenizer
    )

    paren_delta = structure["paren_delta"]
    bracket_delta = structure["bracket_delta"]
    min_paren_prefix = structure["min_paren_prefix"]
    min_bracket_prefix = structure["min_bracket_prefix"]

    # GPU tensors for fast structural masking.

    min_paren_tensor = torch.tensor(
        min_paren_prefix,
        dtype=torch.int16,
        device=device,
    )

    min_bracket_tensor = torch.tensor(
        min_bracket_prefix,
        dtype=torch.int16,
        device=device,
    )

    # --------------------------------------------------------
    # Generation state
    # --------------------------------------------------------

    generated_ids = []

    generated_pieces = []

    open_paren_count = 0
    open_bracket_count = 0

    for step in range(max_new_tokens):

        # ----------------------------------------------------
        # MODEL FORWARD
        # ----------------------------------------------------

        cond_outputs = model(
            input_ids=cond_ids,
            attention_mask=cond_mask,
        )

        uncond_outputs = model(
            input_ids=uncond_ids,
            attention_mask=uncond_mask,
        )

        cond_logits = (
            cond_outputs.logits[:, -1, :]
        )

        uncond_logits = (
            uncond_outputs.logits[:, -1, :]
        )

        # ----------------------------------------------------
        # CFG
        # ----------------------------------------------------

        guided_logits = classifier_free_guidance(
            cond_logits=cond_logits,
            uncond_logits=uncond_logits,
            guidance_scale=guidance_scale,
        )

        if temperature <= 0:
            raise ValueError(
                "temperature must be > 0"
            )

        guided_logits = (
            guided_logits / temperature
        )

        # ----------------------------------------------------
        # MINIMUM LENGTH
        # ----------------------------------------------------

        if step < min_new_tokens:

            if eos_id is not None:
                guided_logits[:, eos_id] = (
                    float("-inf")
                )

            for nid in newline_ids:
                guided_logits[:, nid] = (
                    float("-inf")
                )

        # ----------------------------------------------------
        # REPEATED N-GRAM BLOCKING
        # ----------------------------------------------------

        if block_repeats:

            guided_logits = (
                block_repeated_ngrams(
                    guided_logits,
                    generated_ids,
                    n=repeat_ngram_size,
                )
            )

        # ----------------------------------------------------
        # ROBUST STRUCTURAL MASK
        #
        # The minimum prefix balance is important.
        #
        # Example token:
        #
        #     ")C"
        #
        # At balance=0:
        #
        #     0 -> -1 -> 0
        #
        # This must be rejected because the first character
        # creates an unmatched ')'.
        # ----------------------------------------------------

        if balance_parens:

            invalid_paren = (
                min_paren_tensor
                + open_paren_count
                < 0
            )

            invalid_bracket = (
                min_bracket_tensor
                + open_bracket_count
                < 0
            )

            invalid = (
                invalid_paren
                | invalid_bracket
            )

            if invalid.any():

                guided_logits[
                    :,
                    invalid.nonzero(
                        as_tuple=False
                    ).flatten(),
                ] = float("-inf")

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        probs = F.softmax(
            guided_logits,
            dim=-1,
        )

        if (
            torch.isnan(probs).any()
            or torch.isinf(probs).any()
        ):
            raise RuntimeError(
                "Invalid sampling probabilities."
            )

        # ----------------------------------------------------
        # TOP-P
        # ----------------------------------------------------

        if top_p < 1.0:

            sorted_probs, sorted_indices = (
                torch.sort(
                    probs,
                    descending=True,
                    dim=-1,
                )
            )

            cumulative = torch.cumsum(
                sorted_probs,
                dim=-1,
            )

            mask = cumulative > top_p

            mask[..., 1:] = (
                mask[..., :-1].clone()
            )

            mask[..., 0] = False

            sorted_probs = (
                sorted_probs.masked_fill(
                    mask,
                    0.0,
                )
            )

            denominator = (
                sorted_probs.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)
            )

            sorted_probs = (
                sorted_probs / denominator
            )

            sampled = torch.multinomial(
                sorted_probs,
                num_samples=1,
            )

            next_token = (
                sorted_indices.gather(
                    -1,
                    sampled,
                )
            )

        else:

            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

        token_id = next_token.item()

        # ----------------------------------------------------
        # DECODE TOKEN PIECE
        # ----------------------------------------------------

        token_text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
        )

        # ----------------------------------------------------
        # DETERMINE FIRST MOLECULE PORTION
        # ----------------------------------------------------

        stop_position = None

        for ch in ["\n", ";", ","]:

            if (
                stop_newline
                and ch == "\n"
            ) or (
                stop_separators
                and ch in [";", ","]
            ):

                pos = token_text.find(ch)

                if pos >= 0:

                    if (
                        stop_position is None
                        or pos < stop_position
                    ):
                        stop_position = pos

        if stop_position is not None:

            molecule_piece = (
                token_text[:stop_position]
            )

            # Check whether the molecule portion
            # itself would break structural balance.

            candidate_paren = (
                open_paren_count
            )

            candidate_bracket = (
                open_bracket_count
            )

            structurally_valid = True

            if balance_parens:

                for ch in molecule_piece:

                    if ch == "(":
                        candidate_paren += 1

                    elif ch == ")":

                        candidate_paren -= 1

                        if candidate_paren < 0:

                            structurally_valid = False
                            break

                    elif ch == "[":
                        candidate_bracket += 1

                    elif ch == "]":

                        candidate_bracket -= 1

                        if candidate_bracket < 0:

                            structurally_valid = False
                            break

            # If the molecule portion is valid, finish.
            if structurally_valid:

                if molecule_piece:
                    generated_pieces.append(
                        molecule_piece
                    )

                if verbose:

                    print(
                        f"Step {step:03d} | "
                        f"ID={token_id} | "
                        f"Token={repr(token_text)}"
                    )

                    print(
                        "\nReached molecule separator."
                    )

                break

            # Otherwise reject this token and sample again.
            probs[:, token_id] = 0.0

            probs = probs / probs.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)

            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

            token_id = next_token.item()

            token_text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )

        # ----------------------------------------------------
        # EOS
        # ----------------------------------------------------

        if (
            eos_id is not None
            and token_id == eos_id
        ):

            if verbose:
                print(
                    "\nReached EOS."
                )

            break

        # Exact newline fallback.
        if (
            stop_newline
            and token_id in newline_ids
        ):

            if verbose:
                print(
                    "\nReached newline. "
                    "Stopping after first molecule."
                )

            break

        # If this token contained a separator and we already
        # stopped, don't add the token to the model context.
        if stop_position is not None:
            break

        # ----------------------------------------------------
        # STRUCTURAL STATE UPDATE
        # ----------------------------------------------------

        if balance_parens:

            open_paren_count += (
                paren_delta[token_id]
            )

            open_bracket_count += (
                bracket_delta[token_id]
            )

        # ----------------------------------------------------
        # SAVE TOKEN
        # ----------------------------------------------------

        generated_ids.append(token_id)

        generated_pieces.append(
            token_text
        )

        if verbose:

            print(
                f"Step {step:03d} | "
                f"ID={token_id} | "
                f"Token={repr(token_text)} | "
                f"paren={open_paren_count} | "
                f"bracket={open_bracket_count}"
            )

        # ----------------------------------------------------
        # APPEND TO BOTH CFG BRANCHES
        # ----------------------------------------------------

        cond_ids = torch.cat(
            [
                cond_ids,
                next_token,
            ],
            dim=1,
        )

        uncond_ids = torch.cat(
            [
                uncond_ids,
                next_token,
            ],
            dim=1,
        )

        cond_mask = torch.cat(
            [
                cond_mask,
                torch.ones(
                    (
                        cond_mask.size(0),
                        1,
                    ),
                    dtype=cond_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )

        uncond_mask = torch.cat(
            [
                uncond_mask,
                torch.ones(
                    (
                        uncond_mask.size(0),
                        1,
                    ),
                    dtype=uncond_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )

    # ========================================================
    # FINAL STRING
    # ========================================================

    generated = "".join(
        generated_pieces
    ).strip()

    if verbose:

        print("\nGenerated IDs:")
        print(generated_ids)

        print("\nGenerated SMILES:")
        print(generated)

        print("=" * 80)

    return generated
