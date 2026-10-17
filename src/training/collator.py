"""
CFG Data Collator

Supports:
    - single-property conditioning
    - pair conditioning
    - triple conditioning
    - all-property conditioning
    - random conditioning over all modes
    - explicit EOS after every molecule
    - unconditional CFG examples via <NULL>
"""

import random
import torch


class CFGDataCollator:

    def __init__(
        self,
        tokenizer,
        max_length=512,
        dropout_prob=0.20,
        conditioning_mode="random",
    ):

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dropout_prob = dropout_prob
        self.conditioning_mode = conditioning_mode

        self.molstart_id = (
            tokenizer.convert_tokens_to_ids("<molstart>")
        )

        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer must have an EOS token."
            )

        self.eos_token = tokenizer.eos_token

    # ========================================================
    # PROMPT BUILDER
    # ========================================================

    def build_prompt(
        self,
        smiles,
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
        prompt += str(smiles)

        # IMPORTANT:
        # Explicitly teach the model where the molecule ends.
        prompt += self.eos_token

        return prompt

    # ========================================================
    # PROPERTY SELECTION
    # ========================================================

    def choose_properties(
        self,
        qed,
        logp,
        tpsa,
        sas,
    ):

        props = {
            "qed": qed,
            "logp": logp,
            "tpsa": tpsa,
            "sas": sas,
        }

        names = list(props.keys())

        mode = self.conditioning_mode

        # ----------------------------------------------------
        # ALL
        # ----------------------------------------------------

        if mode == "all":

            return props

        # ----------------------------------------------------
        # SINGLE
        # ----------------------------------------------------

        if mode == "single":

            keep = random.choice(names)

            return {
                k: props[k] if k == keep else None
                for k in names
            }

        # ----------------------------------------------------
        # PAIR
        # ----------------------------------------------------

        if mode == "pair":

            keep = random.sample(names, 2)

            return {
                k: props[k] if k in keep else None
                for k in names
            }

        # ----------------------------------------------------
        # TRIPLE
        # ----------------------------------------------------

        if mode == "triple":

            keep = random.sample(names, 3)

            return {
                k: props[k] if k in keep else None
                for k in names
            }

        # ----------------------------------------------------
        # RANDOM
        #
        # Randomly expose:
        #   single, pair, triple, or all
        #
        # This teaches the model to handle multiple property
        # combinations rather than only the full condition.
        # ----------------------------------------------------

        if mode == "random":

            chosen_mode = random.choice(
                [
                    "single",
                    "pair",
                    "triple",
                    "all",
                ]
            )

            if chosen_mode == "single":

                keep = random.choice(names)

                return {
                    k: props[k] if k == keep else None
                    for k in names
                }

            if chosen_mode == "pair":

                keep = random.sample(names, 2)

                return {
                    k: props[k] if k in keep else None
                    for k in names
                }

            if chosen_mode == "triple":

                keep = random.sample(names, 3)

                return {
                    k: props[k] if k in keep else None
                    for k in names
                }

            return props

        raise ValueError(
            f"Unknown conditioning_mode: {mode}"
        )

    # ========================================================
    # COLLATE
    # ========================================================

    def __call__(self, batch):

        prompts = []

        for sample in batch:

            # ------------------------------------------------
            # CFG UNCONDITIONAL DROPOUT
            # ------------------------------------------------

            if random.random() < self.dropout_prob:

                prompt = (
                    "<NULL>\n"
                    "<molstart>"
                    + str(sample["smiles"])
                    + self.eos_token
                )

                prompts.append(prompt)

                continue

            # ------------------------------------------------
            # PROPERTY CONDITIONING
            # ------------------------------------------------

            selected = self.choose_properties(
                qed=sample["qed"],
                logp=sample["logp"],
                tpsa=sample["tpsa"],
                sas=sample["sas"],
            )

            prompt = self.build_prompt(
                smiles=sample["smiles"],
                qed=selected["qed"],
                logp=selected["logp"],
                tpsa=selected["tpsa"],
                sas=selected["sas"],
            )

            prompts.append(prompt)

        # ====================================================
        # TOKENIZE
        # ====================================================

        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # ====================================================
        # LABELS
        # ====================================================

        labels = encoded["input_ids"].clone()

        # Ignore padding.
        labels[
            encoded["attention_mask"] == 0
        ] = -100

        # Ignore everything through <molstart>.
        for i in range(len(labels)):

            input_ids = labels[i]

            mol_positions = (
                (encoded["input_ids"][i] == self.molstart_id)
                .nonzero(as_tuple=True)[0]
            )

            if len(mol_positions) > 0:

                mol_pos = mol_positions[0].item()

                labels[
                    i,
                    :mol_pos + 1
                ] = -100

        encoded["labels"] = labels

        return encoded
