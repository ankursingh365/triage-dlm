"""
Instrumented denoising for diffusion language models.
=====================================================

The core of TRIAGE. Everything downstream - features, baselines, the
per-failure-mode evaluation - reads what this module records.

Why "patch" is a misnomer
-------------------------
The file is named `logging_patch.py` because the original plan assumed we would
monkeypatch a sampler shipped with the model. Step 3 established that LLaDA-family
checkpoints ship no sampler at all: their remote code (`modeling_lladamoe.py`) is
the architecture only, and the sampling loop lives separately in ML-GSAI/LLaDA's
`generate.py`. So this module implements the loop directly, following that
reference, with instrumentation built in.

That is the better arrangement: nothing breaks when a model repository updates,
there is no fragile injection into third-party code, and the result is
self-contained and citable.

Note for Phase C: Dream-7B *does* ship `generation_utils.py` with its own
`diffusion_generate`. It will need an adapter, not this loop unmodified.

What gets recorded, and what does not
-------------------------------------
Four arrays of shape `(steps, gen_length)`, and nothing else:

    pred_ids     int32    argmax token id at this position this round
    pred_probs   float32  probability the model assigned to that argmax
    entropy      float32  Shannon entropy of the full distribution, in nats
    mask_state   uint8    1 if the position was still masked entering this round

At 32 rounds x 64 positions that is 26 KB per question. Charter Warning #1: the
full logits would be 32 x 256 x 126,464 x 4 bytes = **4.1 GB per question**, which
is not merely wasteful but would make the project impossible to store.

`pred_ids` is captured BEFORE the reference implementation's line 197
(`x0 = torch.where(mask_index, x0, x)`), which overwrites predictions at revealed
positions with the committed token. Keeping the raw argmax is what makes
revealed-position *regret* measurable - see `regret_per_round`. Every published
trajectory method logs after that line and therefore cannot see it.

These four arrays are sufficient. The committed sequence is *derivable* rather
than stored: a position revealed during round r holds `pred_ids[r, i]` forever
after, and `mask_state` says which round that was. `reconstruct_final` performs
that derivation, which is precisely the Step 6 sanity check - if the
reconstruction does not match the model's actual output character for character,
the log is untrustworthy and every downstream number is meaningless.

Charter warnings implemented
----------------------------
#1  Never log full logits.                    - four scalars per position
#2  EOS and padding distort features.         - `eos_id` recorded so features.py
                                                can exclude them; not excluded
                                                here, because logging should
                                                record what happened and leave
                                                interpretation to features.
#3  Prompt positions have no trajectory.      - logits sliced to the generation
                                                region before any computation
#4  Flips live at masked positions.           - `flips_per_round` compares
                                                predictions only where
                                                `mask_state == 1`

References
----------
LLaDA          Nie et al., arXiv:2502.09992 (NeurIPS 2025 Oral)
Sampler        github.com/ML-GSAI/LLaDA, generate.py lines 101-204
TraceDet       Chang et al., arXiv:2510.01274 (ICLR 2026)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np


# ===========================================================================
# The record
# ===========================================================================

@dataclass
class Trajectory:
    """Everything recorded for one question.

    Array shapes are `(steps, gen_length)`. Row `r` describes the state
    *entering* round `r`, together with what the model predicted during it.
    """

    # --- the four logged arrays -------------------------------------------
    pred_ids: np.ndarray      # int32   raw argmax, BEFORE the reveal overwrite
    pred_probs: np.ndarray    # float32 probability of that argmax
    entropy: np.ndarray       # float32 entropy of the full distribution, nats
    mask_state: np.ndarray    # uint8   1 = still masked entering this round

    # --- ground truth for the Step 6 check ---------------------------------
    final_ids: np.ndarray     # int32 (gen_length,) what the model actually output

    # --- metadata, all of it needed to reproduce the run -------------------
    question: str = ""
    question_id: str = ""
    prompt_ids: list = field(default_factory=list)
    model_id: str = ""
    quant_config: str = ""    # which of A/B/C loaded; B changes the logits
    seed: int = 0
    steps: int = 0
    gen_length: int = 0
    block_length: int = 0
    temperature: float = 0.0
    mask_id: int = -1
    eos_id: int = -1
    vocab_size: int = 0
    elapsed_s: float = 0.0

    # ---------------------------------------------------------------- derived

    def revealed_at(self) -> np.ndarray:
        """Round index at which each position was revealed.

        Returns `(gen_length,)` int32. A position masked entering round r and
        unmasked entering round r+1 was revealed *during* round r. Positions
        still masked at the end get `steps - 1`, since the final round reveals
        whatever remains.

        This is the raw material for settle-round features in Step 19. Read it
        against the reveal schedule, not as a free measurement: the number of
        positions revealed per round is fixed in advance by
        `get_num_transfer_tokens`, so an early settle round partly reflects the
        schedule rather than the model's confidence alone.
        """
        steps, gen_length = self.mask_state.shape
        out = np.full(gen_length, steps - 1, dtype=np.int32)
        for i in range(gen_length):
            col = self.mask_state[:, i]
            unmasked = np.flatnonzero(col == 0)
            if unmasked.size:
                # first round entered unmasked => revealed during the one before
                out[i] = max(int(unmasked[0]) - 1, 0)
        return out

    def committed_ids(self) -> np.ndarray:
        """The committed token at every position, at every round.

        `(steps, gen_length)` int32. Before a position is revealed its entry is
        `mask_id`; from the revealing round onward it holds the token that was
        fixed. Derived, never stored - which is what makes the Step 6 check a
        genuine test rather than a tautology.
        """
        steps, gen_length = self.mask_state.shape
        out = np.full((steps, gen_length), self.mask_id, dtype=np.int32)
        rev = self.revealed_at()
        for i in range(gen_length):
            r = int(rev[i])
            out[r:, i] = self.pred_ids[r, i]
        return out

    def reconstruct_final(self) -> np.ndarray:
        """Rebuild the model's output using only the logged arrays.

        THE STEP 6 SANITY CHECK. If this does not match `final_ids` exactly, the
        log does not describe what the model did and every downstream number is
        garbage. Run it on 50 questions before trusting any of Phase C.
        """
        rev = self.revealed_at()
        return np.array(
            [self.pred_ids[int(rev[i]), i] for i in range(self.pred_ids.shape[1])],
            dtype=np.int32,
        )

    def flips_per_round(self) -> np.ndarray:
        """Prediction changes at STILL-MASKED positions, per round.

        Charter Warning #4. A revealed token cannot change - the confidence
        freeze (`torch.where(mask_index, x0_p, -np.inf)`) makes it unselectable
        forever - so a flip can only mean the argmax at a position that is still
        masked differing from the previous round. Counting changes at revealed
        positions instead would return identically zero.

        Returns `(steps,)` int32; element 0 is always 0, having no predecessor.
        """
        steps = self.pred_ids.shape[0]
        out = np.zeros(steps, dtype=np.int32)
        for r in range(1, steps):
            still_masked = self.mask_state[r] == 1
            out[r] = int(np.sum(
                self.pred_ids[r][still_masked] != self.pred_ids[r - 1][still_masked]
            ))
        return out

    def regret_per_round(self) -> np.ndarray:
        """Revealed positions whose committed token differs from the current argmax.

        **Finding B.** Reference line 197 overwrites the model's prediction at
        revealed positions, so anything logged afterwards cannot see what the
        model would now say about a token it has already fixed. We keep the raw
        argmax, so we can.

        Interpretation is open, and deliberately left to Step 19. Regret may be
        zero, which would independently confirm that a locked-in answer really is
        locked in throughout the network. Or it may be non-zero and correlate
        with wrongness, which would be a hesitation signal that exists but cannot
        express itself - and that no published method can observe.

        A control is required before drawing any conclusion: regret on *correct*
        answers establishes the baseline rate. Without it, a high number means
        only that the model's reconstruction of its own output is imperfect.

        Returns `(steps,)` int32.
        """
        steps = self.pred_ids.shape[0]
        committed = self.committed_ids()
        out = np.zeros(steps, dtype=np.int32)
        for r in range(steps):
            revealed = self.mask_state[r] == 0
            if revealed.any():
                out[r] = int(np.sum(
                    committed[r][revealed] != self.pred_ids[r][revealed]
                ))
        return out

    def content_mask(self) -> np.ndarray:
        """Positions that carry real content, per round. `(steps, gen_length)` bool.

        Charter Warning #2. LLaDA's SFT data pads with EOS, which causes EOS to
        settle very early under low-confidence remasking. A settle-round feature
        averaged over EOS positions measures that padding artefact rather than
        the model's behaviour. Step 2c saw it directly: 7 of 16 positions were
        `<|endoftext|>` at entropy 0.0003.

        Excluded here so `features.py` does not have to rediscover the rule, but
        the underlying arrays keep everything - logging records what happened,
        interpretation happens downstream.
        """
        return (self.pred_ids != self.eos_id) & (self.pred_ids != self.mask_id)

    # ------------------------------------------------------------------- io

    def save(self, path: Path) -> Path:
        """Write to a compressed .npz. One file per question.

        One file per question rather than per batch: it makes Phase C resumable
        by simply checking whether the file exists, and a crash mid-write can
        damage at most one question.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {k: v for k, v in asdict(self).items()
                if not isinstance(v, np.ndarray)}
        np.savez_compressed(
            path,
            pred_ids=self.pred_ids,
            pred_probs=self.pred_probs,
            entropy=self.entropy,
            mask_state=self.mask_state,
            final_ids=self.final_ids,
            meta=np.array(json.dumps(meta)),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "Trajectory":
        z = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(
            pred_ids=z["pred_ids"], pred_probs=z["pred_probs"],
            entropy=z["entropy"], mask_state=z["mask_state"],
            final_ids=z["final_ids"], **meta,
        )


# ===========================================================================
# The sampler
# ===========================================================================

def get_num_transfer_tokens(mask_index, steps: int):
    """Positions to reveal on each round. Reference `generate.py` line 28.

    Masked count divided evenly across rounds, remainder to the earliest rounds
    so the totals are exact. **Fixed in advance and not adaptive** - the model's
    confidence chooses *which* positions are revealed, never *how many*.
    """
    import torch
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    out = torch.zeros(mask_num.size(0), steps,
                      device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        out[i, : remainder[i]] += 1
    return out


def generate_with_logging(model, tokenizer, prompt_ids, *,
                          gen_length: int = 64,
                          steps: int = 32,
                          block_length: int = 64,
                          temperature: float = 0.0,
                          question: str = "",
                          question_id: str = "",
                          quant_config: str = "",
                          seed: int = 0,
                          mask_id: Optional[int] = None):
    """Run the denoising loop, recording the trajectory.

    Returns `(final_text, Trajectory)`.

    `temperature=0` means pure argmax - no Gumbel noise, fully deterministic.
    Use it unless an experiment specifically needs sampling; reproducibility is
    worth more here than diversity.

    Memory note: logits are sliced to the generation region before any softmax.
    That is charter Warning #3, and also a hard requirement on a 6 GB card - a
    full-sequence fp32 softmax over a 157k vocabulary does not fit in the
    headroom left after the model loads.

    Precision note: entropy and probabilities are computed in **float32**
    throughout. Development runs on an Ada laptop GPU with bf16; production runs
    on Kaggle's T4 without it. Casting to fp32 on both machines is what keeps the
    two sets of numbers comparable, and a bf16 softmax over a large vocabulary
    loses enough tail precision to shift entropy measurably.
    """
    import time
    import torch

    device = model.device

    # The mask id must never be None. LLaDA-8B's mask token (126336) is a
    # RESERVED token that its tokenizer does not declare, so
    # `tokenizer.mask_token_id` returns None there and a None mask id produces a
    # silently wrong run rather than a crash: no position ever compares equal to
    # it, nothing is ever treated as masked, and the trajectory is empty while
    # every check still passes. Callers should pass the value from
    # `model_utils.resolve_mask_id`, which consults the tokenizer first and a
    # table of known models second, and raises rather than guessing.
    if mask_id is None:
        mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        raise ValueError(
            "mask_id is None. This tokenizer does not declare a mask token; "
            "pass mask_id=model_utils.resolve_mask_id(tokenizer, model_id)."
        )
    mask_id = int(mask_id)
    eos_id = tokenizer.eos_token_id
    prompt_len = len(prompt_ids)

    assert gen_length % block_length == 0, "gen_length must divide by block_length"
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    # x = [prompt] + [MASK] * gen_length
    x = torch.full((1, prompt_len + gen_length), mask_id,
                   dtype=torch.long, device=device)
    x[0, :prompt_len] = torch.tensor(prompt_ids, device=device)

    # Pre-allocate. Nothing grows inside the loop.
    pred_ids = np.zeros((steps, gen_length), dtype=np.int32)
    pred_probs = np.zeros((steps, gen_length), dtype=np.float32)
    entropy_arr = np.zeros((steps, gen_length), dtype=np.float32)
    mask_state = np.zeros((steps, gen_length), dtype=np.uint8)

    t0 = time.time()
    round_idx = 0

    for block in range(num_blocks):
        b0 = prompt_len + block * block_length
        b1 = prompt_len + (block + 1) * block_length

        block_mask = (x[:, b0:b1] == mask_id)
        n_transfer = get_num_transfer_tokens(block_mask, steps_per_block)

        for step in range(steps_per_block):
            # State entering this round.
            mask_index_gen = (x[:, prompt_len:] == mask_id)      # (1, G)
            mask_state[round_idx] = mask_index_gen[0].cpu().numpy().astype(np.uint8)

            with torch.no_grad():
                logits = model(x).logits                          # (1, L, V)
                gen_logits = logits[:, prompt_len:, :].float()    # Warning #3
                probs = torch.softmax(gen_logits, dim=-1)
                ent = -(probs * torch.log(probs + 1e-12)).sum(-1)

                if temperature > 0:
                    # Gumbel-max sampling, reference generate.py line 14.
                    noise = torch.rand_like(gen_logits)
                    gumbel = (-torch.log(noise + 1e-20)) ** temperature
                    x0_gen = (gen_logits.exp() / gumbel).argmax(-1)
                else:
                    x0_gen = gen_logits.argmax(-1)                # (1, G)

                x0_p = probs.gather(-1, x0_gen.unsqueeze(-1)).squeeze(-1)
                del gen_logits, probs, logits

            # RECORD HERE - before the reveal, and before any equivalent of
            # reference line 197. `pred_ids` therefore holds the raw argmax at
            # every position including revealed ones, which is what makes
            # regret measurable.
            pred_ids[round_idx] = x0_gen[0].cpu().numpy().astype(np.int32)
            pred_probs[round_idx] = x0_p[0].float().cpu().numpy()
            entropy_arr[round_idx] = ent[0].float().cpu().numpy()

            # The reveal, following reference lines 195-203.
            sched = x0_p.clone()
            sched[:, (b1 - prompt_len):] = -float("inf")          # future blocks
            neg_inf = torch.tensor(-float("inf"), device=device)
            confidence = torch.where(mask_index_gen, sched, neg_inf)

            k = int(n_transfer[0, step].item())
            if k > 0:
                _, sel = torch.topk(confidence[0], k=k)
                gen_slice = x[:, prompt_len:]
                gen_slice[0, sel] = x0_gen[0, sel]
                x[:, prompt_len:] = gen_slice

            round_idx += 1

    final_ids = x[0, prompt_len:].cpu().numpy().astype(np.int32)
    final_text = tokenizer.decode(final_ids.tolist())

    traj = Trajectory(
        pred_ids=pred_ids, pred_probs=pred_probs, entropy=entropy_arr,
        mask_state=mask_state, final_ids=final_ids,
        question=question, question_id=question_id,
        prompt_ids=list(prompt_ids),
        model_id=getattr(getattr(model, "config", None), "_name_or_path", ""),
        quant_config=quant_config, seed=seed,
        steps=steps, gen_length=gen_length, block_length=block_length,
        temperature=temperature, mask_id=int(mask_id), eos_id=int(eos_id),
        vocab_size=len(tokenizer), elapsed_s=time.time() - t0,
    )
    return final_text, traj