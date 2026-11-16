"""
Your model, with ZERO AI libraries.

sample.py loads the model through PyTorch. This file does the same job using
only numpy — no torch, no GPU, no framework. Same weights, same math, same
Shakespeare. (Output differs by a hair from sample.py only because numpy
lacks the exact `erf` function torch's GELU uses, so we use its standard tanh
twin — two calculators agreeing to two decimal places.)

Why this file exists:

1. PROOF. "An LLM" is not a spell only a framework can cast. It is a short
   list of matrix multiplications. Delete the framework and it still writes
   Shakespeare.

2. BRIDGE. Every function below is ONE plain-English sentence from our
   walkthrough, written out in code:

     "every character gets a list of 384 numbers"        -> the first two lines of logits_for()
     "look back at what's there, take the useful bits"   -> attention()
     "digest what you gathered"                          -> mlp()
     "do that cycle six times"                           -> the for-loop in logits_for()
     "put a probability on each of the 65 characters"    -> the last line of logits_for()

   Read it top to bottom, once. It is ~130 lines and every line is a sentence
   you have already seen happen.

Run:
    .venv/bin/python numpy_infer.py --prompt "ROMEO:" --show-probs
    .venv/bin/python numpy_infer.py --temperature 1.3     # the loose dice, numpy edition
"""

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "firstfolio_weights.npz")


# --------------------------------------------------------------- small helpers
def softmax(x, axis=-1):
    """Turn any list of scores into probabilities that sum to 1."""
    x = x - x.max(axis=axis, keepdims=True)   # stability trick: shift before exponentiating
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, gain, bias):
    """Keep every character's stat sheet at a healthy, consistent size:
    re-center it (mean 0) and re-scale it (spread 1), then apply a learned
    fine-tuning of that scaling. Without this, 6 layers of additions would
    drift the numbers into useless ranges."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * gain + bias


def gelu(x):
    """The 'digest' nonlinearity: without a bend somewhere, six rounds of
    multiplying-and-adding collapse into ONE multiply-and-add. This is the
    standard tanh-flavored GELU (numpy has no exact erf; torch's twin differs
    from this by less than the third decimal place)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


# ------------------------------------------------------------- the two organs
def attention(x, w, i, cfg):
    """LOOK BACK AND BLEND — one round of gathering for every character.

    Each character holds three lists, all learned during training:
      query = "what am I looking for?",
      key   = "what do I advertise that I contain?",
      value = "what do I hand over if someone attends to me?"
    Every query is scored against every key; the scores become probabilities
    ('who should I listen to?'); the output is that weighted mix of values.
    """
    T, C = x.shape
    H = cfg["n_head"]                          # how many independent look-backs per round
    hd = C // H                                # each head sees a 64-number slice of the sheet

    h = layer_norm(x, w[f"h{i}_ln1_w"], w[f"h{i}_ln1_b"])
    # (torch stores linear layers as [outputs x inputs]; numpy matmul wants
    # [inputs x outputs], so every linear weight gets one .T to turn sideways)
    qkv = h @ w[f"h{i}_attn_w"].T              # one matmul produces all queries, keys, values
    q, k, v = np.split(qkv, 3, axis=1)

    # Split the 384 numbers into 6 heads of 64: each head does its own look-back.
    q = q.reshape(T, H, hd).transpose(1, 0, 2)  # (T, C) -> (H, T, hd)
    k = k.reshape(T, H, hd).transpose(1, 0, 2)
    v = v.reshape(T, H, hd).transpose(1, 0, 2)

    # Score every query against every key, scaled so the scores stay in a
    # sane range (raw dot products grow with size and would flatten to
    # winner-takes-all).
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(hd)          # (H, T, T)

    # THE RULE: a character may not look at characters after it. The future is
    # the strictly-upper triangle; -inf becomes exactly 0 probability.
    future = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(future[None, :, :], -np.inf, scores)

    probs = softmax(scores, axis=-1)            # each row sums to 1: a listening plan
    out = probs @ v                             # the BLEND: weighted mix of what others offer

    out = out.transpose(1, 0, 2).reshape(T, C)  # merge the 6 heads back into one 384-sheet
    return out @ w[f"h{i}_attn_proj_w"].T


def mlp(x, w, i):
    """DIGEST — one round of private thinking per character.

    Attention moved information BETWEEN characters; the MLP lets each
    character think about what it gathered, alone, in a 4x-wider workspace.
    This is where two-thirds of the parameters (and most of the 'knowing')
    live."""
    h = layer_norm(x, w[f"h{i}_ln2_w"], w[f"h{i}_ln2_b"])
    h = h @ w[f"h{i}_mlp_fc_w"].T      # widen: 384 -> 1536, the thinking space
    h = gelu(h)                        # the bend
    return h @ w[f"h{i}_mlp_proj_w"].T # squeeze back: 1536 -> 384


# ------------------------------------------------------------ the whole model
def logits_for(ids, cfg, w):
    """The entire model, top to bottom:

    characters -> stat sheets -> [look back and blend + digest] x 6
               -> 65 probabilities per position.

    Returns logits[T, 65]: bigger number = the model considers that character
    more likely to come next at that position.
    """
    T = len(ids)
    # "every character gets a list of 384 numbers" — plus a position sheet,
    # because attention by itself can't tell 'R OME' from 'OME R' (order
    # must be injected here or the model can't know what 'next' means).
    x = w["wte"][ids] + w["wpe"][:T]

    for i in range(cfg["n_layer"]):     # six rounds of refine
        x = x + attention(x, w, i, cfg) # gather — ADDED, not replaced: each round only
        x = x + mlp(x, w, i)            # digest  — annotates the sheet, never rewrites it

    x = layer_norm(x, w["ln_f_w"], w["ln_f_b"])
    # "put a probability on each of 65 characters" — and note the matrix we
    # read out with: it is the SAME sheet that turned characters into numbers
    # at the start, just used sideways. One dictionary for reading and writing
    # (this is 'weight tying', and it saved ~30% of GPT-2's size).
    return x @ w["wte"].T


# ------------------------------------------------------------- the dice roller
def generate(prompt, n_new, temperature, top_k, cfg, w, chars, show_probs=False):
    """Guess one character, print it, add it to the text, guess again.

    Simple but honest: every step re-reads the ENTIRE context from scratch.
    Real engines keep a cache of the attention math to avoid re-computing it
    (the 'KV cache' — remember this when generation feels slow)."""
    stoi = {c: i for i, c in enumerate(chars)}
    for c in prompt:
        assert c in stoi, f"character {c!r} is not in the model's 65-character vocabulary"

    ids = [stoi[c] for c in prompt]
    shown = 0
    for step in range(n_new):
        ctx = np.array(ids[-cfg["block_size"]:], dtype=np.int64)  # it can only see back this far
        logits = logits_for(ctx, cfg, w)[-1].astype(np.float64)   # the LAST position does the guessing

        logits /= temperature   # >1 = roll the dice loose, <1 = roll them tight
        if top_k:
            cutoff = np.partition(logits, -top_k)[-top_k]         # keep only the k most plausible
            logits[logits < cutoff] = -np.inf                     # the rest: impossible
        probs = softmax(logits)

        if show_probs and shown < 5:
            top = np.argsort(-probs)[:8]
            table = " | ".join(f"{chars[j]!r} {probs[j]*100:5.1f}%" for j in top)
            print(f"guess {step + 1:3d}: {table}")
            shown += 1

        ids.append(int(np.random.choice(cfg["vocab_size"], p=probs)))  # ROLL the dice (not argmax)
    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="\n", help="text to continue ('\\n' = start of a line)")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--show-probs", action="store_true", help="print the 65-sided dice for the first guesses")
    args = p.parse_args()

    np.random.seed(args.seed)
    cfg, w, chars = load_model()
    n_params = sum(a.size for a in w.values())
    print(f"pure numpy | {n_params/1e6:.2f}M params | {cfg['n_layer']} layers, "
          f"{cfg['n_head']} heads, width {cfg['n_embd']} | not a single line of torch")

    ids = generate(
        args.prompt, args.max_new_tokens, args.temperature, args.top_k,
        cfg, w, chars, show_probs=args.show_probs,
    )
    print("-" * 60)
    print("".join(chars[i] for i in ids))
    print("-" * 60)


def load_model():
    z = np.load(WEIGHTS)
    cfg = {k: int(z[k]) for k in ("n_layer", "n_head", "n_embd", "block_size", "vocab_size")}
    chars = [str(c) for c in z["chars"]]
    w = {k: z[k] for k in z.files if k not in set(cfg) | {"chars"}}
    return cfg, w, chars


if __name__ == "__main__":
    main()
