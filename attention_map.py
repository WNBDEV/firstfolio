"""
Look inside the model's head.

For every character, at every layer, attention computes a probability table —
"who should I listen to?" — and normally throws it away the instant it is
used. This file keeps those tables and makes them readable, because for a
character-level model they are surprisingly legible:

    * a head that snaps to the last newline (the "a new speaker line starts
      here" detector)
    * heads that sit one or two characters back (local spelling habits)
    * heads that reach far back (scene structure, who is present)

Run:
    .venv/bin/python attention_map.py --prompt "\\nROMEO:"
    .venv/bin/python attention_map.py --prompt "ROMEO:" --head 2 --grid
"""

import argparse
import os

# same reason as numpy_infer.py: tiny matrices, one honest thread.
# Must run BEFORE numpy loads.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from numpy_infer import load_model, logits_for

RAMP = " .:-=+*#%@"     # attention density: invisible -> solid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="\nROMEO:")
    p.add_argument("--layer", type=int, default=-1, help="which layer to draw (-1 = deepest)")
    p.add_argument("--head", type=int, default=None, help="also draw one head's full chart/map")
    p.add_argument("--grid", action="store_true",
                   help="with --head: ASCII heatmap of that head's whole attention matrix")
    p.add_argument("--max-chars", type=int, default=60)
    args = p.parse_args()

    cfg, w, chars = load_model()
    # shells don't expand \n inside quotes — accept the two characters "\n" as a newline
    prompt = args.prompt.replace("\\n", "\n")
    stoi = {c: i for i, c in enumerate(chars)}
    for c in prompt:
        assert c in stoi, f"character {c!r} is not in the model's vocabulary"

    ids = np.array([stoi[c] for c in prompt], dtype=np.int64)
    collect = []                        # one (H, T, T) listening-plan tensor per layer
    logits_for(ids, cfg, w, collect=collect)

    T, L, H = len(ids), cfg["n_layer"], cfg["n_head"]
    print(f"prompt {prompt!r} ({T} chars) | {L} layers x {H} heads = {L * H} attention heads")

    # ---- Part A: each head's personality, in one number --------------------
    # "mean distance looked back": attention-weighted average of how far back
    # a head reaches, averaged over all characters. Small = local head,
    # large = long-range head.
    offset_back = np.arange(T)[:, None] - np.arange(T)[None, :]     # (T, T)
    print("\nmean characters looked back (rows = layers, cols = heads):")
    print("        " + " ".join(f"h{h}".rjust(6) for h in range(H)))
    for l in range(L):
        P = collect[l]                                              # (H, T, T)
        mean_back = (P * offset_back[None]).sum(-1).mean(-1)        # (H,)
        print(f"layer {l}: " + " ".join(f"{v:6.1f}" for v in mean_back))

    # ---- Part B: what the final character listens to, per head -------------
    l = args.layer if args.layer >= 0 else L + args.layer
    final_row = collect[l][:, -1, :]                                # (H, T)
    print(f"\nwhat the final character {chars[ids[-1]]!r} listens to (layer {l}):")
    for h in range(H):
        r = final_row[h]
        top = np.argsort(-r)[:3]
        parts = ", ".join(
            f"{chars[ids[j]]!r} {r[j]*100:.0f}% (-{T - 1 - j} back)"
            for j in top if r[j] > 0.005
        )
        print(f"  head {h}: {parts}")

    # ---- Part C: one head, drawn as bars over the prompt -------------------
    if args.head is not None:
        r = final_row[args.head]
        r = r / max(r.max(), 1e-9)
        print(f"\nlayer {l}, head {args.head} — attention of the final character:")
        for j in range(max(0, T - args.max_chars), T):
            width = int(round(r[j] * 30))
            print(f"  -{T - 1 - j:>3} {chars[ids[j]]!r:>5} {'#' * width}")

        # ---- Part D: optional full heatmap (rows = chars, cols = sources) ---
        if args.grid:
            G = collect[l][args.head][:40, :40]
            G = G / max(G.max(), 1e-9)
            print(f"\nsame head, full map (row = character, cols = who it reads; "
                  f"first {G.shape[0]} characters):")
            for i in range(G.shape[0]):
                line = "".join(RAMP[min(int(v * len(RAMP)), len(RAMP) - 1)] for v in G[i])
                print(f"  {chars[ids[i]]!r:>4}|{line}|")


if __name__ == "__main__":
    main()
