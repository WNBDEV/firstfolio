"""
Phase 4 — sampling: load a trained checkpoint and generate text.

The generation loop lives in GPT.generate (model.py) — this file just handles
the checkpoint round-trip: rebuild the exact architecture from the config
saved inside the checkpoint, load the weights, then hand over to the model.

    python sample.py --preset full
    python sample.py --preset full --prompt "ROMEO:" --max-new-tokens 800
    python sample.py --preset debug                       # sample the quick run
    python sample.py --temperature 1.2                    # unhinged mode
"""

import argparse
import json
import os

import torch

from model import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "shakespeare")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="full", help="which ckpt_<preset>.pt to load")
    p.add_argument("--ckpt", default=None, help="explicit checkpoint path (overrides --preset)")
    p.add_argument("--prompt", default="\n",
                   help="text to condition on ('\\n' = start of a new line/scene). "
                        "Note: every character must exist in the 65-char vocab.")
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    ckpt_path = args.ckpt or os.path.join(HERE, f"ckpt_{args.preset}.pt")
    assert os.path.exists(ckpt_path), f"no checkpoint at {ckpt_path} — run train.py first"

    # Rebuild the tokenizer from the meta file prepare.py wrote.
    meta = json.load(open(os.path.join(DATA_DIR, "meta.json")))
    chars = meta["chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(chars[i] for i in ids)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # weights_only=False: this is a checkpoint we wrote ourselves, containing
    # a plain dict of tensors + ints (torch>=2.6 defaults to the stricter
    # loader, which would refuse it).
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_cfg = GPTConfig(**ckpt["config"])
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()  # inference mode: disables dropout, uses inference-time behavior
    print(
        f"loaded {os.path.basename(ckpt_path)} | "
        f"{model.num_params(non_embedding=False)/1e6:.2f}M params | "
        f"iter {ckpt['iter_num']} | best val loss {ckpt['best_val_loss']:.4f}"
    )

    idx = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("-" * 60)
    print(decode(out[0].tolist()))
    print("-" * 60)


if __name__ == "__main__":
    main()
