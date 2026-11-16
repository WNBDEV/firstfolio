"""
One-time bridge: copy the trained weights out of the PyTorch checkpoint into
a plain .npz file that numpy can read. After this runs, PyTorch is out of the
picture — numpy_infer.py never touches it.

    .venv/bin/python export_weights.py
"""

import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

ckpt = torch.load(os.path.join(HERE, "ckpt_full.pt"), map_location="cpu", weights_only=False)
sd = ckpt["model"]
cfgd = ckpt["config"]

out = {
    "wte": sd["transformer.wte.weight"].numpy(),   # token stat sheets (same matrix the output head reads)
    "wpe": sd["transformer.wpe.weight"].numpy(),   # position stat sheets
    "ln_f_w": sd["transformer.ln_f.weight"].numpy(),
    "ln_f_b": sd["transformer.ln_f.bias"].numpy(),
    "chars": np.array(json.load(open(os.path.join(HERE, "data", "shakespeare", "meta.json")))["chars"]),
    "n_layer": np.array(cfgd["n_layer"]),
    "n_head": np.array(cfgd["n_head"]),
    "n_embd": np.array(cfgd["n_embd"]),
    "block_size": np.array(cfgd["block_size"]),
    "vocab_size": np.array(cfgd["vocab_size"]),
}
for i in range(cfgd["n_layer"]):
    p = f"transformer.h.{i}."
    out[f"h{i}_ln1_w"] = sd[p + "ln_1.weight"].numpy()
    out[f"h{i}_ln1_b"] = sd[p + "ln_1.bias"].numpy()
    out[f"h{i}_attn_w"] = sd[p + "attn.c_attn.weight"].numpy()
    out[f"h{i}_attn_proj_w"] = sd[p + "attn.c_proj.weight"].numpy()
    out[f"h{i}_ln2_w"] = sd[p + "ln_2.weight"].numpy()
    out[f"h{i}_ln2_b"] = sd[p + "ln_2.bias"].numpy()
    out[f"h{i}_mlp_fc_w"] = sd[p + "mlp.c_fc.weight"].numpy()
    out[f"h{i}_mlp_proj_w"] = sd[p + "mlp.c_proj.weight"].numpy()

np.savez(os.path.join(HERE, "firstfolio_weights.npz"), **out)

cfg_keys = {"n_layer", "n_head", "n_embd", "block_size", "vocab_size"}
total = sum(v.size for k, v in out.items() if k not in cfg_keys and k != "chars")
print(f"wrote tiny_gpt_weights.npz | {total/1e6:.2f}M parameters copied | torch officially done")
