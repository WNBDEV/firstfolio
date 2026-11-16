"""
Phase 3 — the training loop.

The objective could not be simpler: for every position in a block of text,
predict the next character. The loss is cross-entropy = -log(probability the
model put on the correct next token). Everything else in this file is
bookkeeping that makes gradient descent stable and fits in a 2 GB GPU.

Presets (the "ladder", see README) — same code, different numbers only:
    --preset debug   ~74k params    ~1 min    proves the pipeline works
    --preset small   ~0.44M params  ~10 min   middle of the ladder
    --preset full    ~10.7M params  ~40 min   the real run (v2: dropout 0.1, 1500 steps)

Useful overrides:
    python train.py --preset full --micro-batch 8 --grad-accum 8   (if CUDA OOM)
    python train.py --preset full --resume                         (continue from ckpt_<preset>_last.pt)

What to expect: the first eval (iter 0, before any training) should show a loss
near -ln(1/65) = 4.17 — the loss of a model guessing uniformly at random. If
you see ~4.17 there, the whole pipeline is wired correctly. Then it's downhill:
roughly 4.17 -> ~1.9 (iter 1k) -> ~1.5 (final) for the full preset.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from model import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "shakespeare")

# The ladder. micro_batch = sequences per forward pass (VRAM-limited);
# grad_accum = how many micro-batches are averaged into ONE optimizer step,
# so effective_batch = micro_batch * grad_accum.
PRESETS = {
    "debug": dict(n_layer=1, n_head=4, n_embd=64, block_size=256,
                  micro_batch=32, grad_accum=1, max_iters=500, warmup_iters=20,
                  eval_interval=100, eval_iters=20, lr=1e-3, dropout=0.0),
    "small": dict(n_layer=2, n_head=4, n_embd=128, block_size=256,
                  micro_batch=32, grad_accum=2, max_iters=2000, warmup_iters=50,
                  eval_interval=200, eval_iters=20, lr=1e-3, dropout=0.0),
    # the "v2 recipe": the v1 run (train_full.log) overfit by iter ~1000, so we
    # (a) regularize with dropout and (b) end the schedule near where v1's val
    # loss peaked, instead of decaying the LR pointlessly for 4000 iters.
    "full": dict(n_layer=6, n_head=6, n_embd=384, block_size=256,
                 micro_batch=12, grad_accum=5, max_iters=1500, warmup_iters=100,
                 eval_interval=100, eval_iters=20, lr=1e-3, dropout=0.1),
}


def get_lr(it, *, base_lr, warmup_iters, max_iters, min_lr_ratio=0.1):
    """Learning-rate schedule: linear warmup, then cosine decay.

    Warmup: a fresh model takes wild, poorly-scaled steps — ramping from 0 to
    the peak LR over the first iterations protects it.
    Cosine decay: late training benefits from small steps (polishing, not
    exploring); the model's final convergence sharpness is largely set by how
    low the LR gets at the end.
    """
    min_lr = base_lr * min_lr_ratio
    if it < warmup_iters:
        return base_lr * (it + 1) / (warmup_iters + 1)
    if it >= max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * decay_ratio))


def configure_optimizers(model, lr, weight_decay=0.1, betas=(0.9, 0.95)):
    """AdamW with the GPT-2 / nanoGPT param-group trick: weight decay only on
    weight MATRICES (dim >= 2), never on LayerNorm gains (dim == 1) — decaying
    a normalization gain just fights the normalization it performs.
    (There are no biases in model.py — bias=False everywhere — so 1D params
    here are exactly the LayerNorm gains.)

    betas=(0.9, 0.95): the lower second beta forgets stale gradient history
    faster than torch's default (0.999); LLM training has converged on ~0.95.
    """
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def get_batch(split, block_size, batch_size, device):
    """Sample `batch_size` random windows of `block_size` tokens from a .bin file.

    x = the window; y = the same window shifted LEFT by one position:
    y[:, t] is the character that comes after x[:, t]. That 1-token shift is
    the entire supervision signal.

    np.memmap means the file is indexed in place — we never load the whole
    corpus into RAM, we just read 12 little windows per step.
    """
    data = np.memmap(os.path.join(DATA_DIR, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        # Pin host memory + async copy: lets the next CPU->GPU transfer overlap
        # with GPU compute. Marginal at our sizes, but it's the real-world pattern.
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, cfg, device):
    """Mean loss over a few fresh random batches, per split. Random windows mean
    every eval sees new data — cheap and unbiased. val >> train signals
    overfitting; here they should track each other closely."""
    was_training = model.training
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg["eval_iters"])
        for k in range(cfg["eval_iters"]):
            X, Y = get_batch(split, cfg["block_size"], cfg["micro_batch"], device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    if was_training:
        model.train()
    return out


def save_checkpoint(path, model, optimizer, config, iter_num, best_val_loss):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),  # saved so training can RESUME exactly
        "config": asdict(config),             # enough to rebuild the architecture
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
    }, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=list(PRESETS), default="full")
    # any of the preset's numbers can be overridden on the command line
    p.add_argument("--n-layer", type=int, dest="n_layer")
    p.add_argument("--n-head", type=int, dest="n_head")
    p.add_argument("--n-embd", type=int, dest="n_embd")
    p.add_argument("--block-size", type=int, dest="block_size")
    p.add_argument("--micro-batch", type=int, dest="micro_batch")
    p.add_argument("--grad-accum", type=int, dest="grad_accum")
    p.add_argument("--max-iters", type=int, dest="max_iters")
    p.add_argument("--eval-interval", type=int, dest="eval_interval")
    p.add_argument("--eval-iters", type=int, dest="eval_iters")
    p.add_argument("--lr", type=float)
    p.add_argument("--dropout", type=float)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    cfg = dict(PRESETS[args.preset])
    cfg.update({k: v for k, v in vars(args).items() if k in cfg and v is not None})

    for f in ("train.bin", "val.bin", "meta.json"):
        assert os.path.exists(os.path.join(DATA_DIR, f)), f"{f} missing — run prepare.py first"

    torch.manual_seed(1337)  # same init, same data order on every run: reproducible experiments

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"device: {props.name} (sm_{props.major}{props.minor}) | torch {torch.__version__} | fp32 training (Maxwell: no bf16, fp16 compute unusable)", flush=True)

    meta = json.load(open(os.path.join(DATA_DIR, "meta.json")))
    model_cfg = GPTConfig(
        block_size=cfg["block_size"],
        vocab_size=meta["vocab_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg.get("dropout", 0.0),
    )
    model = GPT(model_cfg).to(device)
    print(
        f"model: {model.num_params()/1e6:.2f}M non-embedding params "
        f"({model.num_params(non_embedding=False)/1e6:.2f}M incl. embeddings) | "
        f"effective batch = {cfg['micro_batch']} x {cfg['grad_accum']} = {cfg['micro_batch']*cfg['grad_accum']} seqs "
        f"of {cfg['block_size']} tokens", flush=True,
    )

    if device == "cuda":
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"VRAM: {free_b/2**30:.2f} GiB free / {total_b/2**30:.2f} GiB total", flush=True)

    optimizer = configure_optimizers(model, lr=cfg["lr"])

    iter_num = 0
    best_val_loss = float("inf")
    last_path = os.path.join(HERE, f"ckpt_{args.preset}_last.pt")
    best_path = os.path.join(HERE, f"ckpt_{args.preset}.pt")

    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        iter_num = ckpt["iter_num"]
        best_val_loss = ckpt["best_val_loss"]
        print(f"resumed from {os.path.basename(last_path)} at iter {iter_num}", flush=True)

    # ------------------------------------------------------------- training loop
    model.train()
    t_last = time.time()
    last_loss = float("nan")
    while iter_num < cfg["max_iters"]:

        lr = get_lr(
            iter_num,
            base_lr=cfg["lr"],
            warmup_iters=cfg["warmup_iters"],
            max_iters=cfg["max_iters"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # periodic eval + checkpointing
        if iter_num % cfg["eval_interval"] == 0:
            losses = estimate_loss(model, cfg, device)
            print(
                f"[eval ] iter {iter_num:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}",
                flush=True,
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(best_path, model, optimizer, model_cfg, iter_num, best_val_loss)
                print(f"[best ] val improved -> saved {os.path.basename(best_path)}", flush=True)
            save_checkpoint(last_path, model, optimizer, model_cfg, iter_num, best_val_loss)

        # heartbeat (loss of the just-finished step, averaged over its micro-batches)
        if iter_num > 0 and iter_num % 100 == 0:
            dt_ms = 1000 * (time.time() - t_last) / 100
            print(
                f"[train] iter {iter_num:5d} | loss {last_loss:.3f} | lr {lr:.2e} | {dt_ms:.0f} ms/iter",
                flush=True,
            )
            t_last = time.time()

        # ---- one optimizer step, accumulated over micro-batches ----
        # 2 GB of VRAM can't hold activations for a batch of 64 sequences at
        # once, but the gradient of 12 sequences computed 5 times, summed, is
        # (nearly exactly) the gradient of 60 sequences at once. That's all
        # gradient accumulation is.
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for _ in range(cfg["grad_accum"]):
            X, Y = get_batch("train", cfg["block_size"], cfg["micro_batch"], device)
            _, loss = model(X, Y)
            running_loss += loss.item()
            (loss / cfg["grad_accum"]).backward()  # scale so micro-gradients sum to the full-batch gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # tame occasional huge gradients early in training
        optimizer.step()

        last_loss = running_loss / cfg["grad_accum"]
        iter_num += 1

    save_checkpoint(last_path, model, optimizer, model_cfg, iter_num, best_val_loss)
    print(f"training done at iter {iter_num} | best val loss {best_val_loss:.4f}", flush=True)
    print(f"best checkpoint: {best_path}", flush=True)


if __name__ == "__main__":
    main()
