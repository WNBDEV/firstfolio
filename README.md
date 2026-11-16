# firstfolio — a GPT trained from scratch, on a GTX 960

A minimal GPT implementation (~500 lines, 4 files) that you can read line by
line. Nothing is hidden behind a library — the "AI" is `model.py`, and
everything from GPT-2 to Claude is the same architecture with bigger numbers.

Hardware reality this was sized for: 2 GB VRAM, Maxwell GPU (`sm_52`), fp32
training. The `full` config (~10.7M params) is the largest thing that fits
comfortably while the desktop is running.

## Reading order

1. **`prepare.py`** — smallest file. What a tokenizer is, how text becomes
   integers, why the data lives in flat `.bin` files.
2. **`model.py`** — the heart. Attention, MLP, residual blocks, the full stack.
   Start at `CausalSelfAttention.forward` and find the line
   `att = softmax(Q K^T / sqrt(d)) @ V`.
3. **`train.py`** — the objective (next-token cross-entropy) plus all the
   real-world craft: warmup + cosine LR schedule, weight-decay param groups,
   gradient accumulation, grad clipping, checkpointing.
4. **`sample.py`** — how text actually comes out of the model: one token at a
   time, with temperature and top-k.
5. **`numpy_infer.py`** — the same model re-implemented in pure numpy, no torch.
   Every function is one plain-English sentence: "look back and blend",
   "digest", "put a probability on each of 65 characters". The proof that
   inference is just matrix math.

## Running it

```bash
cd ~/Projects/firstfolio

# no activation needed — call the venv's python directly (works in fish, bash, zsh):
#   .venv/bin/python <script>
# (bash users can also: source .venv/bin/activate; fish users: source .venv/bin/activate.fish)

.venv/bin/python prepare.py                # download + tokenize (once)
.venv/bin/python train.py --preset debug   # ~74k params, ~1 min — sanity check
.venv/bin/python sample.py --preset debug  # char soup (expected!) — pipeline works
.venv/bin/python train.py --preset full    # ~10.7M params, ~2 h — the real run
.venv/bin/python sample.py --preset full --prompt "ROMEO:" --max-new-tokens 800

# the model with NO torch at all (pure numpy — see numpy_infer.py):
.venv/bin/python export_weights.py         # one-time: copy weights out of the checkpoint
.venv/bin/python numpy_infer.py --prompt "ROMEO:" --show-probs
```

Interrupted training? `python train.py --preset full --resume` continues from
`ckpt_full_last.pt`. Every checkpoint carries its own architecture config
inside, so `sample.py` can rebuild the model from the file alone.

## The ladder

| preset | params | train time (GTX 960) | expected val loss | output quality |
|---|---|---|---|---|
| debug  | ~74k   | ~1 min   | ~1.8 | Shakespeare-*shaped* words, no syntax |
| small  | ~0.44M | ~10 min  | ~1.6 | real words, occasional phrases |
| full   | ~10.7M | ~2 h     | ~1.5 | recognizable dialogue, names, structure |

Debug-small first, scale after: a 1-minute feedback loop finds bugs that would
otherwise appear 90 minutes into the big run.

## Things worth interrogating

**model.py**
- Why must the attention mask be *causal*? What would the model learn if it
  weren't?
- Why is softmax taken over the key dimension, and why scale by
  `1/sqrt(head_dim)`?
- Why are attention and MLP each wrapped in `x = x + f(x)`? Delete a residual
  and watch training die.
- What does weight tying (`wte.weight = lm_head.weight`) save?
- Why does the mask `register_buffer(..., persistent=False)` instead of being
  a plain tensor or a parameter?

**train.py**
- `loss / grad_accum` before `.backward()`: why divide, and why clip *after*
  accumulation?
- Why does weight decay apply to matrices but not LayerNorm gains?
- Why does the LR warm up from zero and then decay along a cosine?
- The iter-0 eval prints loss ≈ 4.17 = `-ln(1/65)`. Why is that the "random
  model" baseline for a 65-token vocabulary?

**prepare.py**
- Why does a chronological 90/10 split (no shuffling) matter for the val loss?
- Why uint16? What happens when you eventually switch to BPE with a 50k vocab?

## Phase 5 — experiments (the actual education)

Each of these is a one-line change + one training run:

1. **Depth ladder**: `--n-layer 1` vs `2` vs `6` on the full data — watch val
   loss and output quality.
2. **Context ladder**: `--block-size 64` vs `256`.
3. **Sampling tour**: temperature 0.2 / 0.8 / 1.5, top-k 1 (greedy) vs 65 (off).
4. **BPE tokenizer**: replace the char tokenizer with byte-pair encoding
   (Karpathy's `minbpe` is ~200 lines) — your step toward "real" LLM territory.
5. **Better data**: TinyStories (simple children's stories). This is the
   famous result of the [TinyStories paper](https://arxiv.org/abs/2305.06361):
   1–10M param models produce *coherent English* when the data is simple
   enough. Same code, new `.bin` files.
6. **Escape velocity**: when the model you want no longer fits in 2 GB, the
   same code runs unchanged on a free Colab T4 — only the numbers grow.

## Environment notes (why these exact versions)

- **Python 3.12**: the last PyTorch builds with Maxwell (`sm_52`) GPU kernels
  are torch ≤ 2.8, which stop at Python 3.13.
- **torch 2.7.1+cu126**: CUDA 13 and torch 2.9 dropped Maxwell/Pascal. If you
  ever see `CUDA error: no kernel image is available for execution on the
  device`, that's a too-new build.
- **fp32 only**: Maxwell has no bf16 and its fp16 compute is crippled
  (1/64 rate). Don't enable AMP — it's slower here, not faster.
- Driver 580.x is the final legacy branch for this GPU; you're already on it.

## References

- Andrej Karpathy, ["Let's build GPT: from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY) —
  this project's direct ancestor; watch it after reading `model.py`.
- [nanoGPT](https://github.com/karpathy/nanoGPT) — the codebase this is modeled on.
- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — the original transformer paper.
- [TinyStories](https://arxiv.org/abs/2305.06361) — proof that tiny models + well-chosen data = coherent text.
