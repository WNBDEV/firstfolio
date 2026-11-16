"""
Phase 2 — the transformer itself.

This file IS the "AI". Every modern LLM (GPT-4, Claude, Llama, ...) is a
scaled-up version of exactly what is written here — same math, more parameters.

Reading order:

    CausalSelfAttention — the core idea: each token looks back at earlier tokens
                          and collects the information it needs
    MLP                 — after gathering, each token independently "thinks"
    Block               — attention (communicate) + MLP (compute), each wrapped in
                          a residual connection
    GPT                 — the full stack: embeddings -> N blocks -> output head

If you remember one line from this whole project, make it this one (step 3 in
CausalSelfAttention.forward):

    att = softmax(Q K^T / sqrt(d)) @ V

The model's ONLY job is next-token prediction; attention is how it mixes
information across positions to make that prediction good.

Smoke test:
    python -c "from model import GPT, GPTConfig; m = GPT(GPTConfig()); \
               print(f'{m.num_params(non_embedding=False)/1e6:.2f}M params')"
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Hyperparameters. Moving up/down the model "ladder" (see README) is
    nothing but changing these numbers — the architecture never changes."""

    block_size: int = 256   # context length: how many tokens the model can see at once
    vocab_size: int = 65    # number of possible tokens (65 unique chars in Shakespeare)
    n_layer: int = 6        # transformer blocks stacked on top of each other
    n_head: int = 6         # parallel attention "heads" inside each block
    n_embd: int = 384       # width: the size of every token's vector representation
    dropout: float = 0.0    # regularization. 0.0 on purpose: with a tiny model and a
                            # tiny corpus, dropout usually slows learning more than it
                            # helps (nanoGPT does the same for its Shakespeare config)


class CausalSelfAttention(nn.Module):
    """Multi-head *causal* self-attention.

    "Self"   — tokens attend to each other; there is no external memory.
    "Causal" — token i may only look at tokens <= i. Training asks the model to
               predict the next token from the ones it has; if it could peek at
               the future, the task would be trivially cheated and the model
               would learn nothing.
    "Multi-head" — the n_embd channels are split into n_head independent
               attention computations, so different heads can track different
               relationships (who is speaking, punctuation habits, word
               boundaries, ...). Nobody assigns these roles; they emerge.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # One linear layer produces queries, keys and values in a single shot
        # (3 * n_embd output channels, split below). One big matmul is faster
        # than three small ones.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Projection after the heads are merged, back into the residual stream.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # The causal mask: lower triangle of True. mask[i, j] == True means
        # "position i is ALLOWED to look at position j". Registered as a buffer:
        # a tensor that belongs to the module (travels with .to(device)) but is
        # neither a learned parameter nor saved in the state_dict — it is fully
        # determined by block_size, so we rebuild it on load (persistent=False).
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer(
            "mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # 1) Every token produces its query ("what am I looking for?"),
        #    key ("what do I contain?") and value ("what will I hand over if
        #    attended to?") vectors.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 2) Split the C channels into n_head heads: (B, T, C) -> (B, nh, T, C/nh).
        #    Heads operate independently — they never talk to each other.
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 3) THE line. q @ k^T gives every query-token a raw affinity score with
        #    every key-token. Dividing by sqrt(head_dim) keeps the dot products
        #    in a sane range — with large dims the raw scores would saturate the
        #    softmax into near one-hot vectors and gradients would vanish.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # 4) Enforce causality: future positions get -inf, which softmax turns
        #    into exactly 0.0 attention weight. Slicing to [:T, :T] lets this
        #    module work for any T <= block_size (needed during generation).
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 5) Each token's new representation = the attention-weighted average of
        #    all value vectors. "Attention" IS this weighted average.
        y = att @ v                                   # (B, nh, T, head_dim)

        # 6) Merge the heads back together: (B, nh, T, hs) -> (B, T, C).
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 7) Output projection + residual dropout.
        #    (Production note: steps 3-5 are what torch's fused
        #    F.scaled_dot_product_attention computes in one memory-efficient
        #    kernel. We write it out explicitly because this file's job is to
        #    be read.)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Position-wise feed-forward network.

    After attention has *gathered* information from other tokens, this gives
    every token a private, wider space to *compute* with what it gathered.
    The 4x expansion (n_embd -> 4*n_embd -> n_embd) is the classic GPT ratio —
    the MLP holds roughly 2/3 of a model's parameters.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)   # the nonlinearity. Without it, 6 stacked linear layers
                        # would collapse into a single (weak) linear map.
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """One transformer block = communicate (attention) + compute (MLP).

    Both are wrapped in a residual connection: x = x + f(x). Residuals are what
    make deep stacks trainable at all — gradients flow through the "+ x"
    shortcut untouched, so each block only has to learn a small *refinement* of
    the stream rather than re-represent it from scratch.

    Pre-LN (normalize BEFORE each sublayer, as here) is the modern, stable
    choice; the original "Attention Is All You Need" paper used post-LN and
    needed learning-rate warmup tricks to compensate.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """The full model: embeddings -> n_layer blocks -> final norm -> output head.

    Data flow for an input of T token ids:
        token ids -> vectors (wte + wpe) -> [Block x n_layer] -> logits (B,T,vocab)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # wte: token embedding — a learned lookup table (vocab_size x n_embd).
            #      This is the ONLY place tokens exist as symbols; from here on,
            #      everything is geometry.
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            # wpe: position embedding. Attention itself is permutation-invariant
            #      (it treats its input as an unordered set), so order must be
            #      injected explicitly or "dog bites man" == "man bites dog".
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        # lm_head: maps each position's final vector to a score ("logit") for
        # every token in the vocabulary. A softmax over these scores is the
        # model's next-token probability distribution.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the output head reads the SAME matrix that embeds tokens
        # (used transposed). Saves vocab_size*n_embd parameters and helps tiny
        # models generalize a bit. A surprising fact: this "input dictionary ==
        # output dictionary" trick is used by many real models.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # GPT-2 detail: residual-path projections start scaled down by
        # 1/sqrt(2*n_layer), so the residual stream starts near-identity and a
        # deep stack doesn't blow up on the very first step.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        """GPT-2-style init: all weights normal(0, 0.02), biases zero."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=True):
        """Parameter count. Weight tying means the embedding/head matrix is
        counted once — parameters() deduplicates shared tensors."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) int64 token ids — the model's input
        targets: (B, T) int64 — the same ids shifted LEFT by one, i.e.
                 targets[:, t] is the correct next token after idx[:, t].
                 That one-token shift is the entire "task" of an LLM.

        Returns logits (B, T, vocab_size) and, if targets given, the mean
        cross-entropy loss over all B*T predictions.
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )

        tok_emb = self.transformer.wte(idx)            # (B, T, C)
        pos = torch.arange(T, device=idx.device)       # (T,)
        pos_emb = self.transformer.wpe(pos)            # (T, C)
        x = self.transformer.drop(tok_emb + pos_emb)   # (B, T, C)

        for block in self.transformer.h:
            x = block(x)                               # (B, T, C)

        x = self.transformer.ln_f(x)                   # (B, T, C)
        logits = self.lm_head(x)                       # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten (B, T) into one long stream of predictions: for every
            # position, compare the vocab-sized score vector against the id of
            # the true next token. Cross-entropy == -log p(correct token).
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive sampling: predict ONE token, append it, repeat — each
        time re-reading the whole (cropped) context. The model never writes a
        paragraph in one shot; it writes one character at a time.

        temperature < 1.0 sharpens the distribution (safer, more repetitive);
        temperature > 1.0 flattens it (weirder, riskier).
        top_k restricts sampling to the k most likely tokens (cuts rare nonsense).
        """
        for _ in range(max_new_tokens):
            # Crop context to what the model can see (its positional embeddings
            # only exist for block_size positions).
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size:]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # only the LAST position predicts the next token

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")  # non-top-k become impossible

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # SAMPLE, don't argmax —
                                                               # deterministic output is dull
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
