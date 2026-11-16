"""
Phase 1 — data preparation.

An LLM has no concept of "text". Its entire universe is a flat stream of
integers. This file:

  1. downloads the Tiny Shakespeare corpus (~1.1 MB, ~1M characters, 65 unique),
  2. builds a CHARACTER-LEVEL tokenizer (each unique character = one token id),
  3. splits the text 90/10 into a train set and a held-out validation set,
  4. encodes both to raw binary files of uint16 (2 bytes per token).

Why a character-level tokenizer for a first model?
  - zero dependencies and zero configuration (vs. BPE, which trains a merge table),
  - a 65-token vocabulary means the output softmax has 65 options — easy to
    reason about when something looks wrong.
The trade-off: each character is one prediction, so sequences are long and the
model must burn parameters on spelling. Real LLMs use BPE for exactly that
reason (see README, Phase 5).

Run:
    python prepare.py
"""

import json
import os
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "shakespeare")

# The canonical Tiny Shakespeare corpus used by Karpathy's char-rnn / nanoGPT.
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    input_path = os.path.join(DATA_DIR, "input.txt")

    # ------------------------------------------------------------------ 1. get data
    if not os.path.exists(input_path):
        print(f"downloading Tiny Shakespeare from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, input_path)

    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    print(f"corpus: {len(text):,} characters")

    # ------------------------------------------------------------------ 2. tokenizer
    # The vocabulary is just the sorted set of characters that appear in the text.
    # Sorting is arbitrary but makes the id assignment deterministic.
    chars = sorted(set(text))
    vocab_size = len(chars)
    print(f"vocab size: {vocab_size} unique characters: {''.join(chars)!r}")

    stoi = {ch: i for i, ch in enumerate(chars)}   # string -> id
    itos = list(chars)                             # id -> string
    encode = lambda s: [stoi[c] for c in s]        # text -> list[int]
    decode = lambda ids: "".join(itos[i] for i in ids)

    # ------------------------------------------------------------------ 3. split
    # 90/10: train on the first 90% of the text, hold out the last 10% to measure
    # generalization. (A chronological split — no shuffling — so the validation
    # data is genuinely "unseen future text".)
    n = len(text)
    train_text, val_text = text[: int(n * 0.9)], text[int(n * 0.9):]

    # ------------------------------------------------------------------ 4. encode to .bin
    # uint16 holds ids 0..65535; our vocab is 65, so 2 bytes/token is lossless.
    # tofile() writes raw values with no header — later we read these back with
    # np.memmap, which means the training loop can index them like an array
    # WITHOUT ever loading the whole file into RAM.
    train_ids = np.array(encode(train_text), dtype=np.uint16)
    val_ids = np.array(encode(val_text), dtype=np.uint16)
    train_ids.tofile(os.path.join(DATA_DIR, "train.bin"))
    val_ids.tofile(os.path.join(DATA_DIR, "val.bin"))

    # Persist the tokenizer itself, so train.py / sample.py rebuild the exact
    # same id <-> character mapping.
    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump({"vocab_size": vocab_size, "chars": itos}, f)

    print(f"wrote train.bin: {len(train_ids):,} tokens")
    print(f"wrote val.bin:   {len(val_ids):,} tokens")
    print("sanity check — decode(encode('Hello')):", decode(encode("Hello")))


if __name__ == "__main__":
    main()
