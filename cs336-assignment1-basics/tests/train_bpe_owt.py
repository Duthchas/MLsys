"""Train BPE tokenizer on OpenWebText dataset and serialize vocab/merges."""

import json
import os
import time

import psutil

from cs336_basics.bpe_train import train_bpe

INPUT_PATH = "data/owt_train.txt"
VOCAB_SIZE = 32000
SPECIAL_TOKENS = ["</think>"]
OUTPUT_DIR = "data/owt_bpe"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024

    start_time = time.time()
    vocab, merges = train_bpe(INPUT_PATH, VOCAB_SIZE, SPECIAL_TOKENS)
    elapsed = time.time() - start_time

    mem_after = process.memory_info().rss / 1024 / 1024

    # Serialize vocab
    vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump({str(k): list(v) for k, v in vocab.items()}, f)

    # Serialize merges
    merges_path = os.path.join(OUTPUT_DIR, "merges.txt")
    with open(merges_path, "w") as f:
        for t1, t2 in merges:
            f.write(f"{t1.decode('utf-8', errors='replace')} {t2.decode('utf-8', errors='replace')}\n")

    # Longest token
    longest = max(vocab.values(), key=len)
    longest_id = [k for k, v in vocab.items() if v == longest][0]

    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"RSS: {mem_before:.0f} -> {mem_after:.0f} MB")
    print(f"Vocab size: {len(vocab)}")
    print(f"Num merges: {len(merges)}")
    print(f"Longest token: {longest!r} (id={longest_id}, len={len(longest)})")


if __name__ == "__main__":
    main()
