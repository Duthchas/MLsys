"""Train BPE tokenizer on TinyStories dataset and serialize vocab/merges."""

import json
import os
import time
import tracemalloc

from cs336_basics.bpe_train import train_bpe

INPUT_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["<|endoftext|>"]
OUTPUT_DIR = "data/tinystories_bpe"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Start tracking
    tracemalloc.start()
    start_time = time.time()

    # Train
    vocab, merges = train_bpe(INPUT_PATH, VOCAB_SIZE, SPECIAL_TOKENS)

    elapsed = time.time() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

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

    print(f"Time: {elapsed:.1f}s")
    print(f"Peak memory: {peak_mem / 1024 / 1024:.1f} MB")
    print(f"Vocab size: {len(vocab)}")
    print(f"Num merges: {len(merges)}")
    print(f"Longest token: {longest!r} (id={longest_id}, len={len(longest)})")


if __name__ == "__main__":
    main()
