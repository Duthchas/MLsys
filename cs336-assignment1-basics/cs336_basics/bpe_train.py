from __future__ import annotations

import os
from collections import defaultdict
from multiprocessing import Pool
from typing import BinaryIO

import regex as re

from cs336_basics.bpe_rust import train_bpe_rs


# GPT-2 pre-tokenization regex pattern
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping."""
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _pretokenize_chunk(args: tuple) -> dict[bytes, int]:
    """Pre-tokenize a single chunk of the corpus. Runs in a worker process."""
    input_path, chunk_start, chunk_end, special_tokens_bytes, split_pattern = args

    pre_token_counts: dict[bytes, int] = defaultdict(int)
    pretokenize_re = re.compile(GPT2_PAT)

    with open(input_path, "rb") as f:
        f.seek(chunk_start)
        chunk = f.read(chunk_end - chunk_start).decode("utf-8", errors="ignore")

    # Split by special tokens within this chunk
    if split_pattern:
        segments = re.split(f"({split_pattern})", chunk)
    else:
        segments = [chunk]

    special_token_set = set(
        st_bytes.decode("utf-8") for st_bytes in special_tokens_bytes
    )

    for segment in segments:
        if segment in special_token_set:
            continue
        for match in pretokenize_re.finditer(segment):
            token_str = match.group()
            encoded = token_str.encode("utf-8")
            if encoded:
                pre_token_counts[encoded] += 1

    return dict(pre_token_counts)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Pre-tokenization is parallelized across multiple processes using chunking
    at special token boundaries. The BPE merge loop runs in Rust.

    Args:
        input_path: Path to the training text file.
        vocab_size: Maximum vocabulary size (including initial bytes and special tokens).
        special_tokens: List of special tokens to add to the vocabulary.
        num_processes: Number of worker processes. Defaults to os.cpu_count().

    Returns:
        vocab: Mapping from token ID to token bytes.
        merges: Ordered list of merge pairs as (token1_bytes, token2_bytes).
    """
    if num_processes is None:
        num_processes = os.cpu_count() or 1

    # Build the split pattern for special tokens
    split_pattern = ""
    if special_tokens:
        escaped = [re.escape(st) for st in special_tokens]
        split_pattern = "|".join(escaped)

    # Use the first special token as the chunk boundary marker
    # (all special tokens act as hard boundaries, but find_chunk_boundaries needs one)
    split_special_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"

    # Find chunk boundaries
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token)

    special_tokens_bytes = [st.encode("utf-8") for st in special_tokens]

    # Build work items: (input_path, start, end, special_tokens_bytes, split_pattern)
    work = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if start >= end:
            continue
        work.append((str(input_path), start, end, special_tokens_bytes, split_pattern))

    # Parallel pre-tokenization and merge
    pre_token_counts: dict[bytes, int] = defaultdict(int)

    if len(work) > 1:
        with Pool(processes=min(num_processes, len(work))) as pool:
            for chunk_result in pool.imap_unordered(_pretokenize_chunk, work):
                for token, count in chunk_result.items():
                    pre_token_counts[token] += count
    else:
        for token, count in _pretokenize_chunk(work[0]).items():
            pre_token_counts[token] += count

    if not pre_token_counts:
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        next_id = 256
        for st in special_tokens:
            vocab[next_id] = st.encode("utf-8")
            next_id += 1
        return vocab, []

    # Convert to list format for Rust
    pre_token_list = [(token, freq) for token, freq in pre_token_counts.items()]
    special_token_bytes = [st.encode("utf-8") for st in special_tokens]

    # Call Rust implementation
    vocab, merges = train_bpe_rs(pre_token_list, vocab_size, special_token_bytes)

    return vocab, merges
