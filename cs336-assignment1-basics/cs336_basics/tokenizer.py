from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

import regex as re


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Reverse vocab: bytes -> id
        self.byte_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}

        # Build merge rank: (bytes, bytes) -> rank (lower = higher priority)
        self.merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }

        # Build regex for splitting by special tokens (longest match first)
        if self.special_tokens:
            escaped = [re.escape(st) for st in self.special_tokens]
            # Sort by length descending so longer tokens match first
            escaped.sort(key=len, reverse=True)
            self.special_token_pattern = re.compile("|".join(escaped))
        else:
            self.special_token_pattern = None

        # GPT-2 pre-tokenization pattern
        self.pretokenize_pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        with open(vocab_filepath) as f:
            vocab: dict[int, bytes] = {int(k): bytes(v) for k, v in json.load(f).items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned = line.rstrip()
                if cleaned and len(cleaned.split(" ")) == 2:
                    a, b = cleaned.split(" ")
                    merges.append((a.encode("utf-8"), b.encode("utf-8")))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        ids: list[int] = []

        # Split by special tokens
        if self.special_token_pattern:
            segments = self.special_token_pattern.split(text)
            special_matches = self.special_token_pattern.findall(text)
        else:
            segments = [text]
            special_matches = []

        # Interleave: segment, special, segment, special, ...
        for i, segment in enumerate(segments):
            if segment:
                ids.extend(self._encode_segment(segment))
            if i < len(special_matches):
                st_bytes = special_matches[i].encode("utf-8")
                ids.append(self.byte_to_id[st_bytes])

        return ids

    def _encode_segment(self, text: str) -> list[int]:
        """Encode a segment (no special tokens) using BPE."""
        ids: list[int] = []
        # Pre-tokenize with GPT-2 pattern
        for match in self.pretokenize_pattern.finditer(text):
            token_str = match.group()
            token_bytes = token_str.encode("utf-8")
            # Split into individual bytes, then apply BPE merges
            byte_list: list[bytes] = [bytes([b]) for b in token_bytes]
            ids.extend(self._apply_bpe(byte_list))
        return ids

    def _apply_bpe(self, byte_list: list[bytes]) -> list[int]:
        """Apply BPE merges to a list of single-byte tokens."""
        while len(byte_list) > 1:
            # Find the highest-priority (lowest rank) merge pair
            best_pair = None
            best_rank = len(self.merges)
            for i in range(len(byte_list) - 1):
                pair = (byte_list[i], byte_list[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            # Apply all occurrences of the best pair
            new_list: list[bytes] = []
            i = 0
            merged = best_pair[0] + best_pair[1]
            while i < len(byte_list):
                if (
                    i + 1 < len(byte_list)
                    and byte_list[i] == best_pair[0]
                    and byte_list[i + 1] == best_pair[1]
                ):
                    new_list.append(merged)
                    i += 2
                else:
                    new_list.append(byte_list[i])
                    i += 1
            byte_list = new_list

        return [self.byte_to_id[b] for b in byte_list]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode text from an iterable."""
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            # Process complete lines or reasonable chunks
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                yield from self.encode(line + "\n")
        # Process remaining buffer
        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        raw = b""
        for id in ids:
            raw += self.vocab[id]
        return raw.decode("utf-8", errors="replace")
