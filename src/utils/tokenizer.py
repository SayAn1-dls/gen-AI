"""Lightweight BPE tokenizer wrapper for text preprocessing."""

import re
from collections import Counter
from typing import List, Dict, Tuple


class SimpleTokenizer:
    """A minimal byte-pair encoding tokenizer for experimentation."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}

    def _get_pairs(self, tokens: List[str]) -> Counter:
        """Count adjacent token pairs."""
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def train(self, text: str) -> None:
        """Train the tokenizer on a corpus using BPE."""
        tokens = list(text)
        base_vocab = sorted(set(tokens))
        self._token_to_id = {ch: idx for idx, ch in enumerate(base_vocab)}
        next_id = len(base_vocab)

        while next_id < self.vocab_size:
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
            best_pair = pairs.most_common(1)[0][0]
            merged = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self._token_to_id[merged] = next_id
            next_id += 1

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        self._id_to_token = {v: k for k, v in self._token_to_id.items()}

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        tokens = list(text)
        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return [self._token_to_id.get(t, 0) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        return ''.join(self._id_to_token.get(i, '?') for i in ids)


if __name__ == '__main__':
    sample = 'the quick brown fox jumps over the lazy dog ' * 50
    tok = SimpleTokenizer(vocab_size=300)
    tok.train(sample)
    encoded = tok.encode('the quick fox')
    decoded = tok.decode(encoded)
    print(f'Encoded: {encoded}')
    print(f'Decoded: {decoded}')
