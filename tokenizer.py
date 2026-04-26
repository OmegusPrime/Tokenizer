import re
import json
import logging
import collections
from ingest import Ingest                          # H-TOK-1: relative-safe direct import

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("tokenizer")

# ── Special tokens  (C-TOK-4) ─────────────────────────────────────────────────
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>",
                  "<user>", "<assistant>", "<sep>", "<mask>"]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

# ── GPT-2 byte→unicode shim  (H-TOK-3) ───────────────────────────────────────
def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs, n = list(bs), 2**8
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

BYTE_ENCODER = _bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}


class Tokenizer:
    # M-TOK-1: removed unreachable `?\d+` branch
    PRE_TOK = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+""",
        re.UNICODE,
    )

    def __init__(self):
        self.vocab: dict[str, int]    = {}
        self.inv_vocab: dict[int, str] = {}
        self.merges: dict[tuple, str]  = {}
        self.bpe_ranks: dict[tuple, int] = {}   # C-TOK-3: O(1) rank lookup
        self._frozen = False                     # M-TOK-4: freeze after training

    # ── Internals ─────────────────────────────────────────────────────────

    def _add_token(self, token: str) -> int:
        """Add a token to vocab if absent; return its id."""
        if token not in self.vocab:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.inv_vocab[idx] = token
        return self.vocab[token]

    @staticmethod
    def _word_to_bytes(word: str) -> list[str]:
        """Encode a word as its UTF-8 byte-chars using the GPT-2 shim."""
        return [BYTE_ENCODER[b] for b in word.encode("utf-8")]

    @staticmethod
    def _pretokenize(text: str) -> list[list[str]]:
        tokens = []
        for m in Tokenizer.PRE_TOK.finditer(text):
            word = m.group()
            # Ġ prefix marks a leading space (GPT-2 convention)
            raw = ("Ġ" + word.lstrip(" ")) if word.startswith(" ") else word
            chars = Tokenizer._word_to_bytes(raw)   # H-TOK-3: byte-level
            if chars:
                tokens.append(chars)
        return tokens

    @staticmethod
    def _get_pairs(word: list[str]) -> list[tuple]:
        return [(word[i], word[i + 1]) for i in range(len(word) - 1)]

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, texts, vocab_size: int = 10_000, min_freq: int = 2):  # H-TOK-2: default 2
        """
        Train BPE.
        C-TOK-2: uses an inverted pair-index so each merge step is O(occurrences)
                 instead of O(vocab × corpus).
        H-TOK-4: accepts a generator — never materialises the full corpus.
        """
        if self._frozen:
            raise RuntimeError("Tokenizer is frozen; create a new instance to retrain.")

        # ── 1. Reserve special tokens first  (C-TOK-4) ───────────────────
        for tok in SPECIAL_TOKENS:
            self._add_token(tok)

        # ── 2. Build byte base vocab  (H-TOK-3) ──────────────────────────
        for ch in BYTE_ENCODER.values():
            self._add_token(ch)
        self._add_token("Ġ")

        # ── 3. Pre-tokenize corpus (streaming)  (H-TOK-4) ─────────────────
        log.info("Pre-tokenising corpus …")
        corpus: list[tuple] = []        # list of tuples (immutable → hashable)
        for text in texts:
            for chars in self._pretokenize(text):
                corpus.append(tuple(chars))

        log.info(f"Corpus: {len(corpus):,} words")

        # ── 4. Build inverted pair index  (C-TOK-2) ───────────────────────
        # pair_counts[pair]               = total frequency
        # pair_index[pair][word_idx]      = count of pair in that word
        pair_counts: collections.Counter = collections.Counter()
        pair_index: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)

        word_counts: collections.Counter = collections.Counter(corpus)
        # corpus as list-of-unique-words for index
        words: list[tuple] = list(word_counts.keys())
        # working copy — mutable
        work = [list(w) for w in words]

        for idx, word in enumerate(work):
            freq = word_counts[words[idx]]
            for pair in self._get_pairs(word):
                pair_counts[pair]        += freq
                pair_index[pair][idx]    += freq

        num_merges = vocab_size - len(self.vocab)
        log.info(f"Base vocab: {len(self.vocab)}  |  merges to learn: {num_merges}")

        for step in range(num_merges):
            if not pair_counts:
                break
            best_pair, best_freq = pair_counts.most_common(1)[0]
            if best_freq < min_freq:
                log.info(f"Early stop at step {step}: freq {best_freq} < min_freq {min_freq}")
                break

            new_token = best_pair[0] + best_pair[1]

            # M-TOK-3: guard against id collision
            assert new_token not in self.vocab, \
                f"Merge collision: '{new_token}' already in vocab"

            self.merges[best_pair] = new_token
            self.bpe_ranks[best_pair] = len(self.bpe_ranks)   # C-TOK-3
            self._add_token(new_token)

            # ── Incremental index update  (C-TOK-2) ──────────────────────
            affected = dict(pair_index[best_pair])
            del pair_index[best_pair]
            del pair_counts[best_pair]

            a, b = best_pair
            for idx, _ in affected.items():
                word = work[idx]
                freq = word_counts[words[idx]]
                i = 0
                new_word: list[str] = []
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                        # remove old pairs involving the two tokens being merged
                        if i > 0:
                            old = (word[i - 1], a)
                            pair_counts[old]     -= freq
                            pair_index[old][idx] -= freq
                            if pair_counts[old] <= 0:
                                del pair_counts[old]
                        if i < len(word) - 2:
                            old = (b, word[i + 2])
                            pair_counts[old]     -= freq
                            pair_index[old][idx] -= freq
                            if pair_counts[old] <= 0:
                                del pair_counts[old]
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                work[idx] = new_word

                # add new pairs from the merged positions
                for pair in self._get_pairs(new_word):
                    if new_token in pair:
                        pair_counts[pair]        += freq
                        pair_index[pair][idx]    += freq

            if (step + 1) % 200 == 0:
                log.info(f"step {step+1}/{num_merges}  '{a}{b}'→'{new_token}'  freq={best_freq}")

        # M-TOK-4: freeze vocab after training
        self._frozen = True
        log.info(f"Training done. Vocab size: {len(self.vocab)}")

    # ── Encode ────────────────────────────────────────────────────────────

    def _encode_word(self, chars: list[str]) -> list[int]:
        """Apply BPE merges to a single pre-token using O(1) rank lookup."""
        word = list(chars)
        while len(word) > 1:
            pairs = self._get_pairs(word)
            # C-TOK-3: O(1) per pair via bpe_ranks dict
            best = min(
                ((self.bpe_ranks[p], p) for p in pairs if p in self.bpe_ranks),
                default=None,
            )
            if best is None:
                break
            _, best_pair = best
            a, b = best_pair
            merged, i, new_word = a + b, 0, []
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return [self.vocab.get(tok, UNK_ID) for tok in word]  # C-TOK-4: real UNK_ID

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        """
        Encode text to a flat list of token ids.  (C-TOK-1: always flat)
        """
        ids: list[int] = []
        if add_special:
            ids.append(BOS_ID)
        for word_chars in self._pretokenize(text):
            ids.extend(self._encode_word(word_chars))
        if add_special:
            ids.append(EOS_ID)
        return ids

    # ── Decode ────────────────────────────────────────────────────────────

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        Decode a flat list of ids back to text.  (L-TOK-2: no .strip() — preserves whitespace)
        """
        special_ids = {self.vocab[t] for t in SPECIAL_TOKENS if t in self.vocab}
        tokens = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            tokens.append(self.inv_vocab.get(i, ""))

        # re-join Ġ-space markers and decode bytes back to unicode
        text = "".join(tokens).replace("Ġ", " ")
        # decode GPT-2 byte chars back to utf-8
        try:
            text = bytearray([BYTE_DECODER[c] for c in text if c in BYTE_DECODER]).decode("utf-8")
        except Exception:
            pass   # if partial/corrupt, return what we have
        return text

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, path: str):
        data = {
            "vocab":  self.vocab,
            "merges": [list(k) + [v] for k, v in self.merges.items()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)   # M-TOK-2: no indent bloat
        log.info(f"Saved tokenizer → {path}")

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab     = data["vocab"]
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.merges    = {(r[0], r[1]): r[2] for r in data["merges"]}
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}  # C-TOK-3
        self._frozen   = True
        log.info(f"Loaded tokenizer: vocab={len(self.vocab)}")

    # ── Convenience ───────────────────────────────────────────────────────

    @staticmethod
    def train_from_ingest(data_dir: str, vocab_size: int = 10_000,
                          save_path: str = "bpe.json") -> "Tokenizer":
        """Stream from Ingest — never loads full corpus into memory. (H-TOK-4)"""
        def _text_stream():
            for doc in Ingest(data_dir).ingest():
                if doc["text"].strip():
                    yield doc["text"]

        tok = Tokenizer()
        tok.train(_text_stream(), vocab_size=vocab_size)
        tok.save(save_path)
        return tok

    # ── API surface  (L-TOK-3) ────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return (f"Tokenizer(vocab_size={self.vocab_size}, "
                f"merges={len(self.merges)}, frozen={self._frozen})")

    @classmethod
    def from_pretrained(cls, path: str) -> "Tokenizer":
        tok = cls()
        tok.load(path)
        return tok


