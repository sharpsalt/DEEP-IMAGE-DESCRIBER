import re
import tensorflow as tf
import sentencepiece as spm

class TextEncoder:
    """
    Minimal self-contained tokenizer (hash-word). Replace with SentencePiece later.
    """
    def __init__(self, spm_path=None, vocab_size=32000):
        self.start_id, self.end_id, self.pad_id, self.unk_id = 1, 2, 0, 3
        self.vocab_size = vocab_size
        self._sp = None
        if spm_path:
            try:
                self._sp = spm.SentencePieceProcessor(model_file=spm_path)
                self.start_id, self.end_id, self.pad_id = 1, 2, 0
            except Exception:
                self._sp = None

    def encode(self, text: str):
        if self._sp:
            ids = self._sp.encode(text, out_type=int)
        else:
            words = re.findall(r"\S+", text.strip())
            ids = [min(abs(hash(w)) % (self.vocab_size-4) + 4, self.vocab_size-1) for w in words]
        return [self.start_id] + ids + [self.end_id]

    def decode(self, ids):
        if self._sp:
            return self._sp.decode(ids)
        # stub detok
        return " ".join(f"w{i}" for i in range(len(ids)))

def count_words(text: str):
    return len(re.findall(r"\b\w+\b", text))
