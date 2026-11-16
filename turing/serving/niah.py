"""
Needle-in-a-haystack (pass-key retrieval, Mohtashami & Jaggi 2023) through the serving engine: a random pass key
is hidden at a chosen depth in filler text of a chosen token length, the model is asked for it, and a row passes when
the greedy completion contains the key.
"""

import random
import time
from typing import Any, Dict, List

from .accuracy_eval import engine_generate

INTRO = "There is an important pass key hidden inside a lot of irrelevant text. Find it and memorize it.\n\n"
FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
NEEDLE = "The pass key is {key}. Remember it. {key} is the pass key. "
QUESTION = "\n\nWhat is the pass key? The pass key is"


class LongContextNIAHEvaluator:
    """Pass-key retrieval over context lengths (prompt tokens) and depths (0 = start of the haystack, 1 = end)."""

    def __init__(self, engine, seed: int = 0):
        if engine.tokenizer is None:
            raise ValueError("needle-in-a-haystack needs the model's tokenizer; the engine has none")
        self.engine = engine
        self.seed = seed

    def build_prompt(self, context_len: int, depth: float, key: str) -> str:
        """Filler sentences around the needle, sized so the whole prompt is about context_len tokens."""
        tok = self.engine.tokenizer
        fixed = len(tok.encode(INTRO + NEEDLE.format(key=key) + QUESTION))
        n = max(0, (context_len - fixed) // len(tok.encode(FILLER)))
        at = round(depth * n)
        return INTRO + FILLER * at + NEEDLE.format(key=key) + FILLER * (n - at) + QUESTION

    def evaluate_retrieval(self, context_lengths: List[int], depth_fractions: List[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
                           batch_size: int = 1) -> Dict[str, Any]:
        rng = random.Random(self.seed)
        cases = [(c, d, str(rng.randint(10000, 99999))) for c in context_lengths for d in depth_fractions]
        prompts = [self.engine.tokenizer.encode(self.build_prompt(c, d, k)) for c, d, k in cases]
        start = time.perf_counter()
        outputs = engine_generate(self.engine, prompts, max_new_tokens=8, batch_size=batch_size)
        elapsed = time.perf_counter() - start
        rows = []
        for (c, d, key), prompt, ids in zip(cases, prompts, outputs):
            text = self.engine.tokenizer.decode(ids, skip_special_tokens=True)
            rows.append({"context_length": c, "prompt_tokens": len(prompt), "depth": d, "key": key,
                         "completion": text, "found": key in text})
        return {
            "n": len(rows),
            "accuracy": sum(r["found"] for r in rows) / len(rows) if rows else 0.0,
            "wall_seconds": round(elapsed, 2),
            "results": rows,
        }
