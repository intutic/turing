"""
GSM8K exact-match evaluation through the serving engine: k-shot prompts built from the GSM8K train split, greedy
decoding, and the answer read from the completion ("#### N" if the model writes it, else its last number).
"""

import asyncio
import math
import re
import time
from typing import Any, Dict, List, Optional

GSM8K_REPO = "openai/gsm8k"


def engine_generate(engine, prompts: List[List[int]], max_new_tokens: int, batch_size: int = 8) -> List[List[int]]:
    """Greedy completions for token prompts through a ContinuousBatchEngine, batch_size requests in flight at a time."""
    async def run() -> List[List[int]]:
        await engine.start()
        try:
            out = []
            for i in range(0, len(prompts), batch_size):
                reqs = [await engine.add_request(p, max_new_tokens=max_new_tokens, temperature=0.0)
                        for p in prompts[i:i + batch_size]]
                for r in reqs:
                    out.append([t async for t in engine.stream_request(r)])
            return out
        finally:
            await engine.stop()
    return asyncio.run(run())


def load_gsm8k(split: str) -> List[Dict[str, str]]:
    """GSM8K rows {question, answer} from the HuggingFace hub (needs the `datasets` package)."""
    from datasets import load_dataset
    return [{"question": r["question"], "answer": r["answer"]} for r in load_dataset(GSM8K_REPO, "main", split=split)]


def _normalize(num: str) -> str:
    num = num.replace(",", "").replace("$", "").rstrip(".")
    try:
        value = float(num)
    except ValueError:
        return num
    return str(int(value)) if value == int(value) else str(value)


def gold_answer(answer: str) -> str:
    """The number after '####' in a GSM8K reference answer."""
    return _normalize(answer.split("####")[-1].strip())


def extract_answer(completion: str) -> Optional[str]:
    """The model's answer: the number after its first '####', else the last number it wrote; None if it wrote none."""
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", completion)
    if m:
        return _normalize(m.group(1))
    nums = re.findall(r"-?[\d,]*\.?\d+", completion)
    return _normalize(nums[-1]) if nums else None


def build_prompt(question: str, shots: List[Dict[str, str]]) -> str:
    """The lm-evaluation-harness GSM8K format: 'Question: ...\\nAnswer: ...' blocks separated by blank lines."""
    blocks = [f"Question: {s['question']}\nAnswer: {s['answer']}" for s in shots]
    return "\n\n".join(blocks + [f"Question: {question}\nAnswer:"])


def first_answer(completion: str) -> str:
    """The completion up to where the model starts another question (the engine has no stop strings)."""
    return re.split(r"\n\s*\n|Question:", completion, maxsplit=1)[0]


class GSM8KEvaluator:
    """Runs GSM8K problems through engine (which must hold a real tokenizer) and scores exact match."""

    def __init__(self, engine):
        if engine.tokenizer is None:
            raise ValueError("GSM8K needs the model's tokenizer; the engine has none")
        self.engine = engine

    def evaluate(self, problems: List[Dict[str, str]], shots: List[Dict[str, str]],
                 max_new_tokens: int = 256, batch_size: int = 8) -> Dict[str, Any]:
        tok = self.engine.tokenizer
        prompts = [tok.encode(build_prompt(p["question"], shots)) for p in problems]
        start = time.perf_counter()
        outputs = engine_generate(self.engine, prompts, max_new_tokens, batch_size)
        elapsed = time.perf_counter() - start
        rows = []
        for p, ids in zip(problems, outputs):
            completion = first_answer(tok.decode(ids, skip_special_tokens=True))
            pred, gold = extract_answer(completion), gold_answer(p["answer"])
            rows.append({"question": p["question"], "gold": gold, "predicted": pred, "correct": pred == gold,
                         "completion": completion})
        n = len(rows)
        acc = sum(r["correct"] for r in rows) / n if n else 0.0
        return {
            "n": n,
            "shots": len(shots),
            "exact_match": acc,
            "stderr": math.sqrt(acc * (1 - acc) / (n - 1)) if n > 1 else None,
            "wall_seconds": round(elapsed, 2),
            "results": rows,
        }
