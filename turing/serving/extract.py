"""
Span extraction (/v1/extract) with the served LLM: greedy generation through the engine asks for the spans of the
requested labels as JSON, and every span returned is located in the input text, so offsets always index real
substrings. A span the text does not contain is dropped, never invented; output that is not JSON is an error.
"""

import json
import math
from typing import Any, Dict, List

import torch

from .embeddings import chat_prompt_ids
from .structured import StructuredOutputParser

PROMPT = ("Extract every span of the text that is an entity of one of these types: {labels}.\n"
          "Copy each span exactly as it is written in the text. Answer with only a JSON array of objects with keys "
          "\"label\" and \"text\", or [] if there are none.\n\nText: {text}")


def _items(parsed: Any) -> List[tuple]:
    """(label, text) pairs from [{"label", "text"}, ...] or {label: text or [texts]}."""
    if isinstance(parsed, dict) and all(isinstance(v, (str, list)) for v in parsed.values()):
        return [(k, t) for k, v in parsed.items() for t in ([v] if isinstance(v, str) else v)]
    if isinstance(parsed, list):
        return [(e.get("label"), e.get("text")) for e in parsed if isinstance(e, dict)]
    return None


async def extract_spans(engine, text: str, labels: List[str], max_tokens: int = 512) -> Dict[str, Any]:
    """Spans {label, text, start, end, score} of text for the given labels, plus the model's raw output and the
    pairs it named that the text does not contain. If the output is not the JSON asked for, the result is
    {"error": True, "raw_output": ...} instead. score is the geometric-mean probability of the generated tokens
    that spell the span (greedy decoding, so each is the model's top choice)."""
    prompt = chat_prompt_ids(engine, PROMPT.format(labels=", ".join(labels), text=text))
    logprobs = []  # log-probability of each generated token (the argmax, under greedy decoding)
    out = []
    async for tok in engine.stream_generate(prompt, max_new_tokens=max_tokens, temperature=0.0,
                                            logits_processor=lambda _, logits: logprobs.append(
                                                torch.log_softmax(logits.float(), -1).max())):
        out.append(tok)
    raw = engine.decode_tokens(out)
    ok, parsed, _ = StructuredOutputParser.extract_json(raw)
    items = _items(parsed) if ok else None
    if items is None:
        return {"error": True, "raw_output": raw}

    lps = torch.stack(logprobs[:len(out)]).tolist() if out else []
    ends = [len(engine.decode_tokens(out[:i + 1])) for i in range(len(out))]  # char offset in raw after token i
    canon = {lab.lower(): lab for lab in labels}
    spans, unmatched, cursor, used = [], [], 0, set()
    for label, span in items:
        label = canon.get(str(label).lower()) if label is not None else None
        if label is None or not isinstance(span, str) or not span.strip():
            unmatched.append({"label": label, "text": span})
            continue
        start = text.find(span)
        while start != -1 and (label, start) in used:  # a repeated span takes the next occurrence
            start = text.find(span, start + 1)
        if start == -1:
            unmatched.append({"label": label, "text": span})
            continue
        used.add((label, start))
        # the generated tokens that spell this span in the raw output (all of them when it cannot be found there)
        at = raw.find(json.dumps(span, ensure_ascii=False)[1:-1], cursor)
        if at != -1:
            cursor = at + 1
            sel = [lp for lp, e, s in zip(lps, ends, [0] + ends[:-1]) if e > at and s < at + len(span)]
        else:
            sel = lps
        score = math.exp(sum(sel) / len(sel)) if sel else None
        spans.append({"label": label, "text": span, "start": start, "end": start + len(span),
                      "score": round(score, 4) if score is not None else None})
    return {"spans": spans, "unmatched": unmatched, "raw_output": raw, "prompt_tokens": len(prompt),
            "completion_tokens": len(out)}
