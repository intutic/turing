"""
Model-backed embeddings and reranking for the side endpoints (/v1/embeddings, /api/embed, /v1/rerank).

Embeddings come from the served LLM's own final hidden states (embedding, decoder layers and final norm, no LM head),
mean-pooled over the tokens and L2-normalised, or from a sentence-transformers / BERT-style model loaded with
transformers' AutoModel (--embedding-model), pooled the way its 1_Pooling/config.json says (CLS or mean).
Reranking uses a cross-encoder (--rerank-model, AutoModelForSequenceClassification, sigmoid of its logit) or, without
one, the served LLM's probability of answering "yes" rather than "no" to a relevance question.
"""

import base64
import json
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def chat_prompt_ids(engine, content: str) -> List[int]:
    """Token ids of one user turn in the served model's chat template (the plain text when it has none)."""
    tok = engine.tokenizer
    if tok is not None and getattr(tok, "chat_template", None):
        text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                       add_generation_prompt=True)
        return list(tok.encode(text, add_special_tokens=False))
    return engine.encode_prompt(content)


def _padded(token_lists: List[List[int]], device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-padded ids [B, L] and lengths [B]. Right padding is exact for a causal model: real tokens never attend
    to the pad positions after them."""
    lens = torch.tensor([len(t) for t in token_lists], device=device)
    ids = torch.zeros(len(token_lists), int(lens.max()), dtype=torch.long, device=device)
    for i, t in enumerate(token_lists):
        ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    return ids, lens


@torch.inference_mode()
def llm_hidden_states(model, ids: torch.Tensor) -> torch.Tensor:
    """Final hidden states [B, L, H] of SubspaceCausalLM for ids [B, L] (positions from 0, no KV cache): the steps of
    SubspaceCausalLM._body up to and including the final norm. Optional recirculation and LoRA tenants are not applied."""
    cfg = model.config
    h = model.embed_tokens(ids)
    if cfg.embed_proj_dim:
        h = model.project_in(h)
    if cfg.embed_scale is not None:
        h = h * torch.tensor(cfg.embed_scale, dtype=h.dtype)
    if cfg.learned_positions:
        pos = torch.arange(ids.shape[1], device=ids.device)[None]
        h = h + model.embed_positions(pos + cfg.position_offset)
    for layer in model.layers:
        h, _, _ = layer(h, start_pos=0, cache=None)
    return model.norm(h) if model.norm is not None else h


@torch.inference_mode()
def llm_logits(model, hidden: torch.Tensor) -> torch.Tensor:
    """The LM head on final hidden states (what SubspaceCausalLM._body does after the norm)."""
    cfg = model.config
    if cfg.embed_proj_dim:
        hidden = model.project_out(hidden)
    logits = model.lm_head(hidden)
    if cfg.final_softcap is not None:
        logits = torch.tanh(logits / cfg.final_softcap) * cfg.final_softcap
    return logits


def llm_embed(model, token_lists: List[List[int]]) -> torch.Tensor:
    """Mean-pooled, L2-normalised final hidden states [N, H] (float32) of the served model."""
    device = model.embed_tokens.weight.device
    ids, lens = _padded(token_lists, device)
    h = llm_hidden_states(model, ids).float()
    mask = (torch.arange(ids.shape[1], device=device)[None] < lens[:, None]).unsqueeze(-1)
    pooled = (h * mask).sum(1) / lens[:, None]
    return F.normalize(pooled, dim=-1)


@torch.inference_mode()
def llm_yes_scores(engine, query: str, documents: List[str]) -> Tuple[List[float], int]:
    """P(yes | yes or no) of the served model answering whether each document is relevant to the query, from its
    next-token distribution after a relevance question. Returns the scores and the prompt tokens used."""
    model, tok = engine.model, engine.tokenizer
    prompts = [chat_prompt_ids(engine, "Is the document relevant to the query? Answer yes or no.\n\n"
                               f"Query: {query}\nDocument: {d}\n\nAnswer:") for d in documents]

    def first_ids(words):
        ids = {(tok.encode(w, add_special_tokens=False) if tok is not None else engine.encode_prompt(w))[0]
               for w in words}
        return torch.tensor(sorted(ids))

    yes, no = first_ids(["yes", "Yes", " yes", " Yes"]), first_ids(["no", "No", " no", " No"])
    device = model.embed_tokens.weight.device
    ids, lens = _padded(prompts, device)
    h = llm_hidden_states(model, ids)
    last = h[torch.arange(len(prompts), device=device), lens - 1]
    lp = torch.log_softmax(llm_logits(model, last).float(), -1).cpu()
    scores = torch.sigmoid(torch.logsumexp(lp[:, yes], -1) - torch.logsumexp(lp[:, no], -1))
    return scores.tolist(), int(lens.sum())


def _pooling(name: str) -> str:
    """"cls" or "mean", from a sentence-transformers model's 1_Pooling/config.json (mean when there is none)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    try:
        path = (os.path.join(name, "1_Pooling", "config.json") if os.path.isdir(name)
                else hf_hub_download(name, "1_Pooling/config.json"))
        with open(path) as f:
            return "cls" if json.load(f).get("pooling_mode_cls_token") else "mean"
    except (EntryNotFoundError, FileNotFoundError):
        return "mean"


class HFEmbedder:
    """A BERT-style embedding model (e.g. BAAI/bge-small-en-v1.5) through transformers' AutoModel."""

    def __init__(self, name: str, device):
        from transformers import AutoModel, AutoTokenizer
        self.name, self.device = name, torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        self.pooling = _pooling(name)

    @torch.inference_mode()
    def __call__(self, texts: List[str]) -> Tuple[torch.Tensor, int]:
        """L2-normalised embeddings [N, D] and the number of tokens read."""
        batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        h = self.model(**batch).last_hidden_state.float()
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = h[:, 0] if self.pooling == "cls" else (h * mask).sum(1) / mask.sum(1)
        return F.normalize(pooled, dim=-1), int(batch["attention_mask"].sum())


class HFReranker:
    """A cross-encoder (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2, BAAI/bge-reranker-base) scoring (query, document)
    pairs with AutoModelForSequenceClassification."""

    def __init__(self, name: str, device):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.name, self.device = name, torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, query: str, documents: List[str]) -> Tuple[List[float], int]:
        """Relevance in [0, 1] per document (sigmoid of a single logit, else the last class's softmax) and tokens read."""
        batch = self.tokenizer([query] * len(documents), documents, padding=True, truncation=True,
                               return_tensors="pt").to(self.device)
        logits = self.model(**batch).logits.float()
        scores = torch.sigmoid(logits[:, 0]) if logits.shape[-1] == 1 else torch.softmax(logits, -1)[:, -1]
        return scores.cpu().tolist(), int(batch["attention_mask"].sum())


def embedding_shape_error(width: int, dimensions: Optional[int], encoding_format: str) -> Optional[str]:
    """Why shape_embeddings would refuse these arguments for vectors of `width` values, or None if it would not."""
    if dimensions is not None and not 0 < dimensions <= width:
        return f"dimensions must be between 1 and {width}"
    if encoding_format not in ("float", "base64"):
        return "encoding_format must be 'float' or 'base64'"
    return None


def shape_embeddings(vectors: torch.Tensor, dimensions: Optional[int], encoding_format: str) -> list:
    """Matryoshka truncation to `dimensions` (renormalised), then a list of floats or base64 of little-endian float32
    per vector (OpenAI's encoding_format)."""
    if dimensions is not None:
        if not 0 < dimensions <= vectors.shape[-1]:
            raise ValueError(f"dimensions must be between 1 and {vectors.shape[-1]}")
        vectors = F.normalize(vectors[:, :dimensions], dim=-1)
    vectors = vectors.float().cpu()
    if encoding_format == "base64":
        return [base64.b64encode(v.numpy().astype("<f4").tobytes()).decode() for v in vectors]
    if encoding_format != "float":
        raise ValueError("encoding_format must be 'float' or 'base64'")
    return vectors.tolist()
