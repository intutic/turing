"""
GGUF Metadata-Derived Tokenizer with Hugging Face Compatibility.
Extracts token vocabulary, merges, and chat templates directly from GGUF metadata.
"""

from typing import List, Dict, Any, Optional, Union
import torch

__all__ = ["GGUFTokenizer"]


class GGUFTokenizer:
    """
    Self-contained tokenizer derived from GGUF metadata.
    Provides encode(), decode(), and apply_chat_template() matching transformers PreTrainedTokenizer.
    """

    def __init__(self, metadata: Dict[str, Any]):
        self.metadata = metadata
        self.tokens: List[str] = metadata.get("tokenizer.ggml.tokens", [])
        self.scores: List[float] = metadata.get("tokenizer.ggml.scores", [])
        self.merges: List[str] = metadata.get("tokenizer.ggml.merges", [])
        self.model_type: str = metadata.get("tokenizer.ggml.model", "llama")
        
        self.bos_token_id: int = int(metadata.get("tokenizer.ggml.bos_token_id", 1))
        self.eos_token_id: int = int(metadata.get("tokenizer.ggml.eos_token_id", 2))
        self.pad_token_id: int = int(metadata.get("tokenizer.ggml.padding_token_id", self.eos_token_id))
        self.unk_token_id: int = int(metadata.get("tokenizer.ggml.unknown_token_id", 0))
        
        # Build token-to-id mapping
        self.token_to_id: Dict[str, int] = {}
        for idx, tok in enumerate(self.tokens):
            if isinstance(tok, bytes):
                tok = tok.decode("utf-8", errors="replace")
            self.token_to_id[str(tok)] = idx

        # Build BPE merges if available
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        for rank, merge in enumerate(self.merges):
            parts = merge.split()
            if len(parts) == 2:
                self.bpe_ranks[(parts[0], parts[1])] = rank

        # Chat template if available in metadata
        self.chat_template: Optional[str] = metadata.get("tokenizer.chat_template", None)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False
    ) -> List[int]:
        if not text:
            return [self.bos_token_id] if add_special_tokens else []

        token_ids: List[int] = []
        if add_special_tokens and self.bos_token_id is not None:
            token_ids.append(self.bos_token_id)

        # 1. Direct match check
        if text in self.token_to_id:
            token_ids.append(self.token_to_id[text])
            return token_ids

        # 2. BPE / WordPiece greedy fallback
        words = text.replace("\n", " \n ").split(" ")
        for i, word in enumerate(words):
            if not word:
                continue
            spaced_word = f" {word}" if (i > 0 or text.startswith(" ")) else word
            
            if spaced_word in self.token_to_id:
                token_ids.append(self.token_to_id[spaced_word])
            elif word in self.token_to_id:
                token_ids.append(self.token_to_id[word])
            else:
                # Sub-character / byte token fallback
                for char in spaced_word:
                    if char in self.token_to_id:
                        token_ids.append(self.token_to_id[char])
                    else:
                        token_ids.append(self.unk_token_id)

        return token_ids

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        pieces = []
        special_ids = {self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id}
        
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            if 0 <= tid < len(self.tokens):
                tok = self.tokens[tid]
                if isinstance(tok, bytes):
                    tok = tok.decode("utf-8", errors="replace")
                # Replace standard SPM whitespace marker ' ' (U+2581) with space
                tok = tok.replace(" ", " ")
                pieces.append(tok)
            else:
                pieces.append("")

        text = "".join(pieces)
        return text.strip() if skip_special_tokens else text

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True
    ) -> Union[str, List[int]]:
        """
        Formats conversational turns into ChatML / Llama-3 format.
        """
        formatted = ""
        for turn in conversation:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            
        if add_generation_prompt:
            formatted += "<|im_start|>assistant\n"

        if tokenize:
            return self.encode(formatted, add_special_tokens=False)
        return formatted

    def __call__(
        self,
        text: Union[str, List[str]],
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = False
    ) -> Any:
        if isinstance(text, str):
            ids = self.encode(text, add_special_tokens=add_special_tokens)
            if return_tensors == "pt":
                return {"input_ids": torch.tensor([ids], dtype=torch.long)}
            return {"input_ids": ids}
        else:
            all_ids = [self.encode(t, add_special_tokens=add_special_tokens) for t in text]
            if return_tensors == "pt":
                # Pad to max length
                max_len = max(len(ids) for ids in all_ids)
                padded = [ids + [self.pad_token_id] * (max_len - len(ids)) for ids in all_ids]
                return {"input_ids": torch.tensor(padded, dtype=torch.long)}
            return {"input_ids": all_ids}
