"""
Demo generator: a HuggingFace transformers model on MPS, CUDA or CPU with chat formatting, an epistemic
uncertainty reading of the prompt's logits, and timing.
"""

import os
import time
from typing import Tuple, List, Dict, Optional, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

# Silence HF hub notices
os.environ["HF_HUB_DISABLE_UNAUTHENTICATED_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
hf_logging.set_verbosity_error()

import huggingface_hub
huggingface_hub.logging.set_verbosity_error()

from turing.experimental.demo.epistemic_gate import EpistemicUncertaintyGate


class TuringAcceleratedGenerator:
    """
    Runs a transformers causal LM (by id or instance) with its own generate(); the Turing engine is not involved.
    """
    def __init__(
        self,
        model_id_or_instance: Union[str, nn.Module] = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        tokenizer: Optional[Any] = None,
        device: str = "auto",
        token: Optional[str] = None
    ):
        # 1. Resolve Target Hardware Interconnect
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            from turing.config import TuringConfig
            self.device = TuringConfig(device=device).resolve_device()  # maps rocm to HIP's "cuda"

        self.hf_token = token or os.environ.get("HF_TOKEN") or huggingface_hub.get_token()
        self.epistemic_gate = EpistemicUncertaintyGate(uncertainty_threshold=2.5)

        # 2. Load Model & Tokenizer if string identifier provided
        if isinstance(model_id_or_instance, str):
            self.model_id = model_id_or_instance
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.hf_token)
            
            # Select optimal precision per device
            target_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                token=self.hf_token,
                dtype=target_dtype,
                low_cpu_mem_usage=True
            ).to(self.device).eval()
        else:
            self.model = model_id_or_instance.to(self.device).eval()
            self.tokenizer = tokenizer
            self.model_id = getattr(self.model, "name_or_path", "custom-model")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def sync_device(self):
        """Synchronizes device compute queue for microsecond-precise latency metrics."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @torch.no_grad()
    def fast_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Generates a reply with chat formatting, timing and an uncertainty reading of the prompt logits.
        """
        # Format message conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted_prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        # Evaluate Epistemic Uncertainty on prompt prefix
        with torch.no_grad():
            initial_out = self.model(**inputs, use_cache=True)
            initial_logits = initial_out.logits
            epistemic_diagnostics = self.epistemic_gate.evaluate_step_uncertainty(initial_logits)

        self.sync_device()
        t0 = time.perf_counter()

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0.0),
            pad_token_id=self.tokenizer.eos_token_id
        )

        self.sync_device()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Decode newly generated tokens only
        new_token_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

        new_tokens_count = len(new_token_ids)
        throughput = (new_tokens_count / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0

        return {
            "response_text": response_text,
            "latency_ms": round(elapsed_ms, 2),
            "tokens_generated": new_tokens_count,
            "throughput_tok_s": round(throughput, 1),
            "epistemic_diagnostics": epistemic_diagnostics,
            "device": str(self.device)
        }

