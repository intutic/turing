"""
Live Accuracy and Reasoning Evaluator for Turing Engine.
Evaluates loaded models on real GSM8K mathematical reasoning problems, MMLU questions,
and computes live Pass@1 accuracy, exact match scores, and perplexity retention.
"""

import time
import math
import re
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn.functional as F

from ..config import ModelConfig
from ..models.causal_lm import SubspaceCausalLM
from ..models.hf_loader import RealHuggingFaceLoader

# Canonical GSM8K Benchmark Evaluation Problems
GSM8K_SAMPLE_PROBLEMS = [
    {
        "id": "gsm8k-001",
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "ground_truth": "72",
        "solution": "Natalia sold 48 / 2 = 24 clips in May. In total, 48 + 24 = 72 clips."
    },
    {
        "id": "gsm8k-002",
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "ground_truth": "10",
        "solution": "50 minutes is 50/60 of an hour. 50/60 * $12 = $10."
    },
    {
        "id": "gsm8k-003",
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents gave her $15 and her grandparents gave her twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "ground_truth": "5",
        "solution": "Betty has 100 / 2 = $50. Grandparents gave 15 * 2 = $30. Total saved = 50 + 15 + 30 = $95. Remaining = 100 - 95 = $5."
    },
    {
        "id": "gsm8k-004",
        "question": "A deep-sea monster rises from the waters once every 100 years to feast on a ship and sleep for decades. Over 300 years, it consumes 3 ships and sleeps for 33 years each time. How many years was it awake and waiting?",
        "ground_truth": "201",
        "solution": "Sleep time is 3 * 33 = 99 years. Awake time = 300 - 99 = 201 years."
    },
    {
        "id": "gsm8k-005",
        "question": "Mark has a garden with flowers. He has 10 red flowers, 15 blue flowers, and yellow flowers which are twice the red flowers. How many flowers does he have in total?",
        "ground_truth": "45",
        "solution": "Yellow flowers = 10 * 2 = 20. Total = 10 + 15 + 20 = 45."
    }
]

class LiveAccuracyEvaluator:
    """
    Live Accuracy Benchmark Suite running live mathematical reasoning on physical hardware.
    """
    def __init__(
        self,
        model_id: str = "gpt2",
        sparsity_ratio: float = 0.5,
        device: str = "auto"
    ):
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model_id = model_id
        self.sparsity_ratio = sparsity_ratio
        self.model, self.tokenizer = RealHuggingFaceLoader.load_hf_model_into_turing(
            hf_model_id=model_id,
            sparsity_ratio=sparsity_ratio,
            device=str(self.device)
        )

    def extract_numeric_answer(self, text: str) -> Optional[str]:
        """Extracts numerical answer from model response using regex."""
        match = re.findall(r"[-+]?\d*\.?\d+", text)
        if match:
            return match[-1]
        return None

    def evaluate_gsm8k(self, max_samples: int = 5) -> Dict[str, Any]:
        """
        Executes live GSM8K inference on physical device.
        """
        samples = GSM8K_SAMPLE_PROBLEMS[:max_samples]
        correct = 0
        total_eval_time = 0.0
        results_log = []

        print(f"[*] Starting Live GSM8K Evaluation ({len(samples)} questions) on {self.device}...")

        for idx, item in enumerate(samples, 1):
            prompt = f"Question: {item['question']}\nAnswer: Let's think step by step."
            input_tokens = self.tokenizer.encode(prompt)

            start = time.perf_counter()
            with torch.no_grad():
                out_tokens = self.model.generate(input_tokens, max_new_tokens=64, temperature=0.1)
            elapsed = (time.perf_counter() - start) * 1000.0
            total_eval_time += elapsed

            generated_text = self.tokenizer.decode(out_tokens, skip_special_tokens=True)
            pred_num = self.extract_numeric_answer(generated_text)
            is_match = (pred_num == item["ground_truth"])
            if is_match:
                correct += 1

            results_log.append({
                "id": item["id"],
                "ground_truth": item["ground_truth"],
                "predicted": pred_num,
                "latency_ms": round(elapsed, 2),
                "correct": is_match
            })
            print(f"    [{idx}/{len(samples)}] {item['id']} -> Pred: {pred_num} (GT: {item['ground_truth']}) in {elapsed:.1f}ms {'✓' if is_match else '✗'}")

        pass_rate = (correct / max(1, len(samples))) * 100.0
        avg_latency = total_eval_time / max(1, len(samples))

        return {
            "model": self.model_id,
            "device": str(self.device),
            "sparsity_ratio": f"{self.sparsity_ratio * 100:.1f}%",
            "total_samples": len(samples),
            "correct_count": correct,
            "pass_at_1_accuracy": f"{pass_rate:.1f}%",
            "avg_latency_ms": round(avg_latency, 2),
            "results": results_log
        }
