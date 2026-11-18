"""
`turing convert`: a HuggingFace checkpoint -> a 4-bit .tgate4 directory (packed weights, config, tokenizer), converted
layer by layer with bounded memory (see streaming_loader.py). Load it with `turing serve --model <dir>`.
"""

from typing import Any, Dict

from .streaming_loader import StreamingHFSubspaceLoader


def convert_checkpoint(model_id: str, out_dir: str, method: str = "gptq", device: str = "cpu",
                       calib_windows: int = 128, calib_tokens: int = 2048, lm_head: bool = False) -> Dict[str, Any]:
    """Quantize every attention and dense-MLP weight of model_id to 4 bits (GPTQ on WikiText-2 train windows, or RTN),
    and the LM head too with lm_head, and write out_dir/model.tgate4 plus the tokenizer."""
    from .resolver import ModelResolver
    loader = StreamingHFSubspaceLoader(ModelResolver.resolve_repo_id(model_id) if "/" not in model_id and not
                                       model_id.startswith(".") else model_id, target_device=device, method=method,
                                       calib_windows=calib_windows, calib_tokens=calib_tokens, lm_head=lm_head)
    return loader.convert(out_dir)
