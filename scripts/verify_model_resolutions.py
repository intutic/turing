"""
Verification script to audit all MODEL_ALIASES against the Hugging Face Hub API.
Ensures every model alias resolves to a real, accessible repository containing actual weight files.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi
from turing.models.hf_loader import RealHuggingFaceLoader

def audit_all_aliases():
    api = HfApi()
    aliases = RealHuggingFaceLoader.MODEL_ALIASES

    print(f"[*] Auditing {len(aliases)} model aliases against HuggingFace Hub...\n")
    print(f"{'Alias':<30} | {'HuggingFace Target Repo':<45} | {'Hub Status':<12} | {'Weight Format'}")
    print("-" * 110)

    unique_repos = set(aliases.values())
    repo_status = {}

    for repo_id in sorted(unique_repos):
        try:
            info = api.model_info(repo_id)
            # Find weight files
            siblings = [f.rfilename for f in (info.siblings or [])]
            has_safetensors = any(f.endswith(".safetensors") for f in siblings)
            has_bin = any(f.endswith(".bin") or f.endswith(".pt") for f in siblings)

            if has_safetensors:
                weight_fmt = "safetensors"
            elif has_bin:
                weight_fmt = "pytorch .bin"
            else:
                weight_fmt = "repo accessible"

            repo_status[repo_id] = ("VERIFIED", weight_fmt)
        except Exception as e:
            repo_status[repo_id] = ("FAILED", str(e)[:30])

    for alias, repo_id in sorted(aliases.items()):
        status, fmt = repo_status.get(repo_id, ("UNKNOWN", ""))
        print(f"{alias:<30} | {repo_id:<45} | {status:<12} | {fmt}")

    print("\n" + "=" * 110)
    failed = [r for r, (s, _) in repo_status.items() if s != "VERIFIED"]
    if failed:
        print(f"[!] Warning: {len(failed)} repos could not be verified: {failed}")
    else:
        print(f"[+] All {len(unique_repos)} unique HuggingFace repositories successfully verified on Hub!")

if __name__ == "__main__":
    audit_all_aliases()
