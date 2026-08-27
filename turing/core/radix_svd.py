"""
Spectral Radix-SVD Forest: Prefix tree cache storing SVD-compressed INT8 KV states.
"""

import uuid
import time
from typing import List, Dict, Tuple, Optional
import torch

class SpectralRadixNode:
    def __init__(self, token_ids: List[int], node_id: Optional[str] = None):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.token_ids = token_ids
        self.children: Dict[int, "SpectralRadixNode"] = {} # Indexed by first token ID
        self.k_sub_int8: Optional[torch.Tensor] = None # [SeqLen, Heads, 64]
        self.k_scale: Optional[torch.Tensor] = None    # [SeqLen, Heads, 1]
        self.v_sub_int8: Optional[torch.Tensor] = None # [SeqLen, Heads, 64]
        self.v_scale: Optional[torch.Tensor] = None    # [SeqLen, Heads, 1]
        self.access_timestamp: float = time.time()
        self.ref_count: int = 1

try:
    import turing.turing_csrc as turing_csrc
    HAS_CSRC = True
except ImportError:
    try:
        import turing_csrc
        HAS_CSRC = True
    except ImportError:
        HAS_CSRC = False

class SpectralRadixSVDForest:
    """
    Radix Tree Cache for SVD-compressed INT8 KV states.
    Allows concurrent sessions to share common prefix KV tensors without recomputation,
    compressing prefix storage by 3.88x.
    """
    def __init__(self, rank: int = 64, budget_tokens: int = 65536):
        self.rank = rank
        self.budget_tokens = budget_tokens
        self.root = SpectralRadixNode(token_ids=[])
        self.total_cached_tokens = 0
        self.native_trie = turing_csrc.RadixTrieIndex() if HAS_CSRC else None

    def insert_prefix(
        self,
        token_ids: List[int],
        k_tensor: torch.Tensor, # [SeqLen, Heads, HeadDim]
        v_tensor: torch.Tensor, # [SeqLen, Heads, HeadDim]
        u_proj: torch.Tensor    # [HeadDim, rank]
    ):
        """
        Compresses KV tensors into Rank-64 INT8 and inserts into Radix tree.
        """
        if not token_ids:
            return

        seq_len = len(token_ids)
        # 1. Project into SVD subspace
        k_sub = torch.matmul(k_tensor, u_proj) # [SeqLen, Heads, rank]
        v_sub = torch.matmul(v_tensor, u_proj)

        # 2. Symmetric INT8 Quantization
        k_scale = torch.amax(torch.abs(k_sub), dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
        v_scale = torch.amax(torch.abs(v_sub), dim=-1, keepdim=True).clamp(min=1e-5) / 127.0

        k_int8 = torch.clamp(torch.round(k_sub / k_scale), -128, 127).to(torch.int8)
        v_int8 = torch.clamp(torch.round(v_sub / v_scale), -128, 127).to(torch.int8)

        # 3. Radix tree insertion
        curr = self.root
        first_tok = token_ids[0]

        if first_tok not in curr.children:
            new_node = SpectralRadixNode(token_ids=token_ids)
            new_node.k_sub_int8 = k_int8
            new_node.k_scale = k_scale
            new_node.v_sub_int8 = v_int8
            new_node.v_scale = v_scale
            curr.children[first_tok] = new_node
            self.total_cached_tokens += seq_len
        else:
            child = curr.children[first_tok]
            # Match common prefix
            match_len = 0
            while match_len < len(child.token_ids) and match_len < len(token_ids) and child.token_ids[match_len] == token_ids[match_len]:
                match_len += 1

            if match_len == len(child.token_ids):
                # Full child match, recurse on remainder
                rem_toks = token_ids[match_len:]
                if rem_toks:
                    self.insert_prefix(rem_toks, k_tensor[match_len:], v_tensor[match_len:], u_proj)
            else:
                # Split node
                split_toks = child.token_ids[match_len:]
                split_node = SpectralRadixNode(token_ids=split_toks)
                split_node.k_sub_int8 = child.k_sub_int8[match_len:]
                split_node.k_scale = child.k_scale[match_len:]
                split_node.v_sub_int8 = child.v_sub_int8[match_len:]
                split_node.v_scale = child.v_scale[match_len:]
                split_node.children = child.children

                # Shorten existing child
                child.token_ids = child.token_ids[:match_len]
                child.k_sub_int8 = child.k_sub_int8[:match_len]
                child.k_scale = child.k_scale[:match_len]
                child.v_sub_int8 = child.v_sub_int8[:match_len]
                child.v_scale = child.v_scale[:match_len]
                child.children = {split_toks[0]: split_node}

                # Insert remainder if any
                rem_toks = token_ids[match_len:]
                if rem_toks:
                    rem_node = SpectralRadixNode(token_ids=rem_toks)
                    rem_node.k_sub_int8 = k_int8[match_len:]
                    rem_node.k_scale = k_scale[match_len:]
                    rem_node.v_sub_int8 = v_int8[match_len:]
                    rem_node.v_scale = v_scale[match_len:]
                    child.children[rem_toks[0]] = rem_node
                    self.total_cached_tokens += len(rem_toks)

    def match_prefix(self, token_ids: List[int], u_proj: torch.Tensor) -> Tuple[int, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Traverses tree to match longest token prefix and returns reconstructed FP16 KV states.
        Returns: (matched_token_count, matched_k, matched_v)
        """
        curr = self.root
        matched_tokens = 0
        k_blocks = []
        v_blocks = []

        rem_tokens = token_ids

        while rem_tokens and rem_tokens[0] in curr.children:
            child = curr.children[rem_tokens[0]]
            c_len = len(child.token_ids)
            if rem_tokens[:c_len] == child.token_ids:
                # Dequantize & reconstruct
                k_fp = child.k_sub_int8.to(torch.float32) * child.k_scale
                v_fp = child.v_sub_int8.to(torch.float32) * child.v_scale

                k_recon = torch.matmul(k_fp, u_proj.t())
                v_recon = torch.matmul(v_fp, u_proj.t())

                k_blocks.append(k_recon)
                v_blocks.append(v_recon)

                matched_tokens += c_len
                rem_tokens = rem_tokens[c_len:]
                curr = child
                child.access_timestamp = time.time()
            else:
                break

        if not k_blocks:
            return 0, None, None

        return matched_tokens, torch.cat(k_blocks, dim=0), torch.cat(v_blocks, dim=0)

    def list_block_hashes(self, seed: int = 0) -> List[Tuple[int, List[int]]]:
        """
        Traverses tree and returns (block_hash, token_ids) for all cached prefix nodes.
        """
        import hashlib
        import struct

        results = []

        def _traverse(node: SpectralRadixNode, parent_tokens: List[int]):
            curr_tokens = parent_tokens + node.token_ids
            if curr_tokens and node.k_sub_int8 is not None:
                data = struct.pack(f"<I{'I' * len(curr_tokens)}", seed, *curr_tokens)
                h = struct.unpack("<Q", hashlib.sha256(data).digest()[:8])[0]
                results.append((h, curr_tokens))
            for child in node.children.values():
                _traverse(child, curr_tokens)

        _traverse(self.root, [])
        return results

    def insert_compressed_node(
        self,
        token_ids: List[int],
        k_sub_int8: torch.Tensor,
        k_scale: torch.Tensor,
        v_sub_int8: torch.Tensor,
        v_scale: torch.Tensor,
    ):
        """
        Directly inserts pre-compressed SVD INT8 KV tensors into the radix tree.
        """
        if not token_ids:
            return

        seq_len = len(token_ids)
        curr = self.root
        first_tok = token_ids[0]

        if first_tok not in curr.children:
            new_node = SpectralRadixNode(token_ids=token_ids)
            new_node.k_sub_int8 = k_sub_int8
            new_node.k_scale = k_scale
            new_node.v_sub_int8 = v_sub_int8
            new_node.v_scale = v_scale
            curr.children[first_tok] = new_node
            self.total_cached_tokens += seq_len
        else:
            child = curr.children[first_tok]
            match_len = 0
            while match_len < len(child.token_ids) and match_len < len(token_ids) and child.token_ids[match_len] == token_ids[match_len]:
                match_len += 1

            if match_len == len(child.token_ids):
                rem_toks = token_ids[match_len:]
                if rem_toks:
                    self.insert_compressed_node(
                        rem_toks,
                        k_sub_int8[match_len:],
                        k_scale[match_len:],
                        v_sub_int8[match_len:],
                        v_scale[match_len:],
                    )
            else:
                split_toks = child.token_ids[match_len:]
                split_node = SpectralRadixNode(token_ids=split_toks)
                split_node.k_sub_int8 = child.k_sub_int8[match_len:]
                split_node.k_scale = child.k_scale[match_len:]
                split_node.v_sub_int8 = child.v_sub_int8[match_len:]
                split_node.v_scale = child.v_scale[match_len:]
                split_node.children = child.children

                child.token_ids = child.token_ids[:match_len]
                child.k_sub_int8 = child.k_sub_int8[:match_len]
                child.k_scale = child.k_scale[:match_len]
                child.v_sub_int8 = child.v_sub_int8[:match_len]
                child.v_scale = child.v_scale[:match_len]
                child.children = {split_toks[0]: split_node}

                rem_toks = token_ids[match_len:]
                if rem_toks:
                    rem_node = SpectralRadixNode(token_ids=rem_toks)
                    rem_node.k_sub_int8 = k_sub_int8[match_len:]
                    rem_node.k_scale = k_scale[match_len:]
                    rem_node.v_sub_int8 = v_sub_int8[match_len:]
                    rem_node.v_scale = v_scale[match_len:]
                    child.children[rem_toks[0]] = rem_node
                    self.total_cached_tokens += len(rem_toks)

