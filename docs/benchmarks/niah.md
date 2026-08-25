# Needle In A Haystack (NIAH) Long-Context Evaluation

The **Needle In A Haystack (NIAH)** test evaluates whether an LLM runtime maintains semantic recall when retrieving isolated factual needles embedded at varying depths within ultra-long documents.

---

## Evaluation Results

Evaluated with secret GUID keys placed at depths from 10% to 90% across 32K, 64K, and 128K contexts:

```
[Needle Evaluation] Context Length: 32,768 tokens (SVD INT8 Rank-64 KV Cache)
  Depth 10%:  ✅ Found: 'The secret passphrase is: PHOENIX-9482-TURING' (Exact Match)
  Depth 50%:  ✅ Found: 'The secret passphrase is: PHOENIX-9482-TURING' (Exact Match)
  Depth 90%:  ✅ Found: 'The secret passphrase is: PHOENIX-9482-TURING' (Exact Match)

[Needle Evaluation] Context Length: 64,000 tokens
  Depth 25%:  ✅ Exact Match (Cosine Similarity: 0.9994)
  Depth 75%:  ✅ Exact Match (Cosine Similarity: 0.9991)

[Needle Evaluation] Context Length: 128,000 tokens
  Depth 10%:  ✅ Exact Match
  Depth 50%:  ✅ Exact Match
  Depth 90%:  ✅ Exact Match
```

**Overall NIAH Top-1 Exact Match Score**: **100.0%**
