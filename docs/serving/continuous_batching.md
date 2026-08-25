# Continuous Batching Engine

Turing Engine features an iteration-level **Continuous Batching Engine** that schedules dynamic request arrivals, chunks long prompt prefills, and interleaves decoding steps.

---

## Architecture Features

- **Chunked Prefilling**: Divides massive prompt contexts into configurable chunks (default: 512 tokens) to prevent blocking active decoding streams.
- **Static Paged KV Pooling**: Pre-allocates a fixed contiguous memory buffer for KV pages, eliminating runtime memory allocations (`malloc`/`cudaMalloc`).
- **Asynchronous Token Streaming**: Emits tokens as standard asynchronous iterators / Server-Sent Events (SSE).
