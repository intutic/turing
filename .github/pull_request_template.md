## Description
<!-- Provide a clear and concise summary of the changes in this PR and why they are needed. -->

## Associated Issues
<!-- e.g. Fixes #123, Resolves #456 -->
Fixes #

## Type of Change
<!-- Please check all that apply -->
- [ ] 🚀 New Feature (Subspace pruning, SVD KV compression, routing, etc.)
- [ ] ⚡ Performance Optimization / Kernel Acceleration
- [ ] 🐛 Bug Fix
- [ ] 📖 Documentation / Integration Guides
- [ ] 🧪 Tests / CI Automation
- [ ] 🧹 Refactoring / Code Lean Clean-up

## Verification & Testing
<!-- Describe the verification steps and tests run. -->
- [ ] All automated tests pass with 0 warnings (`python -W error -m pytest -v`)
- [ ] SIMD / GPU numerical parity verified against reference implementation (if applicable)
- [ ] Tested on target hardware (e.g. NVIDIA CUDA, Apple Silicon Metal, or x86_64 AVX2)

### Test Command
```bash
python -W error -m pytest -q
```

## Checklist
- [ ] My code adheres to the architecture invariants in `AGENTS.md` (64-byte alignment, no hallucinated jargon)
- [ ] I have signed off all commits (`git commit -s`) per Developer Certificate of Origin (DCO)
- [ ] Relevant documentation in `docs/` has been updated
