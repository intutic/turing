# DSPy Integration Guide

[DSPy](https://github.com/stanfordnlp/dspy) optimizes LM prompts and weights algorithmically. Turing Engine serves as the underlying fast local execution engine for DSPy module compilation.

---

## 1. Quick Start

```bash
pip install dspy-ai
```

```python
import dspy

# Configure DSPy to use Turing Engine local endpoint
lm = dspy.LM("openai/llama-3.1-70b", api_base="http://localhost:8000/v1", api_key="turing-local")
dspy.settings.configure(lm=lm)

# Define a reasoning signature
class MultiStepReasoning(dspy.Signature):
    """Solve math problem with step-by-step reasoning."""
    question = dspy.InputField(desc="The mathematical question")
    reasoning = dspy.OutputField(desc="Step-by-step mathematical deduction")
    answer = dspy.OutputField(desc="Final concise answer")

cot = dspy.ChainOfThought(MultiStepReasoning)
result = cot(question="Janet has 3 times as many marbles as Tom. Tom has 12. Janet gives 10 away. How many does she have?")

print("--- Reasoning ---\n", result.reasoning)
print("\n--- Answer ---\n", result.answer)
```
