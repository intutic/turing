"""
Unit and Integration Tests for Turing Engine × Multi-Agent Constraint Optimization Demo.
"""

import pytest
import torch
import torch.nn as nn
from turing.demo.world_model import DynamicEnvironmentModel
from turing.demo.epistemic_gate import EpistemicUncertaintyGate
from turing.demo.engine_wrapper import TuringAcceleratedGenerator
from turing.demo.agent_system import MultiAgentCoordinator


def test_world_model_state_shift_and_constraint_penalty():
    wm = DynamicEnvironmentModel()

    # 1. Test nominal state
    nominal_state = wm.update_hidden_state_from_pixels_or_logs(force_alert=False)
    assert nominal_state["regional_demand_spike"] is False
    assert len(nominal_state["active_outages"]) == 0

    # 2. Test state shift (outage + spike)
    alert_state = wm.update_hidden_state_from_pixels_or_logs(force_alert=True)
    assert alert_state["regional_demand_spike"] is True
    assert "AWS_EU_WEST_NODE_4" in alert_state["active_outages"]

    # 3. Test unrevised plan: 2000 nodes (capacity 10M < 15M demand), routing through EU_NODE_4
    fb_unrevised = wm.evaluate_constraint_penalty(
        proposed_nodes=2000,
        proposed_distance_km=8000,
        routed_nodes=["AWS_US_EAST_1", "AWS_EU_WEST_NODE_4"]
    )
    assert fb_unrevised["constraint_penalty"] == 1300  # 1000 capacity + 300 dead zone
    assert fb_unrevised["revision_needed"] is True
    assert len(fb_unrevised["violations"]) == 2

    # 4. Test revised plan: 3000 nodes (capacity 15M == 15M demand), bypassing EU_NODE_4
    fb_revised = wm.evaluate_constraint_penalty(
        proposed_nodes=3000,
        proposed_distance_km=8000,
        routed_nodes=["AWS_US_EAST_1", "AWS_EU_CENTRAL_1"]
    )
    assert fb_revised["constraint_penalty"] == 0
    assert fb_revised["revision_needed"] is False
    assert len(fb_revised["violations"]) == 0


def test_epistemic_uncertainty_gate():
    gate = EpistemicUncertaintyGate(uncertainty_threshold=2.5)

    # Sharp/confident distribution (low entropy)
    sharp_logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
    sharp_diag = gate.evaluate_step_uncertainty(sharp_logits)
    assert sharp_diag["entropy"] < 0.1
    assert sharp_diag["is_uncertain"] is False
    assert sharp_diag["action"] == "CONFIDENT_EXECUTION"

    # Uniform/uncertain distribution (high entropy)
    uniform_logits = torch.ones(1, 100)
    uniform_diag = gate.evaluate_step_uncertainty(uniform_logits)
    assert uniform_diag["entropy"] > 4.0
    assert uniform_diag["is_uncertain"] is True
    assert uniform_diag["action"] == "TRIGGER_EPISTEMIC_EXPLORATION"


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(256, 64)
        self.head = nn.Linear(64, 256)
    
    def forward(self, input_ids, **kwargs):
        class Output:
            def __init__(self, logits):
                self.logits = logits
        h = self.embed(input_ids)
        logits = self.head(h)
        return Output(logits)

    def generate(self, input_ids, max_new_tokens=10, **kwargs):
        # Appends 5 dummy token IDs
        new_ids = torch.tensor([[42] * 5], device=input_ids.device)
        return torch.cat([input_ids, new_ids], dim=-1)


class MockTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["content"] for m in messages)

    def __call__(self, text, **kwargs):
        class BatchEncoding(dict):
            def __init__(self):
                super().__init__({"input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long)})
            def to(self, dev):
                self["input_ids"] = self["input_ids"].to(dev)
                return self
        return BatchEncoding()

    def decode(self, token_ids, **kwargs):
        return "Configured 3,000 multi-region nodes bypassing failed cluster."


def test_multi_agent_coordinator_with_mock_engine():
    mock_model = MockModel()
    mock_tokenizer = MockTokenizer()
    engine = TuringAcceleratedGenerator(
        model_id_or_instance=mock_model,
        tokenizer=mock_tokenizer,
        sparsity_ratio=0.57,
        device="cpu"
    )
    wm = DynamicEnvironmentModel()
    agent_sys = MultiAgentCoordinator(engine=engine, world_model=wm)

    results = agent_sys.run_multi_agent_workflow(force_outage=True, max_tokens_per_agent=10)

    assert results["revision_successful"] is True
    assert "3,000" in results["sol_proposal"]["response_text"]
    assert results["initial_world_feedback"]["constraint_penalty"] == 1300
    assert results["final_world_feedback"]["constraint_penalty"] == 0
