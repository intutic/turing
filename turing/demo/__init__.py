"""
Turing Engine — Multi-Agent Deliberation & Interactive Inference Module.
"""

from .world_model import DynamicEnvironmentModel
from .epistemic_gate import EpistemicUncertaintyGate
from .engine_wrapper import TuringAcceleratedGenerator
from .agent_system import MultiAgentCoordinator
from .interactive_demo import run_demo

__all__ = [
    "DynamicEnvironmentModel",
    "EpistemicUncertaintyGate",
    "TuringAcceleratedGenerator",
    "MultiAgentCoordinator",
    "run_demo",
]
