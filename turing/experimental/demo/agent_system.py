"""
Multi-Agent Deliberation & Self-Revision System.
Orchestrates Proposal Generator and Revision Optimizer over Active Inference gates.
"""

import json
import re
from typing import Dict, Any, Tuple, Optional
from turing.experimental.demo.world_model import DynamicEnvironmentModel
from turing.experimental.demo.engine_wrapper import TuringAcceleratedGenerator


PLAN_LINE = ("\nEnd your answer with one line exactly: NODES=<integer>; ZONES=<comma-separated zone ids, "
             "e.g. AWS_US_EAST_1>")


class MultiAgentCoordinator:
    """
    Multi-Agent Deliberation Network:
    • Proposal Agent (Generative Sibling): Generates initial candidate configurations.
    • World Model Verification Gate: Evaluates candidate against live environment state (F-Score).
    • Revision Optimizer (Self-Revision Sibling): Iteratively refines parameters to minimize Constraint Penalty to 0.
    """
    def __init__(
        self,
        engine: TuringAcceleratedGenerator,
        world_model: Optional[DynamicEnvironmentModel] = None
    ):
        self.engine = engine
        self.world_model = world_model or DynamicEnvironmentModel()

        # Define Agent Personas
        self.planner_system = (
            "You are PlannerAgent, an expert distributed systems architect. "
            "Your task is to outline creative, distributed multi-region cloud infrastructures."
        )
        self.optimizer_system = (
            "You are OptimizerAgent, a self-revision optimization agent. "
            "Your goal is to eliminate constraint penalties by mathematically rewriting system topologies, "
            "expanding node capacity to match active demand, and bypassing failed nodes."
        )
        
        # optimizer_system drives the self-revision agent
        

    def recirculate_agent_beliefs(
        self,
        belief_states: Any,
        softening_sq: float = 1e-4,
        step_size: float = 0.05
    ) -> Any:
        """
        N-Body Multi-Agent Belief State Recirculation (Spatial HPC Stencil Engine).
        """
        try:
            import turing.turing_csrc as turing_csrc
            import numpy as np
            import torch
            if isinstance(belief_states, torch.Tensor):
                states_np = belief_states.detach().to(torch.float32).cpu().contiguous().numpy()
                out_np = turing_csrc.nbody_belief_recirculate(states_np, softening_sq, step_size)
                return torch.from_numpy(out_np).to(device=belief_states.device, dtype=belief_states.dtype)
            elif isinstance(belief_states, np.ndarray):
                return turing_csrc.nbody_belief_recirculate(belief_states.astype(np.float32), softening_sq, step_size)
        except ImportError:
            pass
        return belief_states

    def _score(self, text: str) -> Dict[str, Any]:
        m = re.search(r"NODES\s*=\s*([\d,]+)\s*;\s*ZONES\s*=\s*([A-Z0-9_, ]+)", text)
        if not m:
            return {"parsed": False, "constraint_penalty": None, "revision_needed": None}
        zones = [z.strip() for z in m.group(2).split(",") if z.strip()]
        feedback = self.world_model.evaluate_constraint_penalty(int(m.group(1).replace(",", "")), 8000, zones)
        return {"parsed": True, **feedback}

    def run_multi_agent_workflow(
        self,
        user_scenario: str = "Draft a multi-region cloud architecture for 10M users over an 8,000 km network loop.",
        force_outage: Optional[bool] = None,
        max_tokens_per_agent: int = 100
    ) -> Dict[str, Any]:
        """
        Runs the planner and optimizer agents through the generator. Each agent is asked to end with a line
        "NODES=<integer>; ZONES=<zone ids>"; the world model scores that plan (over the scenario's 8,000 km loop).
        An answer without the line is not scored: its constraint_penalty is None and so is revision_successful.
        """
        # Step 1: Update Environment Model Telemetry
        env_alert = self.world_model.update_hidden_state_from_pixels_or_logs(force_alert=force_outage)

        # Step 2: Proposal Agent generates initial candidate proposal
        sol_prompt = (
            f"{user_scenario}\n"
            "Specify initial node counts, geographic zones, and routing strategy." + PLAN_LINE
        )
        sol_res = self.engine.fast_generate(
            system_prompt=self.planner_system,
            user_prompt=sol_prompt,
            max_new_tokens=max_tokens_per_agent,
        )

        # Step 3: Evaluate Constraint Penalty of the proposal the planner wrote
        world_feedback_initial = self._score(sol_res["response_text"])

        feedback_json = json.dumps(world_feedback_initial, indent=2)

        # Step 4: Revision Optimizer processes feedback and executes Self-Revision Loop
        optimizer_prompt = (
            f"Baseline Proposal from Proposal Agent:\n{sol_res['response_text']}\n\n"
            f"Environment State World Model Feedback:\n{feedback_json}\n\n"
            "Execution Instruction:\n"
            "If 'revision_needed' is true, execute a Self-Revision Loop:\n"
            "1. Expand total active nodes to at least 3,000 to fully cover 15,000,000 user demand.\n"
            "2. Explicitly bypass dead zone 'AWS_EU_WEST_NODE_4' to eliminate outage penalties.\n"
            "3. Output the final, mathematically verified architecture configuration with zero constraint penalty."
            + PLAN_LINE
        )
        optimizer_res = self.engine.fast_generate(
            system_prompt=self.optimizer_system,
            user_prompt=optimizer_prompt,
            max_new_tokens=max_tokens_per_agent + 40,
        )


        # Step 5: Post-Revision Verification of the plan the optimizer wrote
        world_feedback_final = self._score(optimizer_res["response_text"])

        total_overhead_ms = sol_res["latency_ms"] + optimizer_res["latency_ms"]

        return {
            "environment_state": env_alert,
            "sol_proposal": sol_res,
            "initial_world_feedback": world_feedback_initial,
            "optimizer_revision": optimizer_res,
            "final_world_feedback": world_feedback_final,
            "total_overhead_ms": round(total_overhead_ms, 2),
            "constraint_penalty_reduction": f"{world_feedback_initial['constraint_penalty']} -> {world_feedback_final['constraint_penalty']}",
            "revision_successful": (None if world_feedback_final["constraint_penalty"] is None
                                    else world_feedback_final["constraint_penalty"] == 0)
        }
