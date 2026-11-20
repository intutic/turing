"""
Dynamic Environment Constraint Evaluator & Environment State State Estimation.
Tracks environmental beliefs, prior preferences, and computes Constraint Penalty.
"""

import json
import random
from typing import Dict, Any, List, Optional


class DynamicEnvironmentModel:
    """
    Dynamic Environment Telemetry & Constraint Model:
    Tracks hidden environmental states, evaluates performance preferences,
    and computes constraint penalty scores to guide and verify agent decisions.
    """
    def __init__(
        self,
        target_latency_ms: float = 100.0,
        max_load_per_node: int = 5000,
        cost_limit_usd: float = 150000.0
    ):
        self.prior_preferences = {
            "target_latency_ms": target_latency_ms,
            "max_load_per_node": max_load_per_node,
            "cost_limit_usd": cost_limit_usd
        }
        self.hidden_environment_state = {
            "active_outages": [],
            "regional_demand_spike": False,
            "observation_history": []
        }

    def update_hidden_state_from_pixels_or_logs(self, force_alert: Optional[bool] = None) -> Dict[str, Any]:
        """
        Updates belief state Q(s) from simulated environment telemetry / logs.
        """
        trigger = force_alert if force_alert is not None else (random.random() > 0.3)
        if trigger:
            self.hidden_environment_state["active_outages"] = ["AWS_EU_WEST_NODE_4"]
            self.hidden_environment_state["regional_demand_spike"] = True
            alert_msg = "⚠️ [Environment State Alert] World Model detected a hidden State Shift: EU_NODE_4 Failure + Demand Spike (15M Active Users)!"
        else:
            self.hidden_environment_state["active_outages"] = []
            self.hidden_environment_state["regional_demand_spike"] = False
            alert_msg = "ℹ️ [Environment Telemetry State] Telemetry Nominal: Standard load distribution (10M Active Users)."

        self.hidden_environment_state["observation_history"].append(alert_msg)
        return {
            "alert": alert_msg,
            "active_outages": list(self.hidden_environment_state["active_outages"]),
            "regional_demand_spike": self.hidden_environment_state["regional_demand_spike"]
        }

    def evaluate_constraint_penalty(
        self,
        proposed_nodes: int,
        proposed_distance_km: float,
        routed_nodes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Computes Constraint Penalty F = Complexity - Accuracy + Penalties.
        A score > 0 signals variational surprise / constraint violation, requiring self-revision.
        """
        actual_latency = (proposed_distance_km * 0.005) + 15.5
        capacity = proposed_nodes * self.prior_preferences["max_load_per_node"]
        demand = 15000000 if self.hidden_environment_state["regional_demand_spike"] else 10000000

        constraint_penalty = 0
        violations = []

        # 1. Latency Constraint Penalty
        if actual_latency > self.prior_preferences["target_latency_ms"]:
            penalty = 500
            constraint_penalty += penalty
            violations.append(f"Latency ({actual_latency:.1f}ms) exceeds target ({self.prior_preferences['target_latency_ms']}ms): +{penalty} F")

        # 2. System Capacity Overload Penalty
        if capacity < demand:
            penalty = 1000
            constraint_penalty += penalty
            violations.append(f"Capacity ({capacity:,} users) < Active Demand ({demand:,} users): +{penalty} F (System Crash Risk)")

        # 3. Dead Zone Routing Penalty
        active_outages = self.hidden_environment_state["active_outages"]
        if "AWS_EU_WEST_NODE_4" in active_outages:
            if routed_nodes is None or "AWS_EU_WEST_NODE_4" in routed_nodes:
                penalty = 300
                constraint_penalty += penalty
                violations.append(f"Traffic routed through failed cluster (AWS_EU_WEST_NODE_4): +{penalty} F")

        return {
            "constraint_penalty": constraint_penalty,
            "inferred_latency_ms": round(actual_latency, 2),
            "system_capacity": capacity,
            "active_demand": demand,
            "active_outages": active_outages,
            "violations": violations,
            "revision_needed": constraint_penalty > 0
        }

