"""
ml_system/orchestration -- Multi-agent lifecycle management.

Modules
-------
agent_registry   : Persistent SQLite registry for all agents, extending
                   retrain_log with agent_id, OOS return series reference,
                   correlation at funding, and a global_trial_count that
                   never resets per agent.
correlation_gate : Pairwise Pearson correlation guard against funded agents.
capital_allocator: Regime-gated, dwell-windowed weight assignment.
orchestrator     : Shared kill-switch, shared OrderRouter/RiskManager, and
                   the agent lifecycle state machine.
"""
from __future__ import annotations

from .agent_registry    import AgentRegistry
from .correlation_gate  import correlation_gate, CORRELATION_THRESHOLD
from .capital_allocator import CapitalAllocator, DWELL_WINDOW, MAX_FUNDED_AGENTS
from .orchestrator      import Orchestrator

__all__ = [
    "AgentRegistry",
    "correlation_gate",
    "CORRELATION_THRESHOLD",
    "CapitalAllocator",
    "DWELL_WINDOW",
    "MAX_FUNDED_AGENTS",
    "Orchestrator",
]
