"""
Orchestrator -- shared harness for all funded agents.

Enforces the agent lifecycle:
    pending -> backtested -> paper_live -> funded -> (running | rejected)

One Orchestrator instance, one kill-switch, one OrderRouter, one RiskManager.

Usage (illustrative)
--------------------
    orch = Orchestrator()
    agent_id = orch.register_agent("DIA")

    # After walk-forward backtest:
    orch.submit_backtest(agent_id, dsr=1.2, spa_p=0.04,
                         oos_returns=dia_returns, oos_csv_path="results/dia_oos.csv")

    # After paper-live phase (called by the live trading harness):
    orch.submit_paper_live_result(agent_id, paper_returns=dia_paper_series)

    # On every new 15-min bar:
    weights = orch.on_bar(current_regime="trending", bar_index=t)
    for agent_id, weight in weights.items():
        allocated_capital = orch.total_capital * weight
        ...
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import pandas as pd

from .agent_registry    import AgentRegistry
from .correlation_gate  import correlation_gate, CORRELATION_THRESHOLD
from .capital_allocator import CapitalAllocator, MAX_FUNDED_AGENTS

log = logging.getLogger(__name__)

# DSR/SPA gates for promotion
DSR_THRESHOLD  = 0.0   # Deflated Sharpe must be positive
SPA_THRESHOLD  = 0.10  # Block-bootstrap SPA p-value must be < 10%


class Orchestrator:
    """
    Central harness for multi-agent trading lifecycle management.

    One instance per trading session.  Not thread-safe for concurrent
    agent lifecycle transitions; use the shared lock if needed.

    Parameters
    ----------
    total_capital : float
        Total capital pool available for allocation across funded agents.
    db_path : Path, optional
        Override for the SQLite DB used by AgentRegistry.
    """

    def __init__(
        self,
        total_capital: float = 10_000.0,
        db_path: Optional[Path] = None,
    ) -> None:
        self.total_capital   = total_capital
        self._registry       = AgentRegistry(db_path=db_path)
        self._allocator: Optional[CapitalAllocator] = None

        # Shared kill-switch -- any agent tripping it halts the whole harness
        self._kill_switch   = False
        self._kill_reason   = ""
        self._lock          = threading.Lock()

        # OOS return series for funded agents (agent_id -> Series)
        self._funded_returns: dict[str, pd.Series] = {}

        # Rebuild allocator from any already-funded agents in the registry
        self._rebuild_allocator()

        log.info(
            "Orchestrator ready | capital=$%.2f | funded=%d",
            total_capital, len(self._funded_returns),
        )

    # ------------------------------------------------------------------
    # Kill-switch (shared across all agents)
    # ------------------------------------------------------------------

    @property
    def is_kill_switch_tripped(self) -> bool:
        return self._kill_switch

    def trip_kill_switch(self, reason: str) -> None:
        """
        Halt all new orders for all agents.  Irreversible for this session.
        Call from any agent when its own drawdown limit is breached.
        """
        with self._lock:
            if not self._kill_switch:
                self._kill_switch = True
                self._kill_reason = reason
                log.critical("ORCHESTRATOR KILL SWITCH TRIPPED: %s", reason)

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------

    def register_agent(
        self,
        symbol: str,
        agent_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> str:
        """Register a new agent in pending state.  Returns agent_id."""
        return self._registry.register_agent(symbol, agent_id=agent_id, notes=notes)

    def submit_backtest(
        self,
        agent_id: str,
        dsr: float,
        spa_p: float,
        oos_returns: pd.Series,
        oos_csv_path: Optional[str] = None,
    ) -> bool:
        """
        Record backtest results and increment global_trial_count.
        Apply DSR/SPA gate; reject if failed.

        Returns True if the agent passes the gate (can proceed to paper-live).
        """
        n_trials = self._registry.update_backtest_result(
            agent_id,
            dsr=dsr,
            spa_p=spa_p,
            oos_return_series=oos_csv_path,
        )

        if dsr <= DSR_THRESHOLD:
            reason = f"DSR {dsr:.4f} <= {DSR_THRESHOLD} (n_trials={n_trials})"
            self._registry.reject_agent(agent_id, reason)
            return False

        if spa_p >= SPA_THRESHOLD:
            reason = f"SPA p={spa_p:.4f} >= {SPA_THRESHOLD} (n_trials={n_trials})"
            self._registry.reject_agent(agent_id, reason)
            return False

        log.info(
            "Agent %s PASSED backtest gate: DSR=%.4f SPA_p=%.4f (n_trials=%d)",
            agent_id, dsr, spa_p, n_trials,
        )
        return True

    def submit_paper_live_result(
        self,
        agent_id: str,
        paper_returns: pd.Series,
    ) -> bool:
        """
        Submit paper-live return series for correlation check.
        Fund the agent if it passes; reject otherwise.

        Returns True if funded.
        """
        if self.is_kill_switch_tripped:
            log.warning("Kill switch is tripped; cannot fund new agent %s.", agent_id)
            return False

        funded = self._registry.funded_agents()
        if len(funded) >= MAX_FUNDED_AGENTS:
            reason = (
                f"MAX_FUNDED_AGENTS={MAX_FUNDED_AGENTS} already reached. "
                "Raise the limit after the next agent completes paper-live."
            )
            self._registry.reject_agent(agent_id, reason)
            return False

        self._registry.set_paper_live(agent_id)

        # Correlation gate against funded agents
        passed, worst_pair, worst_corr = correlation_gate(
            candidate_returns=paper_returns,
            funded_returns=self._funded_returns,
        )

        if not passed:
            reason = (
                f"Correlation {worst_corr:.3f} >= {CORRELATION_THRESHOLD} vs {worst_pair}"
            )
            self._registry.reject_agent(agent_id, reason)
            return False

        # Fund the agent
        self._registry.fund_agent(agent_id, correlation_at_fund=worst_corr or 0.0)
        self._funded_returns[agent_id] = paper_returns
        self._rebuild_allocator()
        log.info("Agent %s is now FUNDED.  Total funded: %d", agent_id, len(self._funded_returns))
        return True

    # ------------------------------------------------------------------
    # Bar-by-bar capital allocation
    # ------------------------------------------------------------------

    def on_bar(self, current_regime: str, bar_index: int) -> dict[str, float]:
        """
        Called once per completed 15-min bar.  Returns capital weights.
        Returns an empty dict if no agents are funded or kill-switch is tripped.
        """
        if self.is_kill_switch_tripped:
            return {}

        if self._allocator is None:
            return {}

        return self._allocator.update(current_regime, bar_index)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_allocator(self) -> None:
        """Reconstruct the CapitalAllocator from the current funded set."""
        funded = self._registry.funded_agents()
        ids = [a["agent_id"] for a in funded]

        if not ids:
            self._allocator = None
            return

        # Cap to MAX_FUNDED_AGENTS (oldest funded first by funded_utc)
        funded_sorted = sorted(funded, key=lambda a: a.get("funded_utc") or "")
        ids = [a["agent_id"] for a in funded_sorted[:MAX_FUNDED_AGENTS]]

        self._allocator = CapitalAllocator(funded_agent_ids=ids)
        log.info("CapitalAllocator rebuilt with agents: %s", ids)
