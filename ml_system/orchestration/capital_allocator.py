"""
CapitalAllocator -- regime-gated, dwell-windowed weight assignment.

Rules
-----
1. Only funded agents receive any weight.
2. A regime must hold for DWELL_WINDOW consecutive bars before any
   reallocation fires.  No reallocation on every regime-probability tick.
3. Maximum concurrently-funded agents is MAX_FUNDED_AGENTS (config value).
   Raise it only after the 2nd agent has a clean paper-live run.
4. Weights are equal-weight across funded agents whose preferred regime
   matches the current regime.  If no funded agent matches, weights are
   equal-weight across all funded agents (don't go flat).
5. Weights always sum to 1.0.

Usage
-----
    alloc = CapitalAllocator(funded_agent_ids=["DIA_v1", "QQQ_v1"])
    weights = alloc.update(current_regime="trending", bar_index=42)
    # -> {"DIA_v1": 0.5, "QQQ_v1": 0.5}
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Optional

log = logging.getLogger(__name__)

# Pre-registered config values.  Raise MAX_FUNDED_AGENTS only after the
# 2nd agent has its own clean paper-live run -- not speculatively.
DWELL_WINDOW: int = 8          # bars the regime must hold before realloc
MAX_FUNDED_AGENTS: int = 2


class CapitalAllocator:
    """
    Regime-gated capital allocator for up to MAX_FUNDED_AGENTS funded agents.

    Parameters
    ----------
    funded_agent_ids : list[str]
        IDs of currently-funded agents (already passed DSR+SPA+corr gates).
    agent_regime_prefs : dict[str, str], optional
        Maps agent_id -> preferred regime label.  Agents with a matching
        regime get full allocation; others get equal-share of the remainder.
        If omitted, all funded agents share equally regardless of regime.
    """

    def __init__(
        self,
        funded_agent_ids: list[str],
        agent_regime_prefs: Optional[dict[str, str]] = None,
    ) -> None:
        if len(funded_agent_ids) > MAX_FUNDED_AGENTS:
            raise ValueError(
                f"Cannot fund {len(funded_agent_ids)} agents; "
                f"MAX_FUNDED_AGENTS={MAX_FUNDED_AGENTS}. "
                "Raise the limit only after the next agent completes paper-live."
            )

        self._agents   = list(funded_agent_ids)
        self._prefs    = agent_regime_prefs or {}
        self._window: deque[str] = deque(maxlen=DWELL_WINDOW)
        self._current_weights: dict[str, float] = self._equal_weights()
        self._last_realloc_regime: Optional[str] = None

    def update(self, current_regime: str, bar_index: int) -> dict[str, float]:
        """
        Push current_regime into the dwell window.  If the window is full
        and all values are the same, fire a reallocation.  Otherwise keep
        the last weights unchanged.

        Parameters
        ----------
        current_regime : str
            HMM regime label for the current bar.
        bar_index : int
            Global bar counter (for logging only).

        Returns
        -------
        dict[str, float]
            Mapping agent_id -> capital weight (sums to 1.0).
        """
        if not self._agents:
            return {}

        self._window.append(current_regime)

        # Only reallocate when the window is full and regime has been stable
        if len(self._window) < DWELL_WINDOW:
            return dict(self._current_weights)

        stable_regime = self._window[0]
        if not all(r == stable_regime for r in self._window):
            return dict(self._current_weights)  # regime not yet stable

        if stable_regime == self._last_realloc_regime:
            return dict(self._current_weights)  # already allocated to this regime

        # Regime has held for DWELL_WINDOW bars -- fire reallocation
        new_weights = self._compute_weights(stable_regime)
        log.info(
            "[bar %d] Realloc fired: regime '%s' held for %d bars -> weights %s",
            bar_index, stable_regime, DWELL_WINDOW, new_weights,
        )
        self._current_weights   = new_weights
        self._last_realloc_regime = stable_regime
        return dict(self._current_weights)

    def _compute_weights(self, regime: str) -> dict[str, float]:
        """
        Agents whose preferred regime matches get priority (full allocation).
        If no match, equal-weight all funded agents -- never go flat.
        """
        matching = [a for a in self._agents if self._prefs.get(a) == regime]

        if matching:
            w = 1.0 / len(matching)
            weights = {a: (w if a in matching else 0.0) for a in self._agents}
        else:
            weights = self._equal_weights()

        return weights

    def _equal_weights(self) -> dict[str, float]:
        if not self._agents:
            return {}
        w = 1.0 / len(self._agents)
        return {a: w for a in self._agents}

    @property
    def weights(self) -> dict[str, float]:
        """Current allocation without triggering an update."""
        return dict(self._current_weights)

    def add_agent(self, agent_id: str, regime_pref: Optional[str] = None) -> None:
        """Fund a new agent (only call after it has passed all gates)."""
        if len(self._agents) >= MAX_FUNDED_AGENTS:
            raise ValueError(
                f"Already at MAX_FUNDED_AGENTS={MAX_FUNDED_AGENTS}. "
                "Cannot add more without raising the limit."
            )
        self._agents.append(agent_id)
        if regime_pref:
            self._prefs[agent_id] = regime_pref
        self._current_weights = self._equal_weights()
        log.info("CapitalAllocator: added agent %s (pref=%s)", agent_id, regime_pref)

    def remove_agent(self, agent_id: str) -> None:
        """Defund an agent (e.g. after kill-switch trip)."""
        if agent_id in self._agents:
            self._agents.remove(agent_id)
            self._prefs.pop(agent_id, None)
            self._current_weights = self._equal_weights()
            log.warning("CapitalAllocator: removed agent %s", agent_id)
