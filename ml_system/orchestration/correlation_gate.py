"""
correlation_gate -- Pairwise OOS-return correlation guard.

A candidate agent must have Pearson correlation < CORRELATION_THRESHOLD
against EVERY currently-funded agent before it can receive capital.
This is pre-registered and never tuned after seeing results.

Usage
-----
    from ml_system.orchestration.correlation_gate import correlation_gate

    passed, worst_pair, worst_corr = correlation_gate(
        candidate_returns=candidate_series,
        funded_returns={"DIA_v1": dia_series, "QQQ_v1": qqq_series},
    )
    if not passed:
        registry.reject_agent(agent_id, f"Corr {worst_corr:.3f} >= {CORRELATION_THRESHOLD} vs {worst_pair}")
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# Pre-registered threshold -- never tuned after seeing results.
CORRELATION_THRESHOLD: float = 0.5


def correlation_gate(
    candidate_returns: pd.Series,
    funded_returns: dict[str, pd.Series],
    threshold: float = CORRELATION_THRESHOLD,
) -> tuple[bool, Optional[str], Optional[float]]:
    """
    Compute pairwise Pearson correlation between a candidate agent's OOS
    return series and every currently-funded agent.

    Parameters
    ----------
    candidate_returns : pd.Series
        Daily (or bar-level) OOS PnL series for the candidate agent.
    funded_returns : dict[str, pd.Series]
        Mapping agent_id -> OOS return series for every funded agent.
    threshold : float
        Any pair >= threshold fails. Default = CORRELATION_THRESHOLD.

    Returns
    -------
    passed : bool
        True if all pairwise correlations are strictly below threshold.
    worst_pair : str or None
        agent_id of the highest-corr funded agent, or None if no funded agents.
    worst_corr : float or None
        Correlation value of that worst pair, or None if no funded agents.
    """
    if not funded_returns:
        # No funded agents -- gate always passes
        log.info("Correlation gate: no funded agents, candidate passes by default.")
        return True, None, None

    worst_pair: Optional[str] = None
    worst_corr: Optional[float] = None

    for agent_id, funded_ser in funded_returns.items():
        # Align on the intersection of dates before computing correlation
        aligned = pd.concat(
            [candidate_returns.rename("cand"), funded_ser.rename("funded")],
            axis=1,
        ).dropna()

        if len(aligned) < 10:
            log.warning(
                "Correlation gate: only %d overlapping bars vs %s; skipping pair.",
                len(aligned), agent_id,
            )
            continue

        corr = float(aligned["cand"].corr(aligned["funded"]))
        log.info("  Correlation vs %s: %.3f (threshold %.2f)", agent_id, corr, threshold)

        if worst_corr is None or corr > worst_corr:
            worst_corr = corr
            worst_pair = agent_id

    if worst_corr is None:
        # All pairs had insufficient overlap -- treat as pass with a warning
        log.warning("Correlation gate: all pairs had insufficient overlap; passing with caution.")
        return True, None, None

    passed = worst_corr < threshold
    if not passed:
        log.warning(
            "Correlation gate FAILED: corr %.3f >= %.2f vs agent %s",
            worst_corr, threshold, worst_pair,
        )
    else:
        log.info(
            "Correlation gate PASSED: worst corr %.3f < %.2f (vs %s)",
            worst_corr, threshold, worst_pair,
        )

    return passed, worst_pair, worst_corr
