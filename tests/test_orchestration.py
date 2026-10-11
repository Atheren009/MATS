"""
tests/test_orchestration.py
============================
Unit tests for Phase C orchestration components.

Tests are self-contained (no DB on disk, no real market data needed).
Uses tmp_path fixtures for SQLite isolation.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# correlation_gate tests
# ---------------------------------------------------------------------------

from ml_system.orchestration.correlation_gate import (
    correlation_gate,
    CORRELATION_THRESHOLD,
)


def _make_returns(n=100, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-02", periods=n, freq="D")
    return pd.Series(rng.normal(0, 1, n), index=idx)


class TestCorrelationGate:

    def test_passes_when_no_funded_agents(self):
        passed, pair, corr = correlation_gate(_make_returns(), funded_returns={})
        assert passed is True
        assert pair is None
        assert corr is None

    def test_passes_below_threshold(self):
        # Independent returns -> low correlation
        cand   = _make_returns(seed=0)
        funded = _make_returns(seed=99)
        passed, pair, corr = correlation_gate(cand, {"agent_A": funded})
        # Independent series should have |r| well below 0.5
        assert corr is not None
        if corr < CORRELATION_THRESHOLD:
            assert passed is True

    def test_fails_above_threshold(self):
        # Perfectly correlated returns -> r=1.0 -> gate must fail
        base   = _make_returns(seed=7)
        cand   = base.copy()
        funded = base.copy()
        passed, pair, corr = correlation_gate(cand, {"agent_A": funded})
        assert passed is False
        assert corr >= CORRELATION_THRESHOLD

    def test_worst_pair_reported(self):
        cand = _make_returns(seed=0)
        # Two funded agents: one correlated, one not
        corr_agent  = cand.copy()
        indep_agent = _make_returns(seed=77)
        passed, pair, corr = correlation_gate(
            cand, {"corr_one": corr_agent, "indep_one": indep_agent}
        )
        assert pair == "corr_one"
        assert not passed

    def test_insufficient_overlap_skips_pair(self):
        cand   = _make_returns(n=100, seed=0)
        # funded series on completely different dates -> no overlap
        idx_other = pd.date_range("2030-01-01", periods=100, freq="D")
        funded = pd.Series(np.ones(100), index=idx_other)
        # Should pass because the pair is skipped (insufficient overlap)
        passed, pair, corr = correlation_gate(cand, {"no_overlap": funded})
        assert passed is True


# ---------------------------------------------------------------------------
# CapitalAllocator tests
# ---------------------------------------------------------------------------

from ml_system.orchestration.capital_allocator import (
    CapitalAllocator,
    DWELL_WINDOW,
    MAX_FUNDED_AGENTS,
)


class TestCapitalAllocator:

    def test_weights_sum_to_one_single_agent(self):
        alloc = CapitalAllocator(["A"])
        weights = alloc.update("trending", bar_index=0)
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_weights_sum_to_one_two_agents(self):
        alloc = CapitalAllocator(["A", "B"])
        weights = alloc.update("trending", bar_index=0)
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_realloc_fires_after_dwell_window(self):
        alloc = CapitalAllocator(
            ["A", "B"],
            agent_regime_prefs={"A": "trending", "B": "ranging"},
        )
        weights_before = alloc.weights.copy()
        # Feed DWELL_WINDOW bars of "trending"
        for i in range(DWELL_WINDOW):
            weights = alloc.update("trending", bar_index=i)
        # After dwell window, A (trending agent) should get full weight
        assert weights["A"] == pytest.approx(1.0)
        assert weights["B"] == pytest.approx(0.0)

    def test_no_realloc_before_dwell_window(self):
        alloc = CapitalAllocator(
            ["A", "B"],
            agent_regime_prefs={"A": "trending", "B": "ranging"},
        )
        # Feed fewer bars than dwell window
        for i in range(DWELL_WINDOW - 1):
            weights = alloc.update("trending", bar_index=i)
        # Weights should still be equal (no realloc yet)
        assert weights["A"] == pytest.approx(0.5)
        assert weights["B"] == pytest.approx(0.5)

    def test_equal_weights_when_no_regime_match(self):
        alloc = CapitalAllocator(
            ["A", "B"],
            agent_regime_prefs={"A": "trending", "B": "ranging"},
        )
        for i in range(DWELL_WINDOW):
            weights = alloc.update("high_vol_breakout", bar_index=i)
        # No agent prefers high_vol_breakout -> equal split
        assert weights["A"] == pytest.approx(0.5)
        assert weights["B"] == pytest.approx(0.5)

    def test_max_agents_enforced(self):
        with pytest.raises(ValueError, match="MAX_FUNDED_AGENTS"):
            CapitalAllocator(["A", "B", "C"])

    def test_no_realloc_on_unstable_regime(self):
        alloc = CapitalAllocator(
            ["A", "B"],
            agent_regime_prefs={"A": "trending", "B": "ranging"},
        )
        # Alternate regime every bar -- window never stabilises
        for i in range(DWELL_WINDOW * 3):
            regime = "trending" if i % 2 == 0 else "ranging"
            weights = alloc.update(regime, bar_index=i)
        # Weights should still be equal (never stabilised)
        assert weights["A"] == pytest.approx(0.5)
        assert weights["B"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# AgentRegistry tests
# ---------------------------------------------------------------------------

from ml_system.orchestration.agent_registry import AgentRegistry


class TestAgentRegistry:

    def test_register_and_retrieve(self, tmp_path):
        reg = AgentRegistry(db_path=tmp_path / "test.db")
        aid = reg.register_agent("DIA", agent_id="DIA_v1")
        assert aid == "DIA_v1"
        agent = reg.get_agent("DIA_v1")
        assert agent["symbol"] == "DIA"
        assert agent["lifecycle_status"] == "pending"

    def test_global_trial_count_increments(self, tmp_path):
        reg = AgentRegistry(db_path=tmp_path / "test.db")
        assert reg.get_trial_count() == 0
        reg.register_agent("DIA", agent_id="DIA_v1")
        n1 = reg.update_backtest_result("DIA_v1", dsr=1.2, spa_p=0.03)
        assert n1 == 1
        assert reg.get_trial_count() == 1

        reg.register_agent("QQQ", agent_id="QQQ_v1")
        n2 = reg.update_backtest_result("QQQ_v1", dsr=0.8, spa_p=0.07)
        assert n2 == 2  # cumulative, never reset
        assert reg.get_trial_count() == 2

    def test_reject_agent(self, tmp_path):
        reg = AgentRegistry(db_path=tmp_path / "test.db")
        reg.register_agent("DIA", agent_id="DIA_v1")
        reg.reject_agent("DIA_v1", reason="DSR <= 0")
        agent = reg.get_agent("DIA_v1")
        assert agent["lifecycle_status"] == "rejected"
        assert "DSR" in agent["rejected_reason"]

    def test_fund_agent(self, tmp_path):
        reg = AgentRegistry(db_path=tmp_path / "test.db")
        reg.register_agent("DIA", agent_id="DIA_v1")
        reg.update_backtest_result("DIA_v1", dsr=1.2, spa_p=0.03)
        reg.set_paper_live("DIA_v1")
        reg.fund_agent("DIA_v1", correlation_at_fund=0.12)
        agent = reg.get_agent("DIA_v1")
        assert agent["lifecycle_status"] == "funded"
        assert agent["correlation_at_fund"] == pytest.approx(0.12)
        funded = reg.funded_agents()
        assert len(funded) == 1


# ---------------------------------------------------------------------------
# Orchestrator integration tests
# ---------------------------------------------------------------------------

from ml_system.orchestration.orchestrator import Orchestrator


class TestOrchestrator:

    def test_kill_switch_blocks_allocation(self, tmp_path):
        orch = Orchestrator(total_capital=10_000, db_path=tmp_path / "orch.db")
        orch.trip_kill_switch("test drawdown breach")
        assert orch.is_kill_switch_tripped
        weights = orch.on_bar("trending", bar_index=0)
        assert weights == {}

    def test_agent_lifecycle_pass(self, tmp_path):
        orch = Orchestrator(total_capital=10_000, db_path=tmp_path / "orch.db")
        aid  = orch.register_agent("DIA", agent_id="DIA_v1")

        oos = _make_returns(seed=0)
        # DSR > 0, SPA < 0.10 -> passes backtest gate
        passed_bt = orch.submit_backtest(aid, dsr=1.2, spa_p=0.03, oos_returns=oos)
        assert passed_bt is True

        # Paper-live with no existing funded agents -> passes correlation gate
        paper = _make_returns(seed=1)
        funded = orch.submit_paper_live_result(aid, paper_returns=paper)
        assert funded is True

        # Should now have an allocator
        weights = orch.on_bar("trending", bar_index=0)
        assert "DIA_v1" in weights

    def test_agent_rejected_bad_dsr(self, tmp_path):
        orch = Orchestrator(total_capital=10_000, db_path=tmp_path / "orch.db")
        aid  = orch.register_agent("DIA", agent_id="DIA_v1")
        oos  = _make_returns(seed=0)
        passed = orch.submit_backtest(aid, dsr=-0.5, spa_p=0.03, oos_returns=oos)
        assert passed is False
        agent = orch._registry.get_agent(aid)
        assert agent["lifecycle_status"] == "rejected"

    def test_agent_rejected_high_correlation(self, tmp_path):
        orch = Orchestrator(total_capital=10_000, db_path=tmp_path / "orch.db")

        # Fund first agent
        aid1 = orch.register_agent("DIA", agent_id="DIA_v1")
        base = _make_returns(seed=42, n=200)
        orch.submit_backtest(aid1, dsr=1.5, spa_p=0.02, oos_returns=base)
        orch.submit_paper_live_result(aid1, paper_returns=base)
        assert len(orch._registry.funded_agents()) == 1

        # Fund second agent with perfectly correlated returns -> should be rejected
        aid2 = orch.register_agent("QQQ", agent_id="QQQ_v1")
        orch.submit_backtest(aid2, dsr=1.2, spa_p=0.04, oos_returns=base)
        funded = orch.submit_paper_live_result(aid2, paper_returns=base)
        assert funded is False
        agent2 = orch._registry.get_agent(aid2)
        assert agent2["lifecycle_status"] == "rejected"
        assert "Correlation" in agent2["rejected_reason"]
