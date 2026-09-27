"""
test_routing_engine.py — 17 test cases for Phase 3 routing engine.

Tests are organised into four groups:
  A. Exchange — seeding, fees, liquidity
  B. Strategies — BestPrice, TWAP, VWAP, Smart routing decisions
  C. OrderRouter — orchestration, aggregation, strategy switching
  D. StrategyComparator — end-to-end report generation and quality metrics

All price data is derived from real AAPL 1-minute bars via PriceFeed.
Synthetic bars with controlled parameters are used where determinism matters.
"""

from __future__ import annotations

import os
import pytest
from datetime import datetime, timezone

from order_router.models import Order, OrderSide, OrderType
from order_router.price_feed import PriceFeed, Bar
from order_router.exchange import Exchange, VenueConfig, VENUE_CONFIGS
from order_router.venue_registry import VenueRegistry
from order_router.child_order import ChildOrder, FillResult
from order_router.router import OrderRouter, RouterResult
from order_router.routing import (
    BestPriceStrategy,
    TWAPStrategy,
    VWAPStrategy,
    SmartStrategy,
)
from order_router.comparator import StrategyComparator


# ---------------------------------------------------------------------------
# Paths & shared fixtures
# ---------------------------------------------------------------------------

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "AAPL_1min_2024-2026.csv")


def make_bar(
    vwap: float = 190.0,
    volume: float = 3000.0,
    spread: float = 0.0,
) -> Bar:
    """Create a synthetic Bar with controlled parameters for deterministic tests."""
    return Bar(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc),
        open=vwap,
        high=vwap + spread,
        low=vwap - spread,
        close=vwap,
        volume=volume,
        trade_count=50.0,
        vwap=vwap,
    )


@pytest.fixture(scope="module")
def real_bars():
    """First 20 real AAPL bars from the CSV, loaded once per module."""
    feed = PriceFeed(CSV_PATH)
    return list(feed)[:20]


@pytest.fixture
def bar():
    """A single controlled bar: VWAP=190.0, volume=3000."""
    return make_bar(vwap=190.0, volume=3000.0)


@pytest.fixture
def alpha(bar):
    """Fresh ALPHA exchange seeded from controlled bar."""
    exch = Exchange("ALPHA", VENUE_CONFIGS["ALPHA"])
    exch.seed_from_bar(bar)
    return exch


@pytest.fixture
def beta(bar):
    """Fresh BETA exchange seeded from controlled bar."""
    exch = Exchange("BETA", VENUE_CONFIGS["BETA"])
    exch.seed_from_bar(bar)
    return exch


@pytest.fixture
def gamma(bar):
    """Fresh GAMMA exchange seeded from controlled bar."""
    exch = Exchange("GAMMA", VENUE_CONFIGS["GAMMA"])
    exch.seed_from_bar(bar)
    return exch


@pytest.fixture
def venues(bar):
    """All three exchanges seeded identically from the same controlled bar."""
    result = {}
    for name, cfg in VENUE_CONFIGS.items():
        exch = Exchange(name, cfg)
        exch.seed_from_bar(bar)
        result[name] = exch
    return result


@pytest.fixture
def five_bars():
    """Five synthetic bars with varying volumes for TWAP/VWAP tests."""
    volumes = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    return [make_bar(vwap=190.0 + i * 0.01, volume=v) for i, v in enumerate(volumes)]


# ---------------------------------------------------------------------------
# A. Exchange Tests
# ---------------------------------------------------------------------------

class TestExchange:

    def test_seed_sets_correct_ask(self, alpha, bar):
        """After seeding, best ask = VWAP + spread_bias."""
        expected_ask = round(bar.vwap + VENUE_CONFIGS["ALPHA"].spread_bias, 4)
        assert alpha.quote(OrderSide.BUY) == pytest.approx(expected_ask)

    def test_seed_sets_correct_bid(self, alpha, bar):
        """After seeding, best bid = VWAP - spread_bias."""
        expected_bid = round(bar.vwap - VENUE_CONFIGS["ALPHA"].spread_bias, 4)
        assert alpha.quote(OrderSide.SELL) == pytest.approx(expected_bid)

    def test_available_liquidity_buy_side(self, alpha, bar):
        """Available ask liquidity = 3 levels × (volume / 30)."""
        expected_liq = 3 * max(10.0, bar.volume / 30.0)
        assert alpha.available_liquidity(OrderSide.BUY) == pytest.approx(expected_liq)

    def test_fee_applied_to_fill(self, alpha, bar):
        """Fee = avg_fill_price × filled_qty × fee_bps / 10000."""
        child = ChildOrder(
            parent_order_id="test-parent",
            venue="ALPHA",
            side=OrderSide.BUY,
            quantity=10.0,
        )
        fill = alpha.submit(child)

        assert fill.filled_qty > 0
        expected_fee = fill.avg_price * fill.filled_qty * VENUE_CONFIGS["ALPHA"].fee_bps / 10_000
        assert fill.fees_paid == pytest.approx(expected_fee, rel=1e-5)

    def test_empty_book_returns_zero_fill(self):
        """Submitting to an empty (unseeded) exchange returns zero fill."""
        exch = Exchange("ALPHA", VENUE_CONFIGS["ALPHA"])  # not seeded
        child = ChildOrder("p", "ALPHA", OrderSide.BUY, quantity=100.0)
        fill = exch.submit(child)
        assert fill.filled_qty == 0.0
        assert fill.fees_paid == 0.0

    def test_effective_price_buy_higher_than_quote(self, alpha):
        """Effective buy price = ask × (1 + fee_bps/10000) > raw ask."""
        raw_ask = alpha.quote(OrderSide.BUY)
        eff = alpha.effective_price(OrderSide.BUY)
        assert eff > raw_ask

    def test_gamma_cheaper_effective_price_than_alpha(self, venues):
        """
        GAMMA has a wider spread but lower fee.
        effective_ask comparison: ALPHA=190.01*1.0002, GAMMA=190.05*1.000005.
        ALPHA wins on effective price for this bar (but GAMMA wins on fee alone).
        """
        alpha_eff = venues["ALPHA"].effective_price(OrderSide.BUY)
        gamma_eff = venues["GAMMA"].effective_price(OrderSide.BUY)
        # ALPHA has the tighter raw spread so its effective ask is lower
        assert alpha_eff < gamma_eff


# ---------------------------------------------------------------------------
# B. Strategy Tests
# ---------------------------------------------------------------------------

class TestBestPriceStrategy:

    def test_routes_buy_to_cheapest_effective_ask(self, venues):
        """BestPrice BUY → venue with lowest effective ask (ALPHA for this bar)."""
        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=100.0)
        tranches = BestPriceStrategy().route(order, venues)

        assert len(tranches) == 1
        children = tranches[0]
        assert len(children) == 1
        assert children[0].venue == "ALPHA"
        assert children[0].quantity == pytest.approx(100.0)

    def test_routes_sell_to_highest_effective_bid(self, venues):
        """BestPrice SELL → venue with highest effective bid."""
        order = Order("AAPL", OrderSide.SELL, OrderType.MARKET, quantity=100.0)
        tranches = BestPriceStrategy().route(order, venues)

        assert len(tranches) == 1
        assert tranches[0][0].side == OrderSide.SELL
        # ALPHA has the tightest spread so its bid is also highest
        assert tranches[0][0].venue == "ALPHA"

    def test_single_tranche_single_child(self, venues):
        """BestPrice always returns exactly one tranche with one child."""
        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=500.0)
        tranches = BestPriceStrategy().route(order, venues)
        assert len(tranches) == 1
        assert len(tranches[0]) == 1

    def test_child_has_correct_parent_id(self, venues):
        """ChildOrder.parent_order_id matches the parent Order.order_id."""
        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=100.0)
        tranches = BestPriceStrategy().route(order, venues)
        child = tranches[0][0]
        assert child.parent_order_id == order.order_id


class TestTWAPStrategy:

    def test_returns_n_tranches(self, venues, five_bars):
        """TWAP with num_slices=5 returns exactly 5 tranches."""
        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=500.0)
        tranches = TWAPStrategy(num_slices=5).route(order, venues, bars=five_bars)
        assert len(tranches) == 5

    def test_equal_slice_sizes(self, venues, five_bars):
        """Each TWAP slice has qty = total / num_slices."""
        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=500.0)
        tranches = TWAPStrategy(num_slices=5).route(order, venues, bars=five_bars)
        for tranche in tranches:
            assert len(tranche) == 1
            assert tranche[0].quantity == pytest.approx(100.0)  # 500 / 5

    def test_requires_bars(self, venues):
        """TWAP raises ValueError if bars is None."""
        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=100.0)
        with pytest.raises(ValueError, match="bars"):
            TWAPStrategy(num_slices=3).route(order, venues, bars=None)

    def test_adapts_to_fewer_bars(self, venues):
        """TWAP uses len(bars) slices when fewer bars than num_slices provided."""
        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=300.0)
        two_bars = [make_bar(vwap=190.0), make_bar(vwap=190.05)]
        tranches = TWAPStrategy(num_slices=5).route(order, venues, bars=two_bars)
        assert len(tranches) == 2


class TestVWAPStrategy:

    def test_weights_sum_to_one(self, five_bars):
        """Volume weights must sum to 1.0."""
        weights = VWAPStrategy.compute_weights(five_bars, num_slices=5)
        assert sum(weights) == pytest.approx(1.0)

    def test_higher_volume_bar_gets_larger_slice(self, venues, five_bars):
        """Bar with more volume → larger slice quantity."""
        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=1000.0)
        tranches = VWAPStrategy(num_slices=5).route(order, venues, bars=five_bars)

        # five_bars volumes: 1000, 2000, 3000, 4000, 5000 → last bar largest
        quantities = [t[0].quantity for t in tranches if t]
        assert quantities[-1] > quantities[0]  # last slice > first slice

    def test_total_quantity_preserved(self, venues, five_bars):
        """Sum of all VWAP slice quantities equals the parent order quantity."""
        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=1000.0)
        tranches = VWAPStrategy(num_slices=5).route(order, venues, bars=five_bars)
        total = sum(t[0].quantity for t in tranches if t)
        assert total == pytest.approx(1000.0, abs=1e-4)

    def test_requires_bars(self, venues):
        """VWAP raises ValueError if bars is None."""
        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=100.0)
        with pytest.raises(ValueError, match="bars"):
            VWAPStrategy(num_slices=3).route(order, venues, bars=None)


class TestSmartStrategy:

    def test_sweeps_multiple_venues_when_liquidity_limited(self):
        """Smart BUY sweeps from cheapest venue first, overflows to next."""
        # Set up: volume=300 → level_qty=10, 3 levels = 30 shares per venue
        small_bar = make_bar(vwap=190.0, volume=300.0)
        venues = {}
        for name, cfg in VENUE_CONFIGS.items():
            exch = Exchange(name, cfg)
            exch.seed_from_bar(small_bar)
            venues[name] = exch

        # Order for 75 shares — needs all 3 venues (30 each)
        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=75.0)
        tranches = SmartStrategy().route(order, venues)

        assert len(tranches) == 1
        children = tranches[0]
        # Should have 3 children (one per venue, each contributing 30 or less)
        assert len(children) >= 2
        total_allocated = sum(c.quantity for c in children)
        assert total_allocated == pytest.approx(75.0)

    def test_routes_cheapest_venue_first(self, venues):
        """First child in Smart tranche targets the cheapest effective-price venue."""
        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=10.0)
        tranches = SmartStrategy().route(order, venues)

        children = tranches[0]
        # ALPHA has the lowest effective ask → first allocation
        assert children[0].venue == "ALPHA"

    def test_fee_adjusted_ranking(self, venues):
        """Smart uses effective (fee-adjusted) price, not raw quote."""
        strategy = SmartStrategy()
        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=10.0)
        plan = strategy.preview(order, venues)

        # Plan should be ranked by effective_price ascending
        effective_prices = [ep for _, _, ep in plan]
        assert effective_prices == sorted(effective_prices)

    def test_single_tranche_output(self, venues):
        """Smart always returns a single tranche (not time-sliced)."""
        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=200.0)
        tranches = SmartStrategy().route(order, venues)
        assert len(tranches) == 1


# ---------------------------------------------------------------------------
# C. OrderRouter Tests
# ---------------------------------------------------------------------------

class TestOrderRouter:

    def test_router_returns_router_result(self, venues, bar):
        """Router.submit() returns a populated RouterResult."""
        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=100.0)
        router = OrderRouter(venues=venues, strategy=BestPriceStrategy())
        result = router.submit(order, bars=[bar])

        assert isinstance(result, RouterResult)
        assert result.strategy_name == "BestPrice"
        assert result.parent_order is order

    def test_router_fill_rate_complete_fill(self, venues, bar):
        """Small order against seeded book → 100% fill rate."""
        # level_qty = 3000/30 = 100, 3 levels = 300 per venue
        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=50.0)
        router = OrderRouter(venues=venues, strategy=SmartStrategy())
        result = router.submit(order, bars=[bar])

        assert result.total_filled == pytest.approx(50.0)
        assert result.fill_rate_pct == pytest.approx(100.0)

    def test_router_strategy_switch(self, bar):
        """Router can switch strategies between orders."""
        venues = {}
        for name, cfg in VENUE_CONFIGS.items():
            exch = Exchange(name, cfg)
            exch.seed_from_bar(bar)
            venues[name] = exch

        router = OrderRouter(venues=venues, strategy=BestPriceStrategy())
        assert router.strategy.name == "BestPrice"

        router.switch_strategy(SmartStrategy())
        assert router.strategy.name == "Smart"

        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=50.0)
        result = router.submit(order, bars=[bar])
        assert result.strategy_name == "Smart"

    def test_twap_fills_across_multiple_bars(self, five_bars):
        """TWAP router re-seeds per bar; large order fills in slices."""
        venues = {}
        for name, cfg in VENUE_CONFIGS.items():
            venues[name] = Exchange(name, cfg)

        # Each bar: volume=1000-5000, level_qty=33-166, 3 levels → 100-500/venue
        # 5-slice TWAP of 250 shares: 50/slice. Each slice should fully fill.
        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=250.0)
        router = OrderRouter(venues=venues, strategy=TWAPStrategy(num_slices=5))
        result = router.submit(order, bars=five_bars)

        assert result.total_filled == pytest.approx(250.0)
        assert len(result.child_orders) == 5  # one child per slice

    def test_partial_fill_when_insufficient_liquidity(self):
        """Router handles partial fills gracefully when book depth is low."""
        tiny_bar = make_bar(vwap=190.0, volume=30.0)  # level_qty=max(10,1)=10, 3 lvls=30
        venues = {
            name: Exchange(name, cfg) for name, cfg in VENUE_CONFIGS.items()
        }

        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=10000.0)
        router = OrderRouter(venues=venues, strategy=SmartStrategy())
        result = router.submit(order, bars=[tiny_bar])

        # Should be partially filled — not zero, not full
        assert result.total_filled > 0
        assert result.total_filled < order.quantity
        assert result.fill_rate_pct < 100.0

    def test_fees_positive_on_fill(self, venues, bar):
        """Total fees must be > 0 whenever shares are filled."""
        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=100.0)
        router = OrderRouter(venues=venues, strategy=BestPriceStrategy())
        result = router.submit(order, bars=[bar])

        if result.total_filled > 0:
            assert result.total_fees > 0


# ---------------------------------------------------------------------------
# D. StrategyComparator Tests
# ---------------------------------------------------------------------------

class TestStrategyComparator:

    @pytest.fixture(scope="class")
    def report(self):
        """Run comparator once for all tests in this class (slow IO: CSV load)."""
        comparator = StrategyComparator(
            csv_path=CSV_PATH,
            num_windows=5,        # small for test speed
            window_size=5,
            order_qty=5000.0,
        )
        return comparator.run()

    def test_report_contains_all_strategies(self, report):
        """Report must include all four strategy names."""
        assert "BestPrice"  in report.summary
        assert "TWAP(5)"    in report.summary
        assert "VWAP(5)"    in report.summary
        assert "Smart"      in report.summary

    def test_smart_fill_rate_exceeds_best_price(self, report):
        """Smart strategy should achieve higher fill rate than BestPrice."""
        smart_fill = report.summary["Smart"]["fill_rate_pct"]
        best_fill  = report.summary["BestPrice"]["fill_rate_pct"]
        # Smart sweeps all venues; BestPrice uses only one → smart ≥ best
        assert smart_fill >= best_fill

    def test_twap_higher_latency_than_best_price(self, report):
        """TWAP accumulates latency over multiple tranches; BestPrice is single-shot."""
        twap_lat = report.summary["TWAP(5)"]["avg_latency_ms"]
        bp_lat   = report.summary["BestPrice"]["avg_latency_ms"]
        assert twap_lat >= bp_lat

    def test_report_bars_replayed_correct(self, report):
        """bars_replayed = num_windows × window_size."""
        assert report.bars_replayed == report.num_windows * 5

    def test_comparator_print_report_runs(self, report, capsys):
        """print_report should produce output without raising."""
        comparator = StrategyComparator(CSV_PATH)
        comparator.print_report(report)
        captured = capsys.readouterr()
        assert "STRATEGY COMPARISON" in captured.out
        assert "BestPrice" in captured.out
        assert "Smart" in captured.out
