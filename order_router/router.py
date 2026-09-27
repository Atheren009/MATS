"""
router.py — OrderRouter: orchestrates strategy → exchange → result pipeline.

The router is the central coordinator:
  1. Accepts a parent Order and (optionally) a list of price bars.
  2. Seeds all venues from bars[0] if bars are provided.
  3. Calls the active strategy to produce execution tranches.
  4. For each tranche (i > 0): re-seeds venues from bars[i] to simulate
     new liquidity arriving each minute (TWAP/VWAP time steps).
  5. Submits each ChildOrder to its designated exchange.
  6. Aggregates all FillResults into a single RouterResult.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .child_order import ChildOrder, FillResult
from .exchange import Exchange
from .models import Order
from .price_feed import Bar
from .routing.base import RoutingStrategy

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RouterResult
# ---------------------------------------------------------------------------

@dataclass
class RouterResult:
    """
    Aggregated execution report for a single parent Order.

    Attributes
    ----------
    parent_order      : The original order that was routed.
    strategy_name     : Name of the strategy that produced this result.
    child_orders      : All ChildOrders dispatched (filled and unfilled).
    total_filled      : Total shares actually matched across all venues.
    avg_fill_price    : Volume-weighted average fill price across all fills.
    total_fees        : Sum of all venue fees paid (dollars).
    total_slippage_bps: Fill-quantity-weighted average slippage in basis points.
    execution_time_ms : Sum of per-venue latency (simulated, not wall-clock).
    fill_rate_pct     : Percentage of requested quantity that was filled.
    """

    parent_order: Order
    strategy_name: str
    child_orders: List[ChildOrder]
    total_filled: float
    avg_fill_price: float
    total_fees: float
    total_slippage_bps: float
    execution_time_ms: int

    @property
    def fill_rate_pct(self) -> float:
        if self.parent_order.quantity <= 0:
            return 0.0
        return 100.0 * self.total_filled / self.parent_order.quantity

    @property
    def venues_used(self) -> List[str]:
        return [c.venue for c in self.child_orders if c.is_filled]

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RouterResult({self.strategy_name} "
            f"filled={self.total_filled:.0f}/{self.parent_order.quantity:.0f} "
            f"@ {self.avg_fill_price:.4f} "
            f"fees=${self.total_fees:.4f} "
            f"slip={self.total_slippage_bps:.2f}bps "
            f"lat={self.execution_time_ms}ms)"
        )


# ---------------------------------------------------------------------------
# OrderRouter
# ---------------------------------------------------------------------------

class OrderRouter:
    """
    Orchestrates routing strategy execution across multiple exchanges.

    Parameters
    ----------
    venues   : Dict mapping venue name → Exchange instance.
    strategy : The RoutingStrategy to use.

    Usage
    -----
    >>> router = OrderRouter(venues=registry.get_all(), strategy=SmartStrategy())
    >>> result = router.submit(order, bars=feed_bars[:5])
    >>> print(result.fill_rate_pct, result.avg_fill_price)
    """

    def __init__(
        self,
        venues: Dict[str, Exchange],
        strategy: RoutingStrategy,
    ) -> None:
        self.venues = venues
        self.strategy = strategy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        order: Order,
        bars: Optional[List[Bar]] = None,
    ) -> RouterResult:
        """
        Route and execute *order* according to the active strategy.

        Parameters
        ----------
        order : Parent order to route.
        bars  : Price bars for time-sliced strategies (TWAP, VWAP).
                If provided, venues are seeded from bars[0] before routing
                and re-seeded from bars[i] between tranches i > 0.
                If None, venues are assumed to be pre-seeded by the caller.
        """
        # ── Seed from first bar ──────────────────────────────────────
        if bars:
            self._seed_all(bars[0])

        # ── Get execution plan from strategy ─────────────────────────
        tranches = self.strategy.route(order, self.venues, bars)
        log.debug(
            "Strategy %s produced %d tranches for order %s",
            self.strategy.name, len(tranches), order.order_id[:8],
        )

        # ── Execute each tranche ──────────────────────────────────────
        all_children: List[ChildOrder] = []
        total_latency = 0

        for i, tranche in enumerate(tranches):
            # Re-seed venues for time step i (bars[0] already done above)
            if i > 0 and bars and i < len(bars):
                self._seed_all(bars[i])

            for child in tranche:
                if child.quantity <= 0:
                    continue
                venue = self.venues.get(child.venue)
                if venue is None:
                    log.warning("Unknown venue %r — skipping child order", child.venue)
                    continue

                fill = venue.submit(child)
                child.fill_result = fill
                total_latency += fill.latency_ms
                all_children.append(child)

        # ── Aggregate metrics ─────────────────────────────────────────
        return self._aggregate(order, all_children, total_latency)

    def switch_strategy(self, strategy: RoutingStrategy) -> None:
        """Hot-swap the routing strategy between orders."""
        log.info("Switching strategy: %s → %s", self.strategy.name, strategy.name)
        self.strategy = strategy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _seed_all(self, bar: Bar) -> None:
        for exch in self.venues.values():
            exch.seed_from_bar(bar)

    def _aggregate(
        self,
        order: Order,
        children: List[ChildOrder],
        total_latency: int,
    ) -> RouterResult:
        """Compute volume-weighted aggregates across all ChildOrder fills."""
        filled_children = [
            c for c in children
            if c.fill_result and c.fill_result.filled_qty > 0
        ]

        total_filled = sum(c.fill_result.filled_qty for c in filled_children)

        if total_filled > 0:
            avg_fill_price = (
                sum(c.fill_result.avg_price * c.fill_result.filled_qty
                    for c in filled_children)
                / total_filled
            )
            total_fees = sum(c.fill_result.fees_paid for c in filled_children)
            total_slippage = (
                sum(c.fill_result.slippage_bps * c.fill_result.filled_qty
                    for c in filled_children)
                / total_filled
            )
        else:
            avg_fill_price = 0.0
            total_fees = 0.0
            total_slippage = 0.0

        return RouterResult(
            parent_order=order,
            strategy_name=self.strategy.name,
            child_orders=children,
            total_filled=round(total_filled, 6),
            avg_fill_price=round(avg_fill_price, 6),
            total_fees=round(total_fees, 6),
            total_slippage_bps=round(total_slippage, 4),
            execution_time_ms=total_latency,
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"OrderRouter(strategy={self.strategy.name!r} "
            f"venues={list(self.venues.keys())})"
        )
