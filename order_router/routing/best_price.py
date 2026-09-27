"""
routing/best_price.py — Best Price (greedy single-venue) routing strategy.

Routes the entire parent order quantity to whichever exchange offers the
best fee-adjusted quote at the moment of submission.

Pros  : Simplest possible strategy; lowest latency (one venue, one round-trip).
Cons  : Concentrates all liquidity risk on one venue; poor fill rate when the
        chosen venue lacks depth for the full order quantity.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import RoutingStrategy
from ..child_order import ChildOrder
from ..models import Order, OrderSide
from ..price_feed import Bar


class BestPriceStrategy(RoutingStrategy):
    """
    Route 100 % of the order to the single venue with the best
    fee-adjusted quote.

    Tie-breaking: when effective prices are equal, ALPHA (lowest latency)
    wins by virtue of being first in the sorted dict iteration.
    """

    @property
    def name(self) -> str:
        return "BestPrice"

    def route(
        self,
        order: Order,
        venues: Dict,
        bars: Optional[List[Bar]] = None,
    ) -> List[List[ChildOrder]]:
        best = self._best_venue(venues, order.side)

        if best is None:
            # No venue has a quote — return empty tranche (unfilled)
            return [[]]

        child = ChildOrder(
            parent_order_id=order.order_id,
            venue=best,
            side=order.side,
            quantity=order.quantity,
            limit_price=None,   # market order — take whatever the book has
        )
        return [[child]]
