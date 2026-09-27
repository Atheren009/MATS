"""
routing/smart.py — Smart Order Router (SOR) strategy.

The Smart strategy is the primary differentiator.  It behaves like a real
Smart Order Router:

  1. Ranks all venues by fee-adjusted effective price (best first).
  2. Allocates quantity greedily to the cheapest venue up to that venue's
     available liquidity.
  3. If a venue is exhausted before the order is fully filled, it moves to
     the next-best venue — sweeping the market depth across all exchanges.
  4. The entire sweep happens at a single logical time step (one tranche).

Pros  : Maximises fill rate across venues; best execution on fee-adjusted price;
        low overall latency (no waiting between slices).
Cons  : More complex than single-venue strategies; may lift multiple venues,
        revealing the order's presence to market participants.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import RoutingStrategy
from ..child_order import ChildOrder
from ..models import Order, OrderSide
from ..price_feed import Bar


class SmartStrategy(RoutingStrategy):
    """
    Cross-venue depth sweep ordered by fee-adjusted effective price.

    Each venue is allocated the lesser of:
      (a) its available liquidity on the taker side, and
      (b) the remaining unfilled quantity.

    Only venues with available liquidity > min_qty are included.

    Parameters
    ----------
    min_qty : float
        Minimum allocation per venue; venues below this threshold are skipped.
        Default 1.0.
    """

    def __init__(self, min_qty: float = 1.0) -> None:
        self.min_qty = min_qty

    @property
    def name(self) -> str:
        return "Smart"

    def route(
        self,
        order: Order,
        venues: Dict,
        bars: Optional[List[Bar]] = None,
    ) -> List[List[ChildOrder]]:
        """
        Single-tranche cross-venue sweep.

        Returns one tranche containing up to len(venues) ChildOrders,
        each targeting a separate exchange for a portion of the quantity.
        """
        ranked = self._rank_venues_by_effective_price(venues, order.side)

        remaining = order.quantity
        children: List[ChildOrder] = []

        for venue_name in ranked:
            if remaining <= 0:
                break

            exch = venues[venue_name]
            liquidity = exch.available_liquidity(order.side)

            if liquidity < self.min_qty:
                continue

            alloc = min(remaining, liquidity)
            alloc = round(alloc, 6)

            child = ChildOrder(
                parent_order_id=order.order_id,
                venue=venue_name,
                side=order.side,
                quantity=alloc,
                limit_price=None,
            )
            children.append(child)
            remaining = round(remaining - alloc, 6)

        if not children:
            return [[]]

        return [children]

    # ------------------------------------------------------------------
    # Utility: expose the allocation plan before execution
    # ------------------------------------------------------------------

    def preview(
        self,
        order: Order,
        venues: Dict,
    ) -> List[tuple]:
        """
        Return planned (venue_name, allocated_qty, effective_price) tuples
        without creating ChildOrders.  Useful for inspection and testing.
        """
        ranked = self._rank_venues_by_effective_price(venues, order.side)
        remaining = order.quantity
        plan = []

        for venue_name in ranked:
            if remaining <= 0:
                break
            exch = venues[venue_name]
            liquidity = exch.available_liquidity(order.side)
            if liquidity < self.min_qty:
                continue
            alloc = min(remaining, liquidity)
            ep = exch.effective_price(order.side)
            plan.append((venue_name, round(alloc, 6), ep))
            remaining = round(remaining - alloc, 6)

        return plan
