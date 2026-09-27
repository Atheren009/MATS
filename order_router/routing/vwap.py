"""
routing/vwap.py — Volume-Weighted Average Price (VWAP) routing strategy.

Divides the parent order into N time slices where each slice's size is
proportional to the corresponding bar's traded volume.  High-volume bars
receive larger slices — targeting the market's own natural volume rhythm.

Pros  : Aligns execution with market liquidity; benchmark-friendly (tracks
        the VWAP reference price); theoretically reduces market impact.
Cons  : Requires volume forecasts (here we use actual bar data); still
        spreads over time like TWAP (higher total latency than BestPrice/Smart).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import RoutingStrategy
from ..child_order import ChildOrder
from ..models import Order, OrderSide
from ..price_feed import Bar


class VWAPStrategy(RoutingStrategy):
    """
    Volume-weighted time-sliced execution.

    Slice sizes are computed as:
        slice_qty[i] = order.quantity × (bars[i].volume / Σ bars[j].volume)

    Parameters
    ----------
    num_slices : int
        Maximum number of time slices (default 5).
        Capped to len(bars) if fewer bars are provided.
    min_qty    : float
        Minimum child order quantity; slices smaller than this are skipped
        to avoid sending negligibly small orders. Default 1.0.
    """

    def __init__(self, num_slices: int = 5, min_qty: float = 1.0) -> None:
        if num_slices < 1:
            raise ValueError(f"num_slices must be >= 1, got {num_slices}")
        self.num_slices = num_slices
        self.min_qty = min_qty

    @property
    def name(self) -> str:
        return f"VWAP({self.num_slices})"

    def route(
        self,
        order: Order,
        venues: Dict,
        bars: Optional[List[Bar]] = None,
    ) -> List[List[ChildOrder]]:
        """
        Return one tranche per bar, sized by that bar's share of total volume.
        """
        if bars is None:
            raise ValueError(
                "VWAPStrategy requires a list of price bars. "
                "Pass bars= to OrderRouter.submit()."
            )

        n = min(self.num_slices, len(bars))
        if n == 0:
            return [[]]

        window = bars[:n]
        total_volume = sum(b.volume for b in window)

        if total_volume <= 0:
            # Fall back to equal weighting if volume data is missing
            weights = [1.0 / n] * n
        else:
            weights = [b.volume / total_volume for b in window]

        best = self._best_venue(venues, order.side)
        if best is None:
            return [[]] * n

        tranches: List[List[ChildOrder]] = []
        allocated = 0.0

        for i, w in enumerate(weights):
            # Last slice takes any rounding remainder
            if i == n - 1:
                slice_qty = order.quantity - allocated
            else:
                slice_qty = round(order.quantity * w, 6)

            if slice_qty < self.min_qty:
                tranches.append([])
                continue

            allocated += slice_qty
            child = ChildOrder(
                parent_order_id=order.order_id,
                venue=best,
                side=order.side,
                quantity=slice_qty,
                limit_price=None,
            )
            tranches.append([child])

        return tranches

    # ------------------------------------------------------------------
    # Utility: expose weights for inspection / testing
    # ------------------------------------------------------------------

    @staticmethod
    def compute_weights(bars: List[Bar], num_slices: int = 5) -> List[float]:
        """Return volume weights for bars[:num_slices].  Sums to 1.0."""
        n = min(num_slices, len(bars))
        window = bars[:n]
        total = sum(b.volume for b in window)
        if total <= 0:
            return [1.0 / n] * n
        return [b.volume / total for b in window]
