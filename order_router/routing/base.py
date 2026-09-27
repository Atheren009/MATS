"""
routing/base.py — Abstract base class for all routing strategies.

A RoutingStrategy decides how to split a parent Order across one or more
exchanges.  It returns a *list of tranches*, where each tranche is itself a
list of ChildOrders that should be submitted simultaneously.

Single-tranche strategies (BestPrice, Smart) return  [[c1, c2, …]].
Multi-tranche strategies (TWAP, VWAP)       return  [[c_t0], [c_t1], …].

The OrderRouter processes each tranche in sequence, optionally re-seeding
venues from the next bar between tranches to simulate time passing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..child_order import ChildOrder
from ..models import Order, OrderSide
from ..price_feed import Bar


class RoutingStrategy(ABC):
    """
    Abstract routing strategy.

    Subclasses implement ``route()`` to return execution tranches.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy identifier used in reports."""
        ...

    @abstractmethod
    def route(
        self,
        order: Order,
        venues: Dict,           # Dict[str, Exchange]  (typed loosely to avoid circular import)
        bars: Optional[List[Bar]] = None,
    ) -> List[List[ChildOrder]]:
        """
        Compute execution plan for *order* given current venue state.

        Parameters
        ----------
        order  : The parent order to route.
        venues : Current exchange snapshots (name → Exchange).
        bars   : Optional list of upcoming price bars; required by time-sliced
                 strategies (TWAP, VWAP).

        Returns
        -------
        List of tranches.  Each tranche is a list of ChildOrders to submit
        at the same logical time step.  Single-step strategies return a list
        with exactly one tranche.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers available to all strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _rank_venues_by_effective_price(
        venues: Dict,
        side: OrderSide,
    ) -> List[str]:
        """
        Return venue names sorted by fee-adjusted quote, best first.

        BUY  → ascending effective ask  (lower = cheaper to buy from)
        SELL → descending effective bid (higher = better to sell into)
        """
        scored = []
        for name, exch in venues.items():
            ep = exch.effective_price(side)
            if ep is not None:
                scored.append((ep, name))

        if not scored:
            return []

        # For SELL we want highest effective bid first → negate for sort
        reverse = side is OrderSide.SELL
        scored.sort(key=lambda x: x[0], reverse=reverse)
        return [name for _, name in scored]

    @staticmethod
    def _best_venue(venues: Dict, side: OrderSide) -> Optional[str]:
        """Return the single best venue name by fee-adjusted price, or None."""
        ranked = RoutingStrategy._rank_venues_by_effective_price(venues, side)
        return ranked[0] if ranked else None
