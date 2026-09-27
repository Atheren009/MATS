"""
child_order.py — FillResult and ChildOrder dataclasses.

A ChildOrder is the atomic unit the router dispatches to a single exchange.
After execution, its fill_result is populated by the exchange.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from .models import OrderSide


# ---------------------------------------------------------------------------
# FillResult
# ---------------------------------------------------------------------------

@dataclass
class FillResult:
    """
    Execution report returned by an Exchange after processing a ChildOrder.

    Attributes
    ----------
    venue        : Exchange name (e.g. "ALPHA").
    order_id     : ID of the aggressor Order object submitted to the book.
    filled_qty   : Shares actually matched (may be < requested).
    avg_price    : Volume-weighted average fill price.
    fees_paid    : Dollar fees charged (avg_price × filled_qty × fee_bps / 10 000).
    slippage_bps : (fill − ref_price) / ref_price × 10 000; positive = unfavourable.
    latency_ms   : Simulated execution latency for this venue.
    """

    venue: str
    order_id: str
    filled_qty: float
    avg_price: float
    fees_paid: float
    slippage_bps: float
    latency_ms: int

    @property
    def total_cost(self) -> float:
        """All-in cost: fill notional + fees."""
        return self.avg_price * self.filled_qty + self.fees_paid

    @property
    def is_filled(self) -> bool:
        return self.filled_qty > 0


# ---------------------------------------------------------------------------
# ChildOrder
# ---------------------------------------------------------------------------

@dataclass
class ChildOrder:
    """
    A single exchange-level execution slice derived from a parent Order.

    Created by a RoutingStrategy; filled by an Exchange; aggregated by the Router.

    Attributes
    ----------
    parent_order_id : ID of the originating parent Order.
    venue           : Target exchange name.
    side            : BUY or SELL.
    quantity        : Requested quantity for this child.
    limit_price     : Optional limit price; None → submit as MARKET.
    child_id        : Auto-generated UUID.
    fill_result     : Populated after exchange execution; None until then.
    """

    parent_order_id: str
    venue: str
    side: OrderSide
    quantity: float
    limit_price: Optional[float] = None

    child_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fill_result: Optional[FillResult] = None

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def is_filled(self) -> bool:
        return self.fill_result is not None and self.fill_result.is_filled

    @property
    def filled_qty(self) -> float:
        return self.fill_result.filled_qty if self.fill_result else 0.0

    @property
    def avg_price(self) -> float:
        return self.fill_result.avg_price if self.fill_result else 0.0

    def __repr__(self) -> str:  # pragma: no cover
        filled = f"{self.filled_qty:.0f}/{self.quantity:.0f}"
        return f"ChildOrder({self.venue} {self.side.value} {filled} @{self.avg_price:.4f})"
