"""
Domain models for the Order Router matching engine.

All financial quantities use Python float for simplicity; a production system
would use decimal.Decimal to avoid floating-point rounding issues.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class OrderStatus(Enum):
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """
    Represents a single order submitted to the matching engine.

    Attributes
    ----------
    symbol      : Ticker symbol (e.g. "AAPL").
    side        : BUY or SELL.
    order_type  : LIMIT or MARKET.
    quantity    : Total requested quantity (shares).
    price       : Limit price; None for MARKET orders.
    order_id    : Unique identifier (auto-generated UUID if not supplied).
    timestamp   : UTC time when the order was created.
    filled_qty  : Quantity matched so far.
    status      : Current lifecycle status.
    """

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None          # None → MARKET order

    # Auto-populated fields
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_qty: float = field(default=0.0, init=False)
    status: OrderStatus = field(default=OrderStatus.OPEN, init=False)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def remaining_qty(self) -> float:
        """Quantity still awaiting a fill."""
        return self.quantity - self.filled_qty

    @property
    def is_active(self) -> bool:
        """True while the order can still receive fills."""
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)

    def fill(self, qty: float) -> None:
        """Record a (partial) fill against this order."""
        self.filled_qty += qty
        if self.filled_qty >= self.quantity:
            self.filled_qty = self.quantity   # clamp to avoid float drift
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Order(id={self.order_id[:8]}… {self.side.value} {self.quantity}"
            f"@{self.price} [{self.status.value}] filled={self.filled_qty})"
        )


# ---------------------------------------------------------------------------
# Trade
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """
    Represents a single executed match between a resting and an aggressor order.

    Trade price is always the resting (maker) order's price.
    """

    symbol: str
    buy_order_id: str
    sell_order_id: str
    price: float        # maker price
    quantity: float     # matched quantity
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Trade({self.symbol} {self.quantity}@{self.price}"
            f" buy={self.buy_order_id[:8]}… sell={self.sell_order_id[:8]}…)"
        )
