"""
exchange.py — Simulated exchange wrapping an OrderBook.

Each exchange has:
  * A per-venue fee rate (basis points)
  * A simulated latency (milliseconds, not a real sleep — tracked for metrics)
  * A spread bias applied when seeding liquidity from a price bar

Three pre-configured venues are exposed via VENUE_CONFIGS:

  ALPHA — tight spread, mid fee,  fast  (2 bps, 1 ms,  ±0.01)
  BETA  — wide spread,  low fee,  medium(1 bps, 5 ms,  ±0.03)
  GAMMA — widest spread, cheapest, slow  (0.5 bps, 15 ms, ±0.05)

This creates a deliberate three-way trade-off so routing strategies have
meaningful decisions to make.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from .child_order import ChildOrder, FillResult
from .models import Order, OrderSide, OrderType
from .order_book import OrderBook
from .price_feed import Bar

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Venue configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VenueConfig:
    """Immutable per-venue parameters."""
    fee_bps: float      # transaction fee in basis points
    latency_ms: int     # simulated round-trip latency (tracked, not slept)
    spread_bias: float  # half-spread added/subtracted from VWAP when seeding


VENUE_CONFIGS: Dict[str, VenueConfig] = {
    "ALPHA": VenueConfig(fee_bps=2.0,  latency_ms=1,  spread_bias=0.01),
    "BETA":  VenueConfig(fee_bps=1.0,  latency_ms=5,  spread_bias=0.03),
    "GAMMA": VenueConfig(fee_bps=0.5,  latency_ms=15, spread_bias=0.05),
}


# ---------------------------------------------------------------------------
# Exchange
# ---------------------------------------------------------------------------

class Exchange:
    """
    Simulated exchange backed by a live OrderBook.

    Workflow
    --------
    1. Call ``seed_from_bar(bar)`` to populate the book with synthetic
       market-maker liquidity derived from a real AAPL OHLCV bar.
    2. Call ``submit(child_order)`` to execute a ChildOrder against the book
       and receive a FillResult.

    Seeding
    -------
    Three ask levels above VWAP and three bid levels below VWAP are placed,
    each with ``qty = max(10.0, bar.volume / 30)``.  Spread bias shifts all
    levels: ALPHA is tightest (cheapest ask), GAMMA is widest (most expensive
    ask) — but GAMMA's lower fee compensates, making strategy choice non-trivial.
    """

    def __init__(self, name: str, config: VenueConfig, symbol: str = "AAPL") -> None:
        self.name = name
        self.config = config
        self.symbol = symbol
        self._book: OrderBook = OrderBook(symbol)
        self._ref_price: float = 0.0

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed_from_bar(self, bar: Bar) -> None:
        """
        Reset the book and populate with synthetic market-maker orders.

        Ask levels: VWAP + spread_bias, + 0.05, + 0.10
        Bid levels: VWAP − spread_bias, − 0.05, − 0.10
        Qty per level: max(10.0, bar.volume / 30)
        """
        self._book = OrderBook(self.symbol)
        self._ref_price = bar.vwap
        level_qty = max(10.0, bar.volume / 30.0)

        ask_base = round(bar.vwap + self.config.spread_bias, 4)
        bid_base = round(bar.vwap - self.config.spread_bias, 4)

        for i in range(3):
            ask_price = round(ask_base + i * 0.05, 4)
            mm_sell = Order(
                self.symbol, OrderSide.SELL, OrderType.LIMIT,
                quantity=level_qty, price=ask_price,
            )
            mm_sell.order_id = f"mm_{self.name}_ask_{i}"
            self._book._rest(mm_sell)

        for i in range(3):
            bid_price = round(bid_base - i * 0.05, 4)
            mm_buy = Order(
                self.symbol, OrderSide.BUY, OrderType.LIMIT,
                quantity=level_qty, price=bid_price,
            )
            mm_buy.order_id = f"mm_{self.name}_bid_{i}"
            self._book._rest(mm_buy)

        log.debug(
            "Exchange %s seeded: ask=%.4f bid=%.4f liq/level=%.0f",
            self.name, ask_base, bid_base, level_qty,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def submit(self, child: ChildOrder) -> FillResult:
        """
        Execute a ChildOrder against this exchange's order book.

        The order is submitted as an aggressor (MARKET or LIMIT).
        Returns a FillResult with filled quantity, VWAP fill price,
        fees charged, slippage vs the seeded reference price, and latency.
        """
        order_type = OrderType.LIMIT if child.limit_price is not None else OrderType.MARKET
        aggressor = Order(
            symbol=self.symbol,
            side=child.side,
            order_type=order_type,
            quantity=child.quantity,
            price=child.limit_price,
        )

        n_before = len(self._book.trades)
        self._book.add_order(aggressor)
        new_trades = self._book.trades[n_before:]

        filled_qty = sum(t.quantity for t in new_trades)

        if filled_qty == 0:
            return FillResult(
                venue=self.name,
                order_id=aggressor.order_id,
                filled_qty=0.0,
                avg_price=0.0,
                fees_paid=0.0,
                slippage_bps=0.0,
                latency_ms=self.config.latency_ms,
            )

        avg_price = sum(t.price * t.quantity for t in new_trades) / filled_qty
        fees_paid = avg_price * filled_qty * self.config.fee_bps / 10_000.0

        ref = self._ref_price if self._ref_price > 0 else avg_price
        if child.side is OrderSide.BUY:
            slippage_bps = (avg_price - ref) / ref * 10_000.0
        else:
            slippage_bps = (ref - avg_price) / ref * 10_000.0

        result = FillResult(
            venue=self.name,
            order_id=aggressor.order_id,
            filled_qty=round(filled_qty, 6),
            avg_price=round(avg_price, 6),
            fees_paid=round(fees_paid, 6),
            slippage_bps=round(slippage_bps, 4),
            latency_ms=self.config.latency_ms,
        )
        log.debug("FILL %s: %s", self.name, result)
        return result

    # ------------------------------------------------------------------
    # Quote / liquidity helpers
    # ------------------------------------------------------------------

    def quote(self, side: OrderSide) -> Optional[float]:
        """
        Best quote on the side a *taker* would hit.

        BUY  taker → looks at asks (lowest ask).
        SELL taker → looks at bids (highest bid).
        """
        if side is OrderSide.BUY:
            return self._book.get_best_ask()
        return self._book.get_best_bid()

    def effective_price(self, side: OrderSide) -> Optional[float]:
        """
        Fee-adjusted quote used for cross-venue ranking.

        BUY:  effective = ask × (1 + fee_bps / 10 000)  → lower is better
        SELL: effective = bid × (1 − fee_bps / 10 000)  → higher is better
        """
        q = self.quote(side)
        if q is None:
            return None
        if side is OrderSide.BUY:
            return q * (1.0 + self.config.fee_bps / 10_000.0)
        return q * (1.0 - self.config.fee_bps / 10_000.0)

    def available_liquidity(self, side: OrderSide) -> float:
        """
        Total resting quantity available for a taker to consume.

        BUY  taker → sums all ask levels.
        SELL taker → sums all bid levels.
        """
        depth = self._book.get_depth(levels=20)
        if side is OrderSide.BUY:
            return sum(qty for _, qty in depth["asks"])
        return sum(qty for _, qty in depth["bids"])

    def reset(self) -> None:
        """Clear the book and reference price without re-seeding."""
        self._book = OrderBook(self.symbol)
        self._ref_price = 0.0

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Exchange({self.name!r} fee={self.config.fee_bps}bps "
            f"lat={self.config.latency_ms}ms "
            f"ask={self.quote(OrderSide.BUY)} bid={self.quote(OrderSide.SELL)})"
        )
