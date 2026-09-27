"""
OrderBook — price-time priority matching engine.

Design
------
* Bids  → SortedDict keyed by *negative* price so the highest bid is first.
* Asks  → SortedDict keyed by price so the lowest ask is first.
* Each price level stores a collections.deque of Orders (FIFO queue).
* Matching always executes at the **resting (maker) order's price**.
* Market orders sweep the opposite side until filled or the book is empty.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, Deque, List, Optional, Tuple

from sortedcontainers import SortedDict

from .exceptions import InvalidOrderError, OrderNotFoundError
from .models import Order, OrderSide, OrderStatus, OrderType, Trade

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

PriceLevel = Deque[Order]
BookSide = SortedDict       # key: float price (or neg-price for bids)


# ---------------------------------------------------------------------------
# OrderBook
# ---------------------------------------------------------------------------

class OrderBook:
    """
    A single-symbol limit order book with a price-time priority matching engine.

    Parameters
    ----------
    symbol : str
        The ticker this book manages (e.g. "AAPL").

    Usage
    -----
    >>> book = OrderBook("AAPL")
    >>> buy  = Order("AAPL", OrderSide.BUY,  OrderType.LIMIT, quantity=100, price=190.00)
    >>> sell = Order("AAPL", OrderSide.SELL, OrderType.LIMIT, quantity=100, price=189.50)
    >>> book.add_order(buy)
    >>> book.add_order(sell)   # triggers a match → 1 trade at 190.00
    >>> book.trades
    [Trade(AAPL 100@190.0 …)]
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

        # Bids keyed by *negative* price → SortedDict iterates lowest-key-first,
        # which means highest-bid first when we negate.
        self._bids: BookSide = SortedDict()   # key: -price → PriceLevel
        self._asks: BookSide = SortedDict()   # key:  price → PriceLevel

        # Fast lookup: order_id → (side, price_key)
        self._order_index: Dict[str, Tuple[OrderSide, float]] = {}

        # Execution log
        self.trades: List[Trade] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_order(self, order: Order) -> None:
        """
        Submit a new order to the book.

        * LIMIT orders attempt to match first; any remainder rests in the book.
        * MARKET orders sweep the opposite side until filled or liquidity exhausted;
          any unfilled remainder is left with PARTIALLY_FILLED / OPEN status
          (not placed in the book as a resting order).

        Raises
        ------
        InvalidOrderError
            If quantity ≤ 0, or if a LIMIT order has price ≤ 0.
        """
        self._validate(order)
        log.debug("add_order: %s", order)

        self._match(order)

        # Resting logic: only unmatched LIMIT orders rest in the book
        if order.order_type is OrderType.LIMIT and order.is_active:
            self._rest(order)

    def cancel_order(self, order_id: str) -> Order:
        """
        Cancel a resting order and remove it from the book.

        Raises
        ------
        OrderNotFoundError
            If order_id is not in the book (already filled, cancelled, or unknown).
        """
        if order_id not in self._order_index:
            raise OrderNotFoundError(order_id)

        side, price_key = self._order_index.pop(order_id)
        book_side = self._bids if side is OrderSide.BUY else self._asks

        level: PriceLevel = book_side[price_key]
        # Find and remove
        for i, o in enumerate(level):
            if o.order_id == order_id:
                del level[i]
                o.status = OrderStatus.CANCELLED
                log.debug("cancel_order: %s", order_id)
                break

        # Clean up empty price level
        if not level:
            del book_side[price_key]

        return o

    # ------------------------------------------------------------------
    # Book snapshot helpers
    # ------------------------------------------------------------------

    def get_best_bid(self) -> Optional[float]:
        """Highest resting bid price, or None if the bid side is empty."""
        if not self._bids:
            return None
        neg_price = self._bids.keys()[0]   # smallest negative = highest bid
        return -neg_price

    def get_best_ask(self) -> Optional[float]:
        """Lowest resting ask price, or None if the ask side is empty."""
        if not self._asks:
            return None
        return self._asks.keys()[0]

    def get_spread(self) -> Optional[float]:
        """Best ask minus best bid, or None if either side is empty."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is None or ask is None:
            return None
        return round(ask - bid, 6)

    def get_depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        """
        Aggregated order book depth snapshot.

        Returns
        -------
        dict with keys "bids" and "asks", each a list of (price, total_qty)
        tuples, sorted by best price first, up to `levels` price levels deep.
        """
        bids_snap: List[Tuple[float, float]] = []
        for neg_p, q in list(self._bids.items())[:levels]:
            total = sum(o.remaining_qty for o in q)
            bids_snap.append((-neg_p, total))

        asks_snap: List[Tuple[float, float]] = []
        for p, q in list(self._asks.items())[:levels]:
            total = sum(o.remaining_qty for o in q)
            asks_snap.append((p, total))

        return {"bids": bids_snap, "asks": asks_snap}

    def get_order(self, order_id: str) -> Optional[Order]:
        """Return a resting order by ID without removing it, or None."""
        if order_id not in self._order_index:
            return None
        side, price_key = self._order_index[order_id]
        book_side = self._bids if side is OrderSide.BUY else self._asks
        for o in book_side[price_key]:
            if o.order_id == order_id:
                return o
        return None  # pragma: no cover

    # ------------------------------------------------------------------
    # Internal: validation
    # ------------------------------------------------------------------

    def _validate(self, order: Order) -> None:
        if order.quantity <= 0:
            raise InvalidOrderError(
                f"Order quantity must be > 0, got {order.quantity}"
            )
        if order.order_type is OrderType.LIMIT:
            if order.price is None or order.price <= 0:
                raise InvalidOrderError(
                    f"LIMIT order must have a positive price, got {order.price}"
                )

    # ------------------------------------------------------------------
    # Internal: matching engine
    # ------------------------------------------------------------------

    def _match(self, aggressor: Order) -> None:
        """
        Run the matching loop.

        BUY  aggressor → sweeps asks ascending  (lowest ask first).
        SELL aggressor → sweeps bids descending (highest bid first).
        """
        if aggressor.side is OrderSide.BUY:
            opposite_book = self._asks
            price_ok = self._buy_price_ok
        else:
            opposite_book = self._bids
            price_ok = self._sell_price_ok

        while aggressor.is_active and opposite_book:
            best_key = opposite_book.keys()[0]
            best_price = best_key if aggressor.side is OrderSide.BUY else -best_key

            # Price check — market orders always accept any price
            if not price_ok(aggressor, best_price):
                break

            level: PriceLevel = opposite_book[best_key]

            while aggressor.is_active and level:
                resting = level[0]          # FIFO: front of queue

                fill_qty = min(aggressor.remaining_qty, resting.remaining_qty)
                trade_price = resting.price  # maker pricing

                # Execute fills
                aggressor.fill(fill_qty)
                resting.fill(fill_qty)

                # Record trade
                if aggressor.side is OrderSide.BUY:
                    trade = Trade(
                        symbol=self.symbol,
                        buy_order_id=aggressor.order_id,
                        sell_order_id=resting.order_id,
                        price=trade_price,
                        quantity=fill_qty,
                    )
                else:
                    trade = Trade(
                        symbol=self.symbol,
                        buy_order_id=resting.order_id,
                        sell_order_id=aggressor.order_id,
                        price=trade_price,
                        quantity=fill_qty,
                    )

                self.trades.append(trade)
                log.info("TRADE: %s", trade)

                # Remove fully-filled resting order from level and index
                if not resting.is_active:
                    level.popleft()
                    self._order_index.pop(resting.order_id, None)

            # Clean up empty price level
            if not level:
                del opposite_book[best_key]

    # ------------------------------------------------------------------
    # Internal: price eligibility checks
    # ------------------------------------------------------------------

    @staticmethod
    def _buy_price_ok(aggressor: Order, ask_price: float) -> bool:
        """A BUY matches if the limit price >= best ask (or it's a market order)."""
        if aggressor.order_type is OrderType.MARKET:
            return True
        return aggressor.price >= ask_price

    @staticmethod
    def _sell_price_ok(aggressor: Order, bid_price: float) -> bool:
        """A SELL matches if the limit price <= best bid (or it's a market order)."""
        if aggressor.order_type is OrderType.MARKET:
            return True
        return aggressor.price <= bid_price

    # ------------------------------------------------------------------
    # Internal: place a resting order in the book
    # ------------------------------------------------------------------

    def _rest(self, order: Order) -> None:
        """Place an unmatched limit order into the correct price level."""
        if order.side is OrderSide.BUY:
            key = -order.price       # negate so highest bid → smallest key
            book_side = self._bids
        else:
            key = order.price
            book_side = self._asks

        if key not in book_side:
            book_side[key] = deque()

        book_side[key].append(order)
        self._order_index[order.order_id] = (order.side, key)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"OrderBook({self.symbol!r} "
            f"bid={self.get_best_bid()} ask={self.get_best_ask()} "
            f"trades={len(self.trades)})"
        )
