"""
test_order_book.py — 14 test cases for the OrderBook matching engine.

The test suite uses real AAPL 1-minute bar data (AAPL_1min_2024-2026.csv) as
a price feed to generate realistic bid/ask prices.  A small module-level
fixture loads the first N bars once; individual tests derive prices from
bar.mid_price so the numbers are always grounded in actual market data.

Test catalogue
--------------
 1. test_add_limit_order_rests               — unmatched limit rests in book
 2. test_simple_full_match                   — buy & sell fully match
 3. test_price_time_priority_fifo            — two asks at same price → FIFO
 4. test_partial_fill_buy_larger             — large buy partially fills
 5. test_partial_fill_sell_larger            — large sell partially fills
 6. test_sweep_multiple_ask_levels           — buy sweeps across price levels
 7. test_market_order_buy                    — market buy hits best ask
 8. test_market_order_sell                   — market sell hits best bid
 9. test_cancel_resting_order               — cancel removes order from book
10. test_cancel_nonexistent_raises           — cancel bad ID → OrderNotFoundError
11. test_no_cross_no_trade                   — bid below ask → no match
12. test_get_depth_snapshot                  — depth aggregates qty correctly
13. test_spread_calculation                  — spread = ask − bid
14. test_market_order_empty_book_unfilled    — market on empty book stays open
15. test_trade_records_maker_price           — trade price equals resting price
16. test_price_feed_integration              — feed produces real AAPL bars
"""

from __future__ import annotations

import os
import pytest

from order_router.models import Order, OrderSide, OrderStatus, OrderType
from order_router.order_book import OrderBook
from order_router.exceptions import OrderNotFoundError, InvalidOrderError
from order_router.price_feed import PriceFeed


# ---------------------------------------------------------------------------
# Module-level price feed fixture
# ---------------------------------------------------------------------------

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "AAPL_1min_2024-2026.csv")


@pytest.fixture(scope="module")
def feed() -> PriceFeed:
    """Load the first 20 bars from the real AAPL CSV once per module."""
    pf = PriceFeed(CSV_PATH)
    return pf


@pytest.fixture(scope="module")
def sample_bars(feed: PriceFeed):
    """Return the first 10 bars as a list."""
    return list(feed)[:10]


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def mk_buy(price: float, qty: float = 100, symbol: str = "AAPL") -> Order:
    return Order(symbol, OrderSide.BUY, OrderType.LIMIT, quantity=qty, price=price)


def mk_sell(price: float, qty: float = 100, symbol: str = "AAPL") -> Order:
    return Order(symbol, OrderSide.SELL, OrderType.LIMIT, quantity=qty, price=price)


def mk_market_buy(qty: float = 100, symbol: str = "AAPL") -> Order:
    return Order(symbol, OrderSide.BUY, OrderType.MARKET, quantity=qty)


def mk_market_sell(qty: float = 100, symbol: str = "AAPL") -> Order:
    return Order(symbol, OrderSide.SELL, OrderType.MARKET, quantity=qty)


# ---------------------------------------------------------------------------
# TEST 1 — Unmatched limit order rests in the book
# ---------------------------------------------------------------------------

def test_add_limit_order_rests(sample_bars):
    """A limit buy with no opposing sell should rest in the book."""
    bar = sample_bars[0]
    book = OrderBook("AAPL")

    buy = mk_buy(price=bar.mid_price - 1.00)   # bid well below market → no fill
    book.add_order(buy)

    assert book.get_best_bid() == pytest.approx(buy.price)
    assert buy.status == OrderStatus.OPEN
    assert len(book.trades) == 0


# ---------------------------------------------------------------------------
# TEST 2 — Simple full match
# ---------------------------------------------------------------------------

def test_simple_full_match(sample_bars):
    """One buy and one sell at crossing prices → single full trade."""
    bar = sample_bars[0]
    mid = bar.mid_price

    book = OrderBook("AAPL")
    sell = mk_sell(price=mid - 0.10, qty=50)   # ask rests first
    buy  = mk_buy( price=mid + 0.10, qty=50)   # bid crosses the ask

    book.add_order(sell)
    book.add_order(buy)

    assert len(book.trades) == 1
    trade = book.trades[0]
    assert trade.quantity == 50
    assert trade.price == pytest.approx(sell.price)   # maker price
    assert buy.status  == OrderStatus.FILLED
    assert sell.status == OrderStatus.FILLED
    # Book should now be empty on both sides
    assert book.get_best_bid() is None
    assert book.get_best_ask() is None


# ---------------------------------------------------------------------------
# TEST 3 — Price-time priority (FIFO at same price level)
# ---------------------------------------------------------------------------

def test_price_time_priority_fifo(sample_bars):
    """Two sells at the same price — first-in should fill first."""
    bar = sample_bars[1]
    ask_price = round(bar.mid_price, 2)

    book = OrderBook("AAPL")
    sell1 = mk_sell(price=ask_price, qty=30)
    sell2 = mk_sell(price=ask_price, qty=30)
    book.add_order(sell1)
    book.add_order(sell2)

    # A buy that only covers sell1
    buy = mk_buy(price=ask_price + 0.50, qty=30)
    book.add_order(buy)

    assert len(book.trades) == 1
    assert book.trades[0].sell_order_id == sell1.order_id   # sell1 filled first
    assert sell1.status == OrderStatus.FILLED
    assert sell2.status == OrderStatus.OPEN   # still resting


# ---------------------------------------------------------------------------
# TEST 4 — Partial fill: large buy against smaller sell
# ---------------------------------------------------------------------------

def test_partial_fill_buy_larger(sample_bars):
    """Buy 200, sell only 80 → buy is partially filled."""
    bar = sample_bars[2]
    mid = bar.mid_price

    book = OrderBook("AAPL")
    sell = mk_sell(price=mid, qty=80)
    buy  = mk_buy( price=mid, qty=200)

    book.add_order(sell)
    book.add_order(buy)

    assert len(book.trades) == 1
    assert book.trades[0].quantity == 80
    assert sell.status == OrderStatus.FILLED
    assert buy.status  == OrderStatus.PARTIALLY_FILLED
    assert buy.filled_qty == 80
    assert buy.remaining_qty == 120
    # Remaining 120 should rest in the book
    assert book.get_best_bid() == pytest.approx(mid)


# ---------------------------------------------------------------------------
# TEST 5 — Partial fill: large sell against smaller buy
# ---------------------------------------------------------------------------

def test_partial_fill_sell_larger(sample_bars):
    """Sell 200, buy only 60 → sell is partially filled."""
    bar = sample_bars[3]
    mid = bar.mid_price

    book = OrderBook("AAPL")
    buy  = mk_buy( price=mid, qty=60)
    sell = mk_sell(price=mid, qty=200)

    book.add_order(buy)
    book.add_order(sell)

    assert sell.filled_qty == 60
    assert sell.status == OrderStatus.PARTIALLY_FILLED
    assert sell.remaining_qty == 140
    assert book.get_best_ask() == pytest.approx(mid)


# ---------------------------------------------------------------------------
# TEST 6 — Sweep multiple ask price levels
# ---------------------------------------------------------------------------

def test_sweep_multiple_ask_levels(sample_bars):
    """An aggressive buy sweeps through two separate ask price levels."""
    bar = sample_bars[4]
    base = round(bar.mid_price, 2)

    book = OrderBook("AAPL")
    sell_low  = mk_sell(price=base,        qty=50)   # level 1
    sell_high = mk_sell(price=base + 0.50, qty=50)   # level 2
    book.add_order(sell_low)
    book.add_order(sell_high)

    # Buy with a limit that crosses both levels and wants 100 shares
    buy = mk_buy(price=base + 1.00, qty=100)
    book.add_order(buy)

    assert len(book.trades) == 2
    assert book.trades[0].price == pytest.approx(base)          # cheaper fill first
    assert book.trades[1].price == pytest.approx(base + 0.50)   # then next level
    assert buy.status == OrderStatus.FILLED
    assert book.get_best_ask() is None   # book is clear


# ---------------------------------------------------------------------------
# TEST 7 — Market buy
# ---------------------------------------------------------------------------

def test_market_order_buy(sample_bars):
    """Market buy fills at the best available ask regardless of price."""
    bar = sample_bars[5]
    ask = round(bar.mid_price + 0.25, 2)

    book = OrderBook("AAPL")
    sell = mk_sell(price=ask, qty=100)
    book.add_order(sell)

    mkt = mk_market_buy(qty=100)
    book.add_order(mkt)

    assert len(book.trades) == 1
    assert mkt.status == OrderStatus.FILLED
    assert book.trades[0].price == pytest.approx(ask)


# ---------------------------------------------------------------------------
# TEST 8 — Market sell
# ---------------------------------------------------------------------------

def test_market_order_sell(sample_bars):
    """Market sell fills at the best available bid regardless of price."""
    bar = sample_bars[6]
    bid = round(bar.mid_price - 0.25, 2)

    book = OrderBook("AAPL")
    buy = mk_buy(price=bid, qty=100)
    book.add_order(buy)

    mkt = mk_market_sell(qty=100)
    book.add_order(mkt)

    assert len(book.trades) == 1
    assert mkt.status == OrderStatus.FILLED
    assert book.trades[0].price == pytest.approx(bid)


# ---------------------------------------------------------------------------
# TEST 9 — Cancel a resting order
# ---------------------------------------------------------------------------

def test_cancel_resting_order(sample_bars):
    """Cancelling a resting order removes it from the book and sets CANCELLED status."""
    bar = sample_bars[7]
    mid = bar.mid_price

    book = OrderBook("AAPL")
    buy = mk_buy(price=mid - 2.00)
    book.add_order(buy)

    assert book.get_best_bid() == pytest.approx(mid - 2.00)

    cancelled = book.cancel_order(buy.order_id)
    assert cancelled.status == OrderStatus.CANCELLED
    assert book.get_best_bid() is None   # level removed


# ---------------------------------------------------------------------------
# TEST 10 — Cancel non-existent order raises OrderNotFoundError
# ---------------------------------------------------------------------------

def test_cancel_nonexistent_raises():
    """Cancelling an unknown order ID raises OrderNotFoundError."""
    book = OrderBook("AAPL")
    with pytest.raises(OrderNotFoundError):
        book.cancel_order("non-existent-uuid")


# ---------------------------------------------------------------------------
# TEST 11 — No cross → no trade
# ---------------------------------------------------------------------------

def test_no_cross_no_trade(sample_bars):
    """A buy below the ask should rest without triggering any trade."""
    bar = sample_bars[8]
    mid = bar.mid_price

    book = OrderBook("AAPL")
    sell = mk_sell(price=mid + 1.00)
    buy  = mk_buy( price=mid - 1.00)   # 2.00 spread — no cross

    book.add_order(sell)
    book.add_order(buy)

    assert len(book.trades) == 0
    assert book.get_best_bid() == pytest.approx(mid - 1.00)
    assert book.get_best_ask() == pytest.approx(mid + 1.00)
    assert book.get_spread()   == pytest.approx(2.00, abs=1e-4)


# ---------------------------------------------------------------------------
# TEST 12 — Depth snapshot
# ---------------------------------------------------------------------------

def test_get_depth_snapshot(sample_bars):
    """Depth aggregates multiple orders at the same price level correctly."""
    bar = sample_bars[9]
    base = round(bar.mid_price, 2)

    book = OrderBook("AAPL")
    # Two bids at the same price
    book.add_order(mk_buy(price=base - 0.50, qty=40))
    book.add_order(mk_buy(price=base - 0.50, qty=60))
    # One bid at a worse price
    book.add_order(mk_buy(price=base - 1.00, qty=100))
    # One ask
    book.add_order(mk_sell(price=base + 0.50, qty=200))

    depth = book.get_depth(levels=5)
    bids = depth["bids"]
    asks = depth["asks"]

    # Best bid level: two orders totalling 100 shares
    assert bids[0][0] == pytest.approx(base - 0.50)
    assert bids[0][1] == pytest.approx(100)
    # Second bid level
    assert bids[1][0] == pytest.approx(base - 1.00)
    assert bids[1][1] == pytest.approx(100)
    # Ask level
    assert asks[0][0] == pytest.approx(base + 0.50)
    assert asks[0][1] == pytest.approx(200)


# ---------------------------------------------------------------------------
# TEST 13 — Spread calculation
# ---------------------------------------------------------------------------

def test_spread_calculation(sample_bars):
    """Spread equals best ask minus best bid."""
    bar = sample_bars[0]
    mid = round(bar.mid_price, 2)

    book = OrderBook("AAPL")
    book.add_order(mk_buy( price=mid - 0.30))
    book.add_order(mk_sell(price=mid + 0.70))

    expected_spread = round((mid + 0.70) - (mid - 0.30), 6)
    assert book.get_spread() == pytest.approx(expected_spread, abs=1e-4)


# ---------------------------------------------------------------------------
# TEST 14 — Market order on empty book stays unfilled
# ---------------------------------------------------------------------------

def test_market_order_empty_book_unfilled():
    """A market order submitted to an empty book remains open/unfilled."""
    book = OrderBook("AAPL")
    mkt = mk_market_buy(qty=50)
    book.add_order(mkt)

    assert len(book.trades) == 0
    # Market order not placed in book — but status stays OPEN (not filled, not cancelled)
    assert mkt.status == OrderStatus.OPEN
    assert mkt.filled_qty == 0


# ---------------------------------------------------------------------------
# TEST 15 — Trade records the resting (maker) price
# ---------------------------------------------------------------------------

def test_trade_records_maker_price(sample_bars):
    """Trade price must equal the resting order's price (maker pricing)."""
    bar = sample_bars[0]
    mid = bar.mid_price

    resting_price = round(mid - 0.10, 2)   # ask rests
    aggressor_limit = round(mid + 0.50, 2) # buy crosses aggressively

    book = OrderBook("AAPL")
    sell = mk_sell(price=resting_price, qty=100)
    buy  = mk_buy( price=aggressor_limit, qty=100)

    book.add_order(sell)   # sell rests
    book.add_order(buy)    # buy aggresses

    assert len(book.trades) == 1
    assert book.trades[0].price == pytest.approx(resting_price)   # NOT aggressor_limit


# ---------------------------------------------------------------------------
# TEST 16 — Price feed integration
# ---------------------------------------------------------------------------

def test_price_feed_integration(feed: PriceFeed):
    """PriceFeed loads real AAPL data and provides usable bar attributes."""
    assert len(feed) > 1000, "Expected many bars in the 2024-2026 dataset"

    feed.reset()
    bar = feed.next_bar()
    assert bar is not None
    assert bar.symbol == "AAPL"
    assert bar.open > 0
    assert bar.high >= bar.low
    assert bar.volume > 0
    assert 0 < bar.mid_price < 1000   # sanity: AAPL is never $0 or $1000

    # Simulate a simple one-bar matching scenario using the real price
    book = OrderBook("AAPL")
    ask = round(bar.vwap + 0.05, 2)   # slightly above VWAP → resting ask
    bid = round(bar.vwap + 0.10, 2)   # willing to pay a bit more → crosses

    sell = mk_sell(price=ask, qty=bar.trade_count)
    buy  = mk_buy( price=bid, qty=bar.trade_count)

    book.add_order(sell)
    book.add_order(buy)

    assert len(book.trades) == 1
    assert book.trades[0].price == pytest.approx(ask)
    assert book.trades[0].quantity == pytest.approx(bar.trade_count)


# ---------------------------------------------------------------------------
# TEST 17 (bonus) — Invalid order raises InvalidOrderError
# ---------------------------------------------------------------------------

def test_invalid_order_zero_quantity():
    """Zero-quantity order should raise InvalidOrderError immediately."""
    book = OrderBook("AAPL")
    bad  = Order("AAPL", OrderSide.BUY, OrderType.LIMIT, quantity=0, price=190.0)
    with pytest.raises(InvalidOrderError):
        book.add_order(bad)


def test_invalid_limit_order_no_price():
    """LIMIT order with price=None should raise InvalidOrderError."""
    book = OrderBook("AAPL")
    bad  = Order("AAPL", OrderSide.BUY, OrderType.LIMIT, quantity=10, price=None)
    with pytest.raises(InvalidOrderError):
        book.add_order(bad)
