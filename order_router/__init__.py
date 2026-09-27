"""
order_router — limit order book, matching engine & multi-exchange routing
"""

# Phase 2: core matching engine
from .models import Order, OrderSide, OrderType, OrderStatus, Trade
from .order_book import OrderBook
from .price_feed import PriceFeed, Bar
from .exceptions import OrderNotFoundError, InvalidOrderError

# Phase 3: routing engine
from .child_order import ChildOrder, FillResult
from .exchange import Exchange, VenueConfig, VENUE_CONFIGS
from .venue_registry import VenueRegistry
from .router import OrderRouter, RouterResult
from .routing import (
    RoutingStrategy,
    BestPriceStrategy,
    TWAPStrategy,
    VWAPStrategy,
    SmartStrategy,
)

__all__ = [
    # Phase 2
    "Order", "OrderSide", "OrderType", "OrderStatus", "Trade",
    "OrderBook",
    "PriceFeed", "Bar",
    "OrderNotFoundError", "InvalidOrderError",
    # Phase 3
    "ChildOrder", "FillResult",
    "Exchange", "VenueConfig", "VENUE_CONFIGS",
    "VenueRegistry",
    "OrderRouter", "RouterResult",
    "RoutingStrategy", "BestPriceStrategy",
    "TWAPStrategy", "VWAPStrategy", "SmartStrategy",
]

