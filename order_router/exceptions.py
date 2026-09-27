"""
Custom exceptions for the order book and matching engine.
"""


class OrderNotFoundError(Exception):
    """Raised when an order ID cannot be located in the book (e.g. during cancel)."""

    def __init__(self, order_id: str):
        super().__init__(f"Order '{order_id}' not found in the order book.")
        self.order_id = order_id


class InvalidOrderError(Exception):
    """Raised when an order has invalid parameters (e.g. non-positive qty or price)."""

    def __init__(self, message: str):
        super().__init__(message)
