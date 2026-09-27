"""
venue_registry.py — Factory and registry for the three simulated exchanges.

VenueRegistry owns canonical Exchange instances and provides helpers for
seeding all venues from a bar and creating independent fresh copies (used
by the comparator to give each strategy a clean slate).
"""

from __future__ import annotations

from typing import Dict

from .exchange import Exchange, VenueConfig, VENUE_CONFIGS
from .price_feed import Bar


class VenueRegistry:
    """
    Manages the three canonical simulated exchanges: ALPHA, BETA, GAMMA.

    Usage
    -----
    >>> registry = VenueRegistry()
    >>> registry.seed_all(bar)          # populate books from a price bar
    >>> venues = registry.get_all()     # Dict[str, Exchange] for strategies
    >>> fresh   = registry.fresh_venues()  # independent copies, same config
    """

    def __init__(self, configs: Dict[str, VenueConfig] = None) -> None:
        self._configs = configs or VENUE_CONFIGS
        self._venues: Dict[str, Exchange] = {
            name: Exchange(name, cfg)
            for name, cfg in self._configs.items()
        }

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, name: str) -> Exchange:
        """Return the exchange by name. Raises KeyError for unknown names."""
        return self._venues[name]

    def get_all(self) -> Dict[str, Exchange]:
        """Return all exchanges as a dict (name → Exchange)."""
        return dict(self._venues)

    @property
    def names(self):
        return list(self._venues.keys())

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def seed_all(self, bar: Bar) -> None:
        """Seed every exchange from the same bar (call once per time step)."""
        for exchange in self._venues.values():
            exchange.seed_from_bar(bar)

    def reset_all(self) -> None:
        """Clear all books without re-seeding (between simulation runs)."""
        for exchange in self._venues.values():
            exchange.reset()

    def fresh_venues(self) -> Dict[str, Exchange]:
        """
        Return a new set of Exchange instances with the same configs but
        empty books.  Use this to give each routing strategy a clean slate
        during comparisons.
        """
        return {
            name: Exchange(name, cfg)
            for name, cfg in self._configs.items()
        }

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"VenueRegistry(venues={self.names})"
