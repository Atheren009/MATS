"""
AlpacaPaperTrader — submit and monitor orders on Alpaca paper trading.

Position sizing: 0.5% risk of $5K equity cap = $25 per trade.
Stop and TP derived from ATR, passed through from the Signal dataclass.

XAUUSD signals are logged but NOT submitted (Alpaca = equities only).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    EQUITY_SYMBOLS,
    MAX_DRAWDOWN_PCT,
    PAPER_EQUITY,
    RISK_PER_TRADE_PCT,
    DATA_DIR,
)
from ..signals.baseline_rules import Signal

log = logging.getLogger(__name__)


class AlpacaPaperTrader:
    """
    Alpaca paper trading wrapper with risk-managed position sizing.

    Parameters
    ----------
    equity_cap : float
        Software-enforced equity limit (default $5K).
        Even though Alpaca paper starts at $100K, this caps exposure.
    risk_pct : float
        Percentage of equity_cap risked per trade (default 0.5%).
    """

    def __init__(
        self,
        equity_cap: float = PAPER_EQUITY,
        risk_pct: float = RISK_PER_TRADE_PCT,
    ) -> None:
        self.equity_cap = equity_cap
        self.risk_pct = risk_pct / 100.0   # 0.5% → 0.005
        self._client = None
        self._trade_log: list[dict] = []
        self._init_client()

    def _init_client(self) -> None:
        try:
            from alpaca.trading.client import TradingClient
            self._client = TradingClient(
                ALPACA_API_KEY,
                ALPACA_SECRET_KEY,
                paper=True,
            )
            log.info("Alpaca trading client initialised (paper mode)")
        except ImportError:
            log.warning(
                "alpaca-py not installed — paper trading disabled. "
                "Install with: pip install alpaca-py"
            )
        except Exception as exc:
            log.warning("Alpaca trading client init failed: %s", exc)

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> Optional[dict]:
        """Return account summary dict."""
        if self._client is None:
            return None
        try:
            acct = self._client.get_account()
            return {
                "equity": float(acct.equity),
                "cash": float(acct.cash),
                "buying_power": float(acct.buying_power),
                "portfolio_value": float(acct.portfolio_value),
            }
        except Exception as exc:
            log.error("Failed to get account: %s", exc)
            return None

    def check_drawdown(self) -> bool:
        """Return True if equity drawdown exceeds MAX_DRAWDOWN_PCT (kill switch)."""
        acct = self.get_account()
        if acct is None:
            return False
        current = acct["equity"]
        dd_pct = (1.0 - current / self.equity_cap) * 100.0
        if dd_pct > MAX_DRAWDOWN_PCT:
            log.warning(
                "KILL SWITCH: Drawdown %.1f%% exceeds limit %.1f%% — halting trades",
                dd_pct, MAX_DRAWDOWN_PCT,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_qty(self, signal: Signal) -> int:
        """
        Calculate share quantity from risk budget.

        risk_dollars = equity_cap × risk_pct
        stop_distance = abs(entry - stop)
        qty = floor(risk_dollars / stop_distance)
        """
        risk_dollars = self.equity_cap * self.risk_pct   # $25 at 5K/0.5%
        stop_distance = abs(signal.entry_price - signal.stop_price)

        if stop_distance <= 0:
            log.warning("  Zero stop distance for %s — skipping", signal.symbol)
            return 0

        qty = int(risk_dollars / stop_distance)
        # Floor at 1 share minimum, but respect notional cap
        if qty < 1:
            log.info("  Qty < 1 share for %s at $%.2f — skipping", signal.symbol, signal.entry_price)
            return 0

        # Notional cap: don't exceed equity_cap on a single position
        max_notional_qty = int(self.equity_cap / signal.entry_price)
        qty = min(qty, max_notional_qty)

        return qty

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_signal(self, signal: Signal, dry_run: bool = False) -> Optional[dict]:
        """
        Convert a Signal into an Alpaca bracket order and submit.

        Parameters
        ----------
        signal : Signal
            From BaselineSignalGenerator.
        dry_run : bool
            If True, log the order but don't submit.

        Returns
        -------
        dict or None
            Order details if submitted, None if skipped.
        """
        # --- Gate: only trade equities on Alpaca ---
        if signal.symbol not in EQUITY_SYMBOLS:
            log.info(
                "  %s is not an Alpaca equity — signal logged, not submitted",
                signal.symbol,
            )
            self._log_trade(signal, qty=0, status="skipped_non_equity")
            return None

        # --- Gate: drawdown kill switch ---
        if self.check_drawdown():
            self._log_trade(signal, qty=0, status="killed_drawdown")
            return None

        # --- Position sizing ---
        qty = self.calculate_qty(signal)
        if qty <= 0:
            self._log_trade(signal, qty=0, status="skipped_zero_qty")
            return None

        order_info = {
            "symbol": signal.symbol,
            "side": "buy" if signal.direction == "long" else "sell",
            "qty": qty,
            "entry_price": signal.entry_price,
            "stop_price": signal.stop_price,
            "tp_price": signal.tp_price,
            "regime": signal.regime,
            "rule": signal.rule,
            "risk_dollars": round(self.equity_cap * self.risk_pct, 2),
        }

        if dry_run:
            log.info("  [DRY RUN] Would submit: %s", json.dumps(order_info))
            self._log_trade(signal, qty=qty, status="dry_run")
            return order_info

        if self._client is None:
            log.warning("  Alpaca client not available — logging only")
            self._log_trade(signal, qty=qty, status="no_client")
            return order_info

        # --- Submit bracket order ---
        try:
            from alpaca.trading.requests import (
                MarketOrderRequest,
                TakeProfitRequest,
                StopLossRequest,
            )
            from alpaca.trading.enums import (
                OrderSide,
                TimeInForce,
                OrderClass,
            )

            side = OrderSide.BUY if signal.direction == "long" else OrderSide.SELL

            request = MarketOrderRequest(
                symbol=signal.symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(
                    limit_price=round(signal.tp_price, 2)
                ),
                stop_loss=StopLossRequest(
                    stop_price=round(signal.stop_price, 2)
                ),
            )

            order = self._client.submit_order(request)

            log.info(
                "  ORDER SUBMITTED: %s %s %d × %s @ market  "
                "SL=%.2f  TP=%.2f  ID=%s",
                signal.direction.upper(), signal.symbol, qty,
                signal.symbol, signal.stop_price, signal.tp_price,
                order.id,
            )

            self._log_trade(
                signal, qty=qty, status="submitted", order_id=str(order.id)
            )
            return {**order_info, "order_id": str(order.id)}

        except Exception as exc:
            log.error("  Order submission failed: %s", exc)
            self._log_trade(signal, qty=qty, status=f"error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def get_positions(self) -> list[dict]:
        """Return current open positions."""
        if self._client is None:
            return []
        try:
            positions = self._client.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "side": p.side,
                    "avg_entry": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "unrealised_pl": float(p.unrealized_pl),
                    "unrealised_pl_pct": float(p.unrealized_plpc) * 100,
                }
                for p in positions
            ]
        except Exception as exc:
            log.error("Failed to get positions: %s", exc)
            return []

    def close_all(self) -> None:
        """Close all open positions (emergency)."""
        if self._client is None:
            return
        try:
            self._client.close_all_positions(cancel_orders=True)
            log.warning("ALL POSITIONS CLOSED")
        except Exception as exc:
            log.error("Failed to close positions: %s", exc)

    # ------------------------------------------------------------------
    # Trade log
    # ------------------------------------------------------------------

    def _log_trade(
        self,
        signal: Signal,
        qty: int,
        status: str,
        order_id: Optional[str] = None,
    ) -> None:
        """Append to in-memory trade log."""
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "signal_timestamp": signal.timestamp,
            "symbol": signal.symbol,
            "direction": signal.direction,
            "qty": qty,
            "entry_price": signal.entry_price,
            "stop_price": signal.stop_price,
            "tp_price": signal.tp_price,
            "regime": signal.regime,
            "rule": signal.rule,
            "status": status,
            "order_id": order_id,
        }
        self._trade_log.append(record)

    def save_trade_log(self, path: Optional[Path] = None) -> Path:
        """Save the trade log to CSV."""
        if path is None:
            path = DATA_DIR / "trade_log.csv"
        df = pd.DataFrame(self._trade_log)
        if path.exists():
            existing = pd.read_csv(path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(path, index=False)
        log.info("  Trade log saved → %s (%d records)", path, len(df))
        return path

    @property
    def trade_count(self) -> int:
        return len(self._trade_log)
