"""
RetrainLog — SQLite-backed registry for every HMM/model training run.

Logs training window, regime distribution, feature set hash, promotion
decision, and reason.  Initialised in Phase 0 and never reset.

Schema matches the JSON spec from the design doc.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import REGISTRY_DIR

log = logging.getLogger(__name__)

DB_PATH = REGISTRY_DIR / "retrain.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS retrain_log (
    retrain_id          TEXT PRIMARY KEY,
    timestamp_utc       TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    model_type          TEXT NOT NULL DEFAULT 'hmm',

    trigger_type        TEXT,
    trigger_metric      TEXT,
    trigger_threshold   REAL,
    trigger_observed    REAL,

    train_start         TEXT,
    train_end           TEXT,
    train_bars          INTEGER,

    regime_distribution TEXT,
    feature_set_hash    TEXT,

    log_likelihood      REAL,
    n_states            INTEGER,

    promotion_decision  TEXT,
    rejection_reason    TEXT,

    walk_forward_folds  TEXT,
    challenger_sharpe   REAL,
    baseline_sharpe     REAL,

    model_path          TEXT,
    notes               TEXT
);
"""

_CREATE_TRADE_SUMMARY = """
CREATE TABLE IF NOT EXISTS trade_summary (
    summary_id      TEXT PRIMARY KEY,
    timestamp_utc   TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    run_type        TEXT NOT NULL DEFAULT 'forward_test',

    total_signals   INTEGER,
    long_signals    INTEGER,
    short_signals   INTEGER,

    regime_counts   TEXT,
    timeframe       TEXT,
    data_start      TEXT,
    data_end        TEXT,
    total_bars      INTEGER,

    notes           TEXT
);
"""


class RetrainLog:
    """
    SQLite-backed model training and retrain registry.

    Every fit(), retrain, and promotion attempt is logged here
    from Phase 0 onwards.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_TRADE_SUMMARY)
            conn.commit()
        log.info("Registry DB ready: %s", self.db_path)

    # ------------------------------------------------------------------
    # Log a training run
    # ------------------------------------------------------------------

    def log_training_run(
        self,
        symbol: str,
        model_type: str = "hmm",
        trigger_type: str = "initial_fit",
        trigger_metric: Optional[str] = None,
        trigger_threshold: Optional[float] = None,
        trigger_observed: Optional[float] = None,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None,
        train_bars: Optional[int] = None,
        regime_distribution: Optional[dict] = None,
        feature_set_hash: Optional[str] = None,
        log_likelihood: Optional[float] = None,
        n_states: Optional[int] = None,
        promotion_decision: str = "baseline",
        rejection_reason: Optional[str] = None,
        walk_forward_folds: Optional[list] = None,
        challenger_sharpe: Optional[float] = None,
        baseline_sharpe: Optional[float] = None,
        model_path: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> str:
        """Log one training run.  Returns the retrain_id."""
        retrain_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        row = (
            retrain_id,
            timestamp,
            symbol,
            model_type,
            trigger_type,
            trigger_metric,
            trigger_threshold,
            trigger_observed,
            train_start,
            train_end,
            train_bars,
            json.dumps(regime_distribution) if regime_distribution else None,
            feature_set_hash,
            log_likelihood,
            n_states,
            promotion_decision,
            rejection_reason,
            json.dumps(walk_forward_folds) if walk_forward_folds else None,
            challenger_sharpe,
            baseline_sharpe,
            model_path,
            notes,
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO retrain_log (
                    retrain_id, timestamp_utc, symbol, model_type,
                    trigger_type, trigger_metric, trigger_threshold, trigger_observed,
                    train_start, train_end, train_bars,
                    regime_distribution, feature_set_hash,
                    log_likelihood, n_states,
                    promotion_decision, rejection_reason,
                    walk_forward_folds, challenger_sharpe, baseline_sharpe,
                    model_path, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                row,
            )
            conn.commit()

        log.info(
            "  Registry: logged %s run for %s (id=%s)",
            model_type, symbol, retrain_id[:8],
        )
        return retrain_id

    # ------------------------------------------------------------------
    # Log a forward-test trade summary
    # ------------------------------------------------------------------

    def log_trade_summary(
        self,
        symbol: str,
        total_signals: int,
        long_signals: int,
        short_signals: int,
        regime_counts: Optional[dict] = None,
        timeframe: str = "15min",
        data_start: Optional[str] = None,
        data_end: Optional[str] = None,
        total_bars: int = 0,
        notes: Optional[str] = None,
    ) -> str:
        """Log a trade summary for one symbol run."""
        summary_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO trade_summary (
                    summary_id, timestamp_utc, symbol, run_type,
                    total_signals, long_signals, short_signals,
                    regime_counts, timeframe, data_start, data_end,
                    total_bars, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    summary_id, timestamp, symbol, "forward_test",
                    total_signals, long_signals, short_signals,
                    json.dumps(regime_counts) if regime_counts else None,
                    timeframe, data_start, data_end, total_bars, notes,
                ),
            )
            conn.commit()

        return summary_id

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_history(
        self, symbol: Optional[str] = None, limit: int = 50
    ) -> list[dict]:
        """Return recent retrain log entries."""
        query = "SELECT * FROM retrain_log"
        params: list = []
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)
        query += " ORDER BY timestamp_utc DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def last_run(self, symbol: str) -> Optional[dict]:
        """Return the most recent training run for a symbol."""
        history = self.get_history(symbol, limit=1)
        return history[0] if history else None

    def run_count(self, symbol: Optional[str] = None) -> int:
        """Count total training runs."""
        query = "SELECT COUNT(*) FROM retrain_log"
        params: list = []
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)

        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(query, params).fetchone()[0]
