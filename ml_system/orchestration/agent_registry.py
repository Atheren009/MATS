"""
AgentRegistry -- extends retrain_log with multi-agent fields.

New tables
----------
agent_registry
    agent_id            TEXT  PK   -- e.g. "DIA_v1", "QQQ_v1"
    symbol              TEXT
    created_utc         TEXT
    lifecycle_status    TEXT       -- pending | backtested | paper_live |
                                   --   funded | rejected
    oos_return_series   TEXT       -- path to CSV with daily OOS returns
    dsr_at_backtest     REAL
    spa_p_at_backtest   REAL
    global_trial_count  INTEGER    -- n_trials used in DSR call (never reset)
    correlation_at_fund REAL       -- worst pairwise corr when funded
    funded_utc          TEXT
    rejected_reason     TEXT
    notes               TEXT

global_trial_counter          -- single-row counter, atomic increment
    id                  INTEGER  PK DEFAULT 1
    trial_count         INTEGER  DEFAULT 0
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

DB_PATH = REGISTRY_DIR / "agent_registry.db"

_DDL = """
CREATE TABLE IF NOT EXISTS agent_registry (
    agent_id            TEXT PRIMARY KEY,
    symbol              TEXT NOT NULL,
    created_utc         TEXT NOT NULL,
    lifecycle_status    TEXT NOT NULL DEFAULT 'pending',
    oos_return_series   TEXT,
    dsr_at_backtest     REAL,
    spa_p_at_backtest   REAL,
    global_trial_count  INTEGER,
    correlation_at_fund REAL,
    funded_utc          TEXT,
    rejected_reason     TEXT,
    notes               TEXT
);

CREATE TABLE IF NOT EXISTS global_trial_counter (
    id          INTEGER PRIMARY KEY DEFAULT 1,
    trial_count INTEGER NOT NULL DEFAULT 0
);

-- Seed the counter if the table is empty
INSERT OR IGNORE INTO global_trial_counter (id, trial_count) VALUES (1, 0);
"""


class AgentRegistry:
    """
    SQLite-backed registry for every agent that has entered the lifecycle.

    global_trial_count increments every time any agent completes a backtest
    DSR call.  It is never reset per agent -- this is the Deflated Sharpe
    correction for multiple testing across the system lifetime.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_DDL)
            conn.commit()
        log.debug("AgentRegistry DB ready: %s", self.db_path)

    # ------------------------------------------------------------------
    # global_trial_count -- the system-wide DSR trial counter
    # ------------------------------------------------------------------

    def increment_trial_count(self) -> int:
        """
        Atomically increment and return the new global trial count.
        Call once per completed backtest DSR evaluation, regardless of result.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE global_trial_counter SET trial_count = trial_count + 1 WHERE id = 1"
            )
            conn.commit()
            row = conn.execute(
                "SELECT trial_count FROM global_trial_counter WHERE id = 1"
            ).fetchone()
        count = row[0]
        log.info("Global trial count incremented to %d", count)
        return count

    def get_trial_count(self) -> int:
        """Return the current global trial count without incrementing."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT trial_count FROM global_trial_counter WHERE id = 1"
            ).fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Agent CRUD
    # ------------------------------------------------------------------

    def register_agent(
        self,
        symbol: str,
        agent_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> str:
        """Create an agent entry in pending status.  Returns agent_id."""
        aid = agent_id or f"{symbol}_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO agent_registry
                   (agent_id, symbol, created_utc, lifecycle_status, notes)
                   VALUES (?, ?, ?, 'pending', ?)""",
                (aid, symbol, now, notes),
            )
            conn.commit()
        log.info("Registered agent %s for %s", aid, symbol)
        return aid

    def update_backtest_result(
        self,
        agent_id: str,
        dsr: float,
        spa_p: float,
        oos_return_series: Optional[str] = None,
    ) -> int:
        """
        Record backtest stats and increment global trial count.
        Returns the new global_trial_count used in this DSR call.
        """
        n_trials = self.increment_trial_count()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE agent_registry SET
                    lifecycle_status  = 'backtested',
                    dsr_at_backtest   = ?,
                    spa_p_at_backtest = ?,
                    global_trial_count = ?,
                    oos_return_series  = ?
                   WHERE agent_id = ?""",
                (dsr, spa_p, n_trials, oos_return_series, agent_id),
            )
            conn.commit()
        log.info(
            "Agent %s backtest: DSR=%.4f SPA_p=%.4f (n_trials=%d)",
            agent_id, dsr, spa_p, n_trials,
        )
        return n_trials

    def set_paper_live(self, agent_id: str) -> None:
        """Mark agent as in paper-live."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE agent_registry SET lifecycle_status='paper_live' WHERE agent_id=?",
                (agent_id,),
            )
            conn.commit()
        log.info("Agent %s -> paper_live", agent_id)

    def fund_agent(self, agent_id: str, correlation_at_fund: float) -> None:
        """Fund an agent after it passes the correlation gate."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE agent_registry SET
                    lifecycle_status    = 'funded',
                    correlation_at_fund = ?,
                    funded_utc          = ?
                   WHERE agent_id = ?""",
                (correlation_at_fund, now, agent_id),
            )
            conn.commit()
        log.info("Agent %s FUNDED (worst corr=%.3f)", agent_id, correlation_at_fund)

    def reject_agent(self, agent_id: str, reason: str) -> None:
        """Log a rejection at any lifecycle stage.  Never silently dropped."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE agent_registry SET
                    lifecycle_status = 'rejected',
                    rejected_reason  = ?
                   WHERE agent_id = ?""",
                (reason, agent_id),
            )
            conn.commit()
        log.warning("Agent %s REJECTED: %s", agent_id, reason)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def funded_agents(self) -> list[dict]:
        """Return all currently-funded agents."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM agent_registry WHERE lifecycle_status='funded'"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_agent(self, agent_id: str) -> Optional[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM agent_registry WHERE agent_id=?", (agent_id,)
            ).fetchone()
        return dict(row) if row else None

    def all_agents(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM agent_registry ORDER BY created_utc DESC"
            ).fetchall()
        return [dict(r) for r in rows]
