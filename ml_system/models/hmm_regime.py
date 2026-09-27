"""
RegimeHMM — Gaussian Hidden Markov Model for market regime classification.

Classifies each bar into one of 4 latent states:
    - trending:         Strong directional move, high abs(mean_return)
    - ranging:          Low volatility, near-zero drift
    - high_vol_breakout: Highest volatility, expansion bars
    - low_liquidity:    Lowest activity / volume (pre-market, Asian session)

State labels are assigned automatically after fitting by inspecting the
emission parameters (mean return, volatility per state).
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..config import (
    HMM_COVARIANCE,
    HMM_FEATURES,
    HMM_N_ITER,
    HMM_N_STATES,
    HMM_RANDOM_STATE,
    MODEL_DIR,
)

log = logging.getLogger(__name__)


# Regime label constants
TRENDING = "trending"
RANGING = "ranging"
HIGH_VOL_BREAKOUT = "high_vol_breakout"
LOW_LIQUIDITY = "low_liquidity"

ALL_REGIMES = [TRENDING, RANGING, HIGH_VOL_BREAKOUT, LOW_LIQUIDITY]
NO_TRADE_REGIMES = {HIGH_VOL_BREAKOUT, LOW_LIQUIDITY}


class RegimeHMM:
    """
    Gaussian HMM for market regime classification.

    Parameters
    ----------
    n_states : int
        Number of hidden states (default 4).
    symbol : str
        Symbol this model is fitted for (used in save/load naming).
    """

    def __init__(
        self,
        n_states: int = HMM_N_STATES,
        symbol: str = "UNKNOWN",
    ) -> None:
        self.n_states = n_states
        self.symbol = symbol
        self.feature_cols = list(HMM_FEATURES)

        # Lazy import — hmmlearn is a heavy dependency
        from hmmlearn.hmm import GaussianHMM

        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type=HMM_COVARIANCE,
            n_iter=HMM_N_ITER,
            random_state=HMM_RANDOM_STATE,
            verbose=False,
        )

        self._scaler = StandardScaler()
        self._fitted = False
        self._state_labels: dict[int, str] = {}
        self._state_stats: Optional[pd.DataFrame] = None
        self._feature_hash: Optional[str] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "RegimeHMM":
        """
        Fit the HMM on a feature DataFrame.

        Parameters
        ----------
        df : DataFrame
            Must contain columns listed in HMM_FEATURES.
            Produced by DataLoader.compute_features().
        """
        self._validate_features(df)
        X = df[self.feature_cols].values.copy()

        # Hash the feature config for provenance tracking
        self._feature_hash = hashlib.sha256(
            json.dumps(self.feature_cols, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Standardise
        X_scaled = self._scaler.fit_transform(X)

        # Replace any remaining inf/nan with 0 (safety net)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        log.info(
            "Fitting HMM (%d states) on %d bars for %s",
            self.n_states, len(X_scaled), self.symbol,
        )

        self.model.fit(X_scaled)
        self._fitted = True

        # Auto-label states
        self._label_states(df, X_scaled)

        log.info("  HMM fitted.  State labels: %s", self._state_labels)
        log.info("  Log-likelihood: %.2f", self.model.score(X_scaled))

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict regime label for each bar.

        Returns a Series with the same index as *df*, values are
        string labels from ALL_REGIMES.
        """
        if not self._fitted:
            raise RuntimeError("HMM not fitted — call .fit() first")

        self._validate_features(df)
        X = df[self.feature_cols].values.copy()
        X_scaled = self._scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        raw_states = self.model.predict(X_scaled)
        labels = pd.Series(
            [self._state_labels.get(s, f"state_{s}") for s in raw_states],
            index=df.index,
            name="regime",
        )
        return labels

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return posterior probabilities for each state."""
        if not self._fitted:
            raise RuntimeError("HMM not fitted — call .fit() first")

        self._validate_features(df)
        X = df[self.feature_cols].values.copy()
        X_scaled = self._scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        probs = self.model.predict_proba(X_scaled)
        col_names = [self._state_labels.get(i, f"state_{i}") for i in range(self.n_states)]
        return pd.DataFrame(probs, index=df.index, columns=col_names)

    # ------------------------------------------------------------------
    # State labelling
    # ------------------------------------------------------------------

    def _label_states(self, df: pd.DataFrame, X_scaled: np.ndarray) -> None:
        """
        Auto-label states by inspecting per-state statistics.

        Heuristic:
            1. Highest mean realised_vol   → high_vol_breakout
            2. Lowest  mean rel_volume     → low_liquidity
            3. Highest abs(mean log_return) among remaining → trending
            4. Remaining state             → ranging
        """
        raw_states = self.model.predict(X_scaled)
        tmp = df[["log_return", "realised_vol_20", "rel_volume"]].copy()
        tmp["state"] = raw_states

        stats = tmp.groupby("state").agg(
            mean_return=("log_return", "mean"),
            abs_mean_return=("log_return", lambda x: x.abs().mean()),
            mean_vol=("realised_vol_20", "mean"),
            mean_rel_volume=("rel_volume", "mean"),
            count=("log_return", "count"),
        )
        self._state_stats = stats

        labels: dict[int, str] = {}
        used: set[int] = set()

        # 1. Highest volatility → high_vol_breakout
        hv_state = int(stats["mean_vol"].idxmax())
        labels[hv_state] = HIGH_VOL_BREAKOUT
        used.add(hv_state)

        # 2. Lowest relative volume (among remaining) → low_liquidity
        remaining_vol = stats.loc[~stats.index.isin(used), "mean_rel_volume"]
        if len(remaining_vol) > 0:
            ll_state = int(remaining_vol.idxmin())
            labels[ll_state] = LOW_LIQUIDITY
            used.add(ll_state)

        # 3. Highest abs mean return (among remaining) → trending
        remaining_ret = stats.loc[~stats.index.isin(used), "abs_mean_return"]
        if len(remaining_ret) > 0:
            tr_state = int(remaining_ret.idxmax())
            labels[tr_state] = TRENDING
            used.add(tr_state)

        # 4. Everything else → ranging
        for s in range(self.n_states):
            if s not in used:
                labels[s] = RANGING

        self._state_labels = labels

    # ------------------------------------------------------------------
    # Regime distribution (for registry)
    # ------------------------------------------------------------------

    def regime_distribution(self, df: pd.DataFrame) -> dict[str, float]:
        """Return fraction of bars in each regime."""
        regimes = self.predict(df)
        counts = regimes.value_counts(normalize=True)
        return {label: round(float(counts.get(label, 0.0)), 4) for label in ALL_REGIMES}

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        """Pickle the entire model (HMM + scaler + labels) to disk."""
        if path is None:
            path = MODEL_DIR / f"hmm_{self.symbol}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model": self.model,
            "scaler": self._scaler,
            "state_labels": self._state_labels,
            "state_stats": self._state_stats,
            "feature_cols": self.feature_cols,
            "feature_hash": self._feature_hash,
            "n_states": self.n_states,
            "symbol": self.symbol,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh)
        log.info("  HMM saved → %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "RegimeHMM":
        """Load a previously saved model."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)

        obj = cls.__new__(cls)
        obj.model = state["model"]
        obj._scaler = state["scaler"]
        obj._state_labels = state["state_labels"]
        obj._state_stats = state["state_stats"]
        obj.feature_cols = state["feature_cols"]
        obj._feature_hash = state["feature_hash"]
        obj.n_states = state["n_states"]
        obj.symbol = state["symbol"]
        obj._fitted = True
        log.info("  HMM loaded ← %s", path)
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_features(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame missing HMM features: {missing}.  "
                f"Run DataLoader.compute_features() first."
            )

    @property
    def feature_hash(self) -> Optional[str]:
        return self._feature_hash

    @property
    def state_labels(self) -> dict[int, str]:
        return dict(self._state_labels)

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"RegimeHMM(symbol={self.symbol}, states={self.n_states}, {status})"
