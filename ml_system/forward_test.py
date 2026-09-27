"""
Forward Test Runner — ML Trading System Phase 0

End-to-end loop for one or more symbols:
1. Fetch/load data (Alpaca / yfinance -> CSV -> DataLoader)
2. Resample to target timeframe (e.g. 15min)
3. Compute features
4. Fit HMM (or load existing) and label regimes
5. Generate baseline signals
6. (If --live) Submit bracket orders to Alpaca paper account
7. Log training run and trade summary to registry

Usage:
  python -m ml_system.forward_test --symbol SPY --tf 15min --dry-run
  python -m ml_system.forward_test --symbol all --tf 15min --live
"""

import argparse
import logging
from datetime import datetime, timezone
import sys
from concurrent.futures import ThreadPoolExecutor

from ml_system.config import ALL_SYMBOLS, DEFAULT_TIMEFRAME, DATA_DIR, MODEL_DIR
from ml_system.data.alpaca_fetcher import AlpacaFetcher
from ml_system.data.loader import DataLoader
from ml_system.models.hmm_regime import RegimeHMM
from ml_system.signals.baseline_rules import BaselineSignalGenerator
from ml_system.execution.alpaca_paper import AlpacaPaperTrader
from ml_system.registry.retrain_log import RetrainLog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
log = logging.getLogger("forward_test")


def process_symbol(
    symbol: str, 
    tf: str, 
    is_live: bool, 
    is_dry_run: bool,
    use_lstm: bool,
    start_date: datetime | None,
    end_date: datetime | None
):
    """Run the pipeline for a single symbol."""
    log.info("=" * 60)
    log.info("Starting pipeline for %s", symbol)
    log.info("=" * 60)

    try:
        # 1. Fetch data
        fetcher = AlpacaFetcher()
        fetcher.fetch_if_missing(symbol, start=start_date, end=end_date, timeframe_minutes=1)

        # 2. Load & resample
        loader = DataLoader(symbol)
        df_1min = loader.load()
        
        # apply date filtering if provided
        if start_date:
            df_1min = df_1min[df_1min.index >= start_date]
        if end_date:
            df_1min = df_1min[df_1min.index <= end_date]
            
        loader.set_dataframe(df_1min)
        df_resampled = loader.resample(tf)
        
        # 3. Compute features
        df_features = loader.compute_features(df_resampled)
        if df_features.empty:
            log.warning("No data left after computing features for %s. Skipping.", symbol)
            return

        # 4. HMM Fit & Predict
        hmm = RegimeHMM(symbol=symbol)
        hmm.fit(df_features)
        regimes = hmm.predict(df_features)
        
        # Save model
        hmm.save()

        # 4.5. LSTM Fit & Predict (Phase 1)
        lstm_probs = None
        if use_lstm:
            from ml_system.models.lstm_filter import SignalFilterLSTM
            lstm = SignalFilterLSTM(symbol)
            lstm_features = loader.compute_lstm_features(df_features, regimes)
            try:
                lstm = SignalFilterLSTM.load(symbol)
                log.info("Loaded pre-trained LSTM for %s", symbol)
            except Exception:
                log.info("No pre-trained LSTM found for %s, training now...", symbol)
                lstm.fit(lstm_features)
                lstm.save()
            lstm_probs = lstm.predict(lstm_features)

        # 5. Generate Signals
        generator = BaselineSignalGenerator(symbol=symbol)
        signals = generator.generate(df_features, regimes, lstm_probs=lstm_probs)

        # 6. Execution
        trader = AlpacaPaperTrader()
        
        if is_live or is_dry_run:
            log.info("Processing %d signals for %s...", len(signals), symbol)
            for sig in signals:
                trader.submit_signal(sig, dry_run=is_dry_run)
            
            # Save trade log
            trader.save_trade_log()
        else:
            log.info("Generated %d signals for %s. Pass --live to submit or --dry-run to simulate.", len(signals), symbol)

        # 7. Registry
        registry = RetrainLog()
        
        regime_dist = hmm.regime_distribution(df_features)
        log_ll = hmm.model.score(hmm._scaler.transform(df_features[hmm.feature_cols].values))
        
        retrain_id = registry.log_training_run(
            symbol=symbol,
            train_start=str(df_features.index[0]),
            train_end=str(df_features.index[-1]),
            train_bars=len(df_features),
            regime_distribution=regime_dist,
            feature_set_hash=hmm.feature_hash,
            log_likelihood=log_ll,
            n_states=hmm.n_states,
            model_path=str(MODEL_DIR / f"hmm_{symbol}.pkl"),
        )
        
        long_sigs = sum(1 for s in signals if s.direction == "long")
        short_sigs = sum(1 for s in signals if s.direction == "short")
        
        regime_counts = regimes.value_counts().to_dict()
        
        registry.log_trade_summary(
            symbol=symbol,
            total_signals=len(signals),
            long_signals=long_sigs,
            short_signals=short_sigs,
            regime_counts=regime_counts,
            timeframe=tf,
            data_start=str(df_features.index[0]),
            data_end=str(df_features.index[-1]),
            total_bars=len(df_features),
        )

        log.info("Pipeline complete for %s (retrain_id: %s)", symbol, retrain_id[:8])

    except Exception as e:
        log.exception("Error processing %s: %s", symbol, e)


def main():
    parser = argparse.ArgumentParser(description="ML Trading System Forward Test Runner")
    parser.add_argument("--symbol", type=str, default="SPY", help="Symbol to run (AAPL, MSFT, SPY, XAUUSD) or 'all'")
    parser.add_argument("--tf", type=str, default=DEFAULT_TIMEFRAME, help="Timeframe (e.g., 15min, 1H)")
    parser.add_argument("--live", action="store_true", help="Submit bracket orders to Alpaca paper account")
    parser.add_argument("--dry-run", action="store_true", help="Simulate signal submission (no orders sent)")
    parser.add_argument("--use-lstm", action="store_true", help="Apply Phase 1 LSTM confirmation filter")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()

    symbols = ALL_SYMBOLS if args.symbol.lower() == "all" else [args.symbol.upper()]
    
    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.end else None

    # Run concurrently for multiple symbols
    with ThreadPoolExecutor(max_workers=len(symbols)) as executor:
        for symbol in symbols:
            executor.submit(
                process_symbol, 
                symbol, 
                args.tf, 
                args.live, 
                args.dry_run, 
                args.use_lstm,
                start_date, 
                end_date
            )

if __name__ == "__main__":
    main()
