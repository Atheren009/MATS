"""
profile_latency.py — Phase 5: Microsecond latency profiler for the routing pipeline.

Two timing modes are measured and reported:

  HOT PATH  — pre-seeded venues, only strategy.route() + Exchange.submit() timed.
              Represents steady-state production latency (venues are kept live;
              seeding happens on every market-data tick, not per order).
              This is the mode that targets 150-200 µs P99.

  COLD START — full pipeline including venue creation + seeding per iteration.
              Shows the cost of a complete "cold" order lifecycle; ~600-700 µs P99
              is expected and acceptable (this path runs once per bar, not per order).

Component breakdown is also measured in isolation:
    Seeding   — building 3 Exchange books from a price bar
    Routing   — strategy.route() call (decision only, no I/O)
    Matching  — Exchange.submit() per child order (book crossing)
    Overhead  — object creation, UUID4, Python dispatch

Deliverables:
    results/latency_stats.csv       — full percentile table (both modes)
    results/latency_histogram.png   — 3-panel dark chart

Usage:
    $env:PYTHONIOENCODING="utf-8"
    uv run python scripts/profile_latency.py
    uv run python scripts/profile_latency.py --iters 20000
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Silence ALL logging before importing — log calls inside tight loops
# (even filtered ones) add measurable Python dispatch overhead.
logging.disable(logging.CRITICAL)

from order_router.child_order import ChildOrder
from order_router.exchange import Exchange, VENUE_CONFIGS
from order_router.models import Order, OrderSide, OrderType
from order_router.price_feed import Bar
from order_router.router import OrderRouter
from order_router.routing import (
    BestPriceStrategy,
    TWAPStrategy,
    VWAPStrategy,
    SmartStrategy,
)

logging.disable(logging.NOTSET)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
RESULTS_DIR      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
STRATEGY_ORDER   = ["BestPrice", "TWAP(5)", "VWAP(5)", "Smart"]
TARGET_P99_US    = 200.0       # µs — Phase 5 target

# ── Visual identity (same as Phase 4) ─────────────────────────────────────
PALETTE = {
    "BestPrice": "#e74c3c",
    "TWAP(5)":   "#3498db",
    "VWAP(5)":   "#f39c12",
    "Smart":     "#27ae60",
}
BG, PANEL, GRID, TEXT, SUBTEXT = "#0d1117", "#161b22", "#21262d", "#e6edf3", "#8b949e"


# =========================================================================
# Reference bar
# =========================================================================

def _ref_bar() -> Bar:
    """Synthetic AAPL 1-min bar used throughout all profiling iterations."""
    return Bar(
        symbol      = "AAPL",
        timestamp   = datetime(2024, 6, 3, 14, 30, tzinfo=timezone.utc),
        open=189.50, high=190.20, low=189.30, close=189.95,
        volume=8_500.0, trade_count=120.0, vwap=189.82,
    )


# =========================================================================
# Core timing helpers
# =========================================================================

def _new_venues(bar: Bar) -> dict:
    return {name: Exchange(name, cfg) for name, cfg in VENUE_CONFIGS.items()}


def _seeded_venues(bar: Bar) -> dict:
    venues = _new_venues(bar)
    for exch in venues.values():
        exch.seed_from_bar(bar)
    return venues


def profile_hot_path(
    strategy,
    bar: Bar,
    n: int,
    n_warmup: int = 500,
) -> np.ndarray:
    """
    HOT-PATH timing: call strategy.route() then exchange.submit() directly,
    bypassing OrderRouter.submit() so no re-seeding occurs inside the timer.

    This is the true steady-state routing latency — venues are pre-seeded
    once per bar (a market-data event), then N orders are routed against
    that snapshot. Only the routing decision + book crossing are timed.

    Returns latencies in **microseconds**.
    """
    bars_arg = [bar] * 5   # needed by TWAP/VWAP for slice-weight computation

    logging.disable(logging.CRITICAL)
    gc.disable()

    # Warmup
    for _ in range(n_warmup):
        venues = _seeded_venues(bar)
        order  = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=5_000.0)
        tranches = strategy.route(order, venues, bars_arg)
        for tranche in tranches:
            for child in tranche:
                if child.quantity > 0 and child.venue in venues:
                    venues[child.venue].submit(child)

    lats = np.empty(n, dtype=np.float64)
    for i in range(n):
        venues = _seeded_venues(bar)
        order  = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=5_000.0)

        t0 = time.perf_counter_ns()
        tranches = strategy.route(order, venues, bars_arg)
        for tranche in tranches:
            for child in tranche:
                if child.quantity > 0 and child.venue in venues:
                    venues[child.venue].submit(child)
        t1 = time.perf_counter_ns()

        lats[i] = (t1 - t0) * 1e-3

    gc.enable()
    logging.disable(logging.NOTSET)
    return lats



def profile_cold_start(
    strategy,
    bar: Bar,
    n: int,
    n_warmup: int = 500,
) -> np.ndarray:
    """
    COLD-START timing: full pipeline including venue creation and seeding.
    Expected P99: 600-800 µs (dominated by UUID4 + SortedDict initialisation).
    Runs once per bar in production, not per order.
    """
    logging.disable(logging.CRITICAL)
    gc.disable()

    for _ in range(n_warmup):
        venues = _new_venues(bar)
        order  = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=5_000.0)
        router = OrderRouter(venues=venues, strategy=strategy)
        router.submit(order, bars=[bar])

    lats = np.empty(n, dtype=np.float64)
    for i in range(n):
        venues = _new_venues(bar)
        order  = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=5_000.0)
        router = OrderRouter(venues=venues, strategy=strategy)

        t0 = time.perf_counter_ns()
        router.submit(order, bars=[bar])
        t1 = time.perf_counter_ns()

        lats[i] = (t1 - t0) * 1e-3

    gc.enable()
    logging.disable(logging.NOTSET)
    return lats


def profile_components(bar: Bar, n: int = 5_000) -> dict[str, np.ndarray]:
    """
    Measure the four cost centres in isolation.

    Returns dict:  component_label → float64 array of µs latencies.
    """
    logging.disable(logging.CRITICAL)
    gc.disable()

    # ── 1. Seeding  (3 venues × 6 orders each) ───────────────────────────
    seed_ns = np.empty(n, dtype=np.float64)
    for i in range(n):
        venues = _new_venues(bar)
        t0 = time.perf_counter_ns()
        for exch in venues.values():
            exch.seed_from_bar(bar)
        t1 = time.perf_counter_ns()
        seed_ns[i] = (t1 - t0) * 1e-3

    # ── 2. Routing decision  (BestPrice, pre-seeded) ─────────────────────
    strat = BestPriceStrategy()
    route_ns = np.empty(n, dtype=np.float64)
    for i in range(n):
        venues = _seeded_venues(bar)
        order  = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=5_000.0)
        t0 = time.perf_counter_ns()
        strat.route(order, venues)
        t1 = time.perf_counter_ns()
        route_ns[i] = (t1 - t0) * 1e-3

    # ── 3. Order matching  (single ALPHA submit, pre-seeded) ─────────────
    match_ns = np.empty(n, dtype=np.float64)
    for i in range(n):
        venues = _seeded_venues(bar)
        child  = ChildOrder("parent", "ALPHA", OrderSide.BUY, quantity=1_500.0)
        t0 = time.perf_counter_ns()
        venues["ALPHA"].submit(child)
        t1 = time.perf_counter_ns()
        match_ns[i] = (t1 - t0) * 1e-3

    # ── 4. Object overhead  (Order + Router creation, no actual work) ─────
    obj_ns = np.empty(n, dtype=np.float64)
    for i in range(n):
        venues = _seeded_venues(bar)
        t0 = time.perf_counter_ns()
        _ = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=5_000.0)
        _ = OrderRouter(venues=venues, strategy=strat)
        t1 = time.perf_counter_ns()
        obj_ns[i] = (t1 - t0) * 1e-3

    gc.enable()
    logging.disable(logging.NOTSET)
    return {
        "Seeding\n(3 venues)":   seed_ns,
        "Routing\ndecision":     route_ns,
        "Order\nmatching":       match_ns,
        "Object\noverhead":      obj_ns,
    }


# =========================================================================
# Statistics
# =========================================================================

PERCENTILES = [50, 75, 90, 95, 99, 99.9]

def compute_stats(
    results: dict[str, np.ndarray],
    mode: str = "hot",
) -> pd.DataFrame:
    rows = []
    for name in STRATEGY_ORDER:
        lats = results[name]
        row  = {
            "mode":     mode,
            "strategy": name,
            "n_iters":  len(lats),
            "mean_us":  round(np.mean(lats), 2),
            "std_us":   round(np.std(lats), 2),
            "min_us":   round(np.min(lats), 2),
        }
        for p in PERCENTILES:
            key = f"p{str(p).replace('.', '_')}_us"
            row[key] = round(float(np.percentile(lats, p)), 2)
        row["max_us"] = round(np.max(lats), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def print_stats(df: pd.DataFrame, mode: str = "hot") -> None:
    SEP  = "=" * 82
    LINE = "-" * 80
    target = TARGET_P99_US

    if mode == "hot":
        title = "PHASE 5 -- HOT-PATH LATENCY  (pre-seeded venues: route + match only)"
    else:
        title = "PHASE 5 -- COLD-START LATENCY  (venue creation + seeding included)"

    print()
    print(SEP)
    print(f"  {title}")
    print(SEP)
    print(
        f"  {'Strategy':<12}"
        f"{'Mean':>9}"
        f"{'Std':>8}"
        f"{'P50':>8}"
        f"{'P90':>8}"
        f"{'P95':>8}"
        f"{'P99':>9}"
        f"{'P99.9':>9}"
        f"{'Max':>9}"
        f"  vs {target:.0f}us"
    )
    print("  " + LINE)
    for _, row in df.iterrows():
        p99 = row["p99_us"]
        flag = "OK " if (mode == "hot" and p99 <= target) else ("OK " if mode == "cold" else "OVER")
        if mode == "cold":
            flag = "(cold)"  # cold-start is not compared to the 200us target
        elif p99 > target:
            flag = "OVER"
        print(
            f"  {row['strategy']:<12}"
            f"{row['mean_us']:>8.1f}us"
            f"{row['std_us']:>7.1f}"
            f"{row['p50_us']:>8.1f}"
            f"{row['p90_us']:>8.1f}"
            f"{row['p95_us']:>8.1f}"
            f"{p99:>8.1f}"
            f"{row['p99_9_us']:>9.1f}"
            f"{row['max_us']:>9.1f}"
            f"  [{flag}]"
        )
    print("  " + LINE)
    if mode == "hot":
        print(f"  P99 target (hot path): <= {target:.0f} us")
    else:
        print("  Note: cold-start includes venue creation + seeding (not the P99 target path)")
    print(SEP)
    print()


# =========================================================================
# Chart
# =========================================================================

def _style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, linewidth=0.6, which="both")


def plot_latency(
    results: dict[str, np.ndarray],
    components: dict[str, np.ndarray],
    stats_df: pd.DataFrame,
    n_iters: int,
) -> str:
    """
    Generate and save the 3-panel latency chart.

    Panel layout
    ────────────
    [  TOP — full width  ]  Log-scale histogram PDF (all strategies)
    [ BOTTOM-LEFT        ]  CDF with P50 / P99 markers
    [ BOTTOM-RIGHT       ]  Component median breakdown (stacked bar)
    """

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
    })

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.44, wspace=0.30,
        left=0.07, right=0.97, top=0.90, bottom=0.08,
    )
    ax_hist = fig.add_subplot(gs[0, :])   # top — full width
    ax_cdf  = fig.add_subplot(gs[1, 0])   # bottom-left
    ax_comp = fig.add_subplot(gs[1, 1])   # bottom-right

    for ax in (ax_hist, ax_cdf, ax_comp):
        _style_ax(ax)

    # ── Panel 1: Log-scale histogram PDF ─────────────────────────────────
    # Bin edges spanning 10 µs … 50 000 µs on a log scale
    all_lats = np.concatenate(list(results.values()))
    lo = max(10, np.percentile(all_lats, 0.1))
    hi = min(50_000, np.percentile(all_lats, 99.99))
    bins = np.logspace(np.log10(lo), np.log10(hi), 80)

    for name in STRATEGY_ORDER:
        lats  = results[name]
        color = PALETTE[name]
        p50   = np.percentile(lats, 50)
        p99   = np.percentile(lats, 99)
        label = (
            f"{name}  "
            f"P50={p50:.0f}µs  "
            f"P99={p99:.0f}µs"
        )
        ax_hist.hist(
            lats, bins=bins,
            density   = True,
            histtype  = "stepfilled",
            alpha     = 0.25,
            color     = color,
        )
        ax_hist.hist(
            lats, bins=bins,
            density   = True,
            histtype  = "step",
            linewidth = 1.8,
            color     = color,
            label     = label,
        )
        # P99 marker
        ax_hist.axvline(
            p99,
            color     = color,
            linestyle = ":",
            linewidth = 1.4,
            alpha     = 0.8,
        )

    # Target P99 line
    ax_hist.axvline(
        TARGET_P99_US,
        color     = "white",
        linestyle = "--",
        linewidth = 1.6,
        alpha     = 0.55,
        label     = f"Target P99 = {TARGET_P99_US:.0f} µs",
    )

    ax_hist.set_xscale("log")
    ax_hist.set_title(
        f"End-to-End Latency Distribution  (N = {n_iters:,} iterations per strategy)  "
        f"—  log scale, dotted lines = P99",
        color=TEXT, pad=8,
    )
    ax_hist.set_xlabel("Latency (µs, log scale)", color=TEXT)
    ax_hist.set_ylabel("Probability Density", color=TEXT)
    ax_hist.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0f}µs" if x >= 1 else f"{x:.1f}µs")
    )
    leg = ax_hist.legend(
        loc="upper right", framealpha=0.25,
        facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
    )

    # ── Panel 2: CDF ─────────────────────────────────────────────────────
    for name in STRATEGY_ORDER:
        lats  = np.sort(results[name])
        cdf   = np.arange(1, len(lats) + 1) / len(lats) * 100
        ax_cdf.plot(lats, cdf, color=PALETTE[name], linewidth=2.0, label=name)

    # Percentile reference lines
    for pct, ls in [(50, ":"), (95, "--"), (99, "-.")]:
        ax_cdf.axhline(pct, color=SUBTEXT, linestyle=ls, linewidth=0.9, alpha=0.6)
        ax_cdf.text(
            ax_cdf.get_xlim()[0] if ax_cdf.get_xlim()[0] > 0 else 5,
            pct + 0.5, f"P{pct}", color=SUBTEXT, fontsize=7,
        )

    ax_cdf.axvline(
        TARGET_P99_US,
        color="white", linestyle="--", linewidth=1.4, alpha=0.5,
        label=f"Target = {TARGET_P99_US:.0f}µs",
    )
    ax_cdf.set_xscale("log")
    ax_cdf.set_title("Cumulative Distribution (CDF)\n— P99 target line at 200 µs", color=TEXT, pad=8)
    ax_cdf.set_xlabel("Latency (µs, log scale)", color=TEXT)
    ax_cdf.set_ylabel("Percentile (%)", color=TEXT)
    ax_cdf.set_ylim(0, 101)
    ax_cdf.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0f}" if x >= 1 else f"{x:.1f}")
    )
    ax_cdf.legend(
        loc="upper left", framealpha=0.25,
        facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8,
    )

    # ── Panel 3: Component median breakdown ───────────────────────────────
    comp_labels = list(components.keys())
    comp_medians = [float(np.median(v)) for v in components.values()]
    comp_p99s    = [float(np.percentile(v, 99)) for v in components.values()]

    colors_comp = ["#5dade2", "#a569bd", "#45b39d", "#f0b27a"]
    x = np.arange(len(comp_labels))
    bar_w = 0.35

    bars_med = ax_comp.bar(
        x - bar_w / 2, comp_medians, bar_w,
        color=[c + "bb" for c in colors_comp],
        label="Median",
        edgecolor="white", linewidth=0.5,
    )
    bars_p99 = ax_comp.bar(
        x + bar_w / 2, comp_p99s, bar_w,
        color=colors_comp,
        label="P99",
        edgecolor="white", linewidth=0.5,
    )

    # Value labels
    for bar_ in list(bars_med) + list(bars_p99):
        h = bar_.get_height()
        ax_comp.text(
            bar_.get_x() + bar_.get_width() / 2,
            h + 0.5,
            f"{h:.1f}",
            ha="center", va="bottom", fontsize=7.5, color=TEXT,
        )

    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(comp_labels, fontsize=8.5, ha="center")
    ax_comp.set_title(
        "Cost-Centre Breakdown\n(isolated median vs P99 per component)",
        color=TEXT, pad=8,
    )
    ax_comp.set_ylabel("Latency (µs)", color=TEXT)
    ax_comp.legend(
        framealpha=0.25, facecolor=PANEL, edgecolor=GRID,
        labelcolor=TEXT, fontsize=8,
    )

    # ── Master title ─────────────────────────────────────────────────────
    fig.suptitle(
        "AAPL Order Router  ·  Phase 5 — HOT-PATH Latency  ·  "
        "Pre-seeded venues (route + match only)  ·  "
        f"time.perf_counter_ns()  ·  GC off  ·  N={n_iters:,}/strategy",
        fontsize=12, fontweight="bold", color=TEXT, y=0.95,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "latency_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.warning("Chart saved → %s", path)
    return path


# =========================================================================
# Entry point
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 — Latency profiler")
    parser.add_argument("--iters",  type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=500)
    args = parser.parse_args()

    bar = _ref_bar()
    strategies = [
        BestPriceStrategy(),
        TWAPStrategy(num_slices=5),
        VWAPStrategy(num_slices=5),
        SmartStrategy(),
    ]

    # ── HOT-PATH timing  (pre-seeded, only route+match timed) ─────────────
    print(f"\n[HOT PATH] {len(strategies)} strategies x {args.iters:,} iters (warmup={args.warmup}) ...")
    hot: dict[str, np.ndarray] = {}
    for strategy in strategies:
        print(f"  [{strategy.name:<12}] ...", end="", flush=True)
        lats = profile_hot_path(strategy, bar, n=args.iters, n_warmup=args.warmup)
        hot[strategy.name] = lats
        p99 = np.percentile(lats, 99)
        flag = "OK" if p99 <= TARGET_P99_US else f"OVER ({p99:.0f}us)"
        print(f"  P50={np.percentile(lats,50):.0f}us  P99={p99:.1f}us  [{flag}]")

    # ── COLD-START timing  (venue creation + seeding included) ────────────
    print(f"\n[COLD START] {len(strategies)} strategies x {args.iters:,} iters ...")
    cold: dict[str, np.ndarray] = {}
    for strategy in strategies:
        print(f"  [{strategy.name:<12}] ...", end="", flush=True)
        lats = profile_cold_start(strategy, bar, n=args.iters, n_warmup=args.warmup)
        cold[strategy.name] = lats
        p99 = np.percentile(lats, 99)
        print(f"  P50={np.percentile(lats,50):.0f}us  P99={p99:.1f}us  (cold start)")

    # ── Component breakdown ───────────────────────────────────────────────
    print(f"\n[COMPONENTS] {args.iters // 2:,} iters each ...")
    components = profile_components(bar, n=args.iters // 2)
    for name, lats in components.items():
        label = name.replace("\n", " ")
        print(f"  [{label:<22}]  median={np.median(lats):.1f}us  P99={np.percentile(lats, 99):.1f}us")

    # ── Print stats tables ────────────────────────────────────────────────
    hot_df  = compute_stats(hot,  mode="hot")
    cold_df = compute_stats(cold, mode="cold")
    print_stats(hot_df,  mode="hot")
    print_stats(cold_df, mode="cold")

    # ── Save combined CSV ─────────────────────────────────────────────────
    combined = pd.concat([hot_df, cold_df], ignore_index=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "latency_stats.csv")
    combined.to_csv(csv_path, index=False)
    print(f"  Stats CSV  -> {csv_path}")

    # ── Chart (hot-path results — this is the P99 deliverable) ───────────
    chart_path = plot_latency(hot, components, hot_df, args.iters)
    print(f"  Chart PNG  -> {chart_path}")
    print()


if __name__ == "__main__":
    main()
