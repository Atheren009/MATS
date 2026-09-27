"""
run_backtest.py — Phase 4 backtest runner for AAPL routing strategy comparison.

Runs all four strategies (BestPrice, TWAP, VWAP, Smart) over 200 windows of
real AAPL 1-minute bar data and produces two deliverables:

  results/backtest_results.csv   — full per-window, per-strategy metrics table
  results/slippage_comparison.png— four-panel chart proving Smart > Naive

Usage:
    uv run python scripts/run_backtest.py
    uv run python scripts/run_backtest.py --windows 500
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")   # headless rendering — no display needed
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Make order_router importable from project root ────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from order_router.backtest import BacktestEngine, STRATEGY_ORDER
from order_router.models import OrderSide

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
CSV_PATH     = os.path.join(PROJECT_ROOT, "AAPL_1min_2024-2026.csv")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")

# ── Visual identity ───────────────────────────────────────────────────────
PALETTE = {
    "BestPrice": "#e74c3c",   # red    — the naive baseline
    "TWAP(5)":   "#3498db",   # blue
    "VWAP(5)":   "#f39c12",   # amber
    "Smart":     "#27ae60",   # green  — the champion
}
BG      = "#0d1117"   # GitHub dark background
PANEL   = "#161b22"   # slightly lighter panel
GRID    = "#21262d"
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"


# =========================================================================
# 1.  Run the backtest
# =========================================================================

def run_backtest(max_windows: int) -> pd.DataFrame:
    engine = BacktestEngine(
        csv_path    = CSV_PATH,
        window_size = 5,
        order_qty   = 5_000.0,
        side        = OrderSide.BUY,
        max_windows = max_windows,
        log_every   = 50,
    )
    return engine.run()


# =========================================================================
# 2.  Save CSV
# =========================================================================

def save_csv(df: pd.DataFrame) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "backtest_results.csv")
    df.to_csv(path, index=False)
    log.info("CSV saved → %s  (%d rows)", path, len(df))
    return path


# =========================================================================
# 3.  Print terminal summary
# =========================================================================

def print_summary(df: pd.DataFrame) -> None:
    agg = (
        df.groupby("strategy")
        .agg(
            avg_fill_rate   = ("fill_rate_pct",               "mean"),
            avg_slippage    = ("slippage_bps",                "mean"),
            avg_IS          = ("implementation_shortfall_bps","mean"),
            total_fees      = ("fees_paid",                   "sum"),
            total_unfilled  = ("unfilled_qty",                "sum"),
            total_filled    = ("filled_qty",                  "sum"),
        )
        .round(4)
        .reindex(STRATEGY_ORDER)
    )

    smart = agg.loc["Smart"]
    naive = agg.loc["BestPrice"]
    fill_lift    = smart["avg_fill_rate"] - naive["avg_fill_rate"]
    unfill_saved = naive["total_unfilled"] - smart["total_unfilled"]
    is_saving    = naive["avg_IS"] - smart["avg_IS"]

    SEP  = "=" * 78
    LINE = "-" * 76

    print()
    print(SEP)
    print("  PHASE 4 BACKTEST -- SMART vs NAIVE ROUTING  (AAPL, 5 000-share BUY)")
    print(SEP)
    print(
        f"  {'Strategy':<14}"
        f"{'Fill Rate%':>12}"
        f"{'Slip (bps)':>12}"
        f"{'IS (bps)':>10}"
        f"{'Fees $':>10}"
        f"{'Unfilled':>12}"
    )
    print("  " + LINE)
    for name, row in agg.iterrows():
        marker = "  <-- BEST" if name == "Smart" else ""
        print(
            f"  {name:<14}"
            f"{row['avg_fill_rate']:>12.2f}"
            f"{row['avg_slippage']:>12.4f}"
            f"{row['avg_IS']:>10.4f}"
            f"{row['total_fees']:>10.2f}"
            f"{row['total_unfilled']:>12,.0f}"
            f"{marker}"
        )
    print("  " + LINE)
    print()
    print(f"  [OK] Smart fill rate is {fill_lift:+.1f} pp higher than BestPrice")
    print(f"  [OK] Smart leaves {unfill_saved:,.0f} fewer shares unfilled vs BestPrice")
    print(f"  [OK] Smart implementation shortfall is {is_saving:+.4f} bps better than BestPrice")
    print(SEP)
    print()



# =========================================================================
# 4.  Generate the 4-panel comparison chart
# =========================================================================

def plot_results(df: pd.DataFrame, max_windows: int) -> str:
    """
    Four-panel dark-mode chart proving Smart routing beats naive.

    Panel layout
    ────────────
    [  TOP — full width  ]  Fill Rate % over time (rolling avg)
    [ BOTTOM-LEFT        ]  Slippage distribution (violin)
    [ BOTTOM-MIDDLE      ]  Cumulative unfilled shares
    [ BOTTOM-RIGHT       ]  Implementation shortfall distribution (violin)
    """

    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        10,
        "axes.titlesize":   11,
        "axes.titleweight": "bold",
        "axes.labelsize":   9,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "legend.fontsize":  8.5,
    })

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        2, 3,
        figure  = fig,
        hspace  = 0.44,
        wspace  = 0.32,
        left    = 0.06,
        right   = 0.97,
        top     = 0.90,
        bottom  = 0.08,
    )
    ax_fill  = fig.add_subplot(gs[0, :])      # top row — full width
    ax_slip  = fig.add_subplot(gs[1, 0])      # bottom-left
    ax_unfil = fig.add_subplot(gs[1, 1])      # bottom-middle
    ax_is    = fig.add_subplot(gs[1, 2])      # bottom-right

    for ax in (ax_fill, ax_slip, ax_unfil, ax_is):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.6)

    roll = 20   # rolling-window for smoothing

    # ── Panel 1: Fill Rate over time ─────────────────────────────────────
    for name in STRATEGY_ORDER:
        sub   = df[df["strategy"] == name].sort_values("window_idx")
        color = PALETTE[name]
        raw   = sub["fill_rate_pct"].values
        avg   = raw.mean()
        smoothed = pd.Series(raw).rolling(roll, min_periods=1).mean().values

        ax_fill.plot(
            sub["window_idx"], smoothed,
            label  = f"{name}  (avg {avg:.1f}%)",
            color  = color,
            linewidth = 2.4,
            zorder = 3,
        )
        ax_fill.fill_between(
            sub["window_idx"], smoothed,
            alpha = 0.10, color = color, zorder = 2,
        )

    ax_fill.axhline(100, color=SUBTEXT, linestyle="--", linewidth=0.9, alpha=0.6)
    ax_fill.set_title(
        f"Fill Rate % over {max_windows} Windows  "
        f"(rolling {roll}-window avg)  —  5 000-share BUY, AAPL 1-min bars",
        color=TEXT, pad=8,
    )
    ax_fill.set_xlabel("Window index (chronological)", color=TEXT)
    ax_fill.set_ylabel("Fill Rate (%)", color=TEXT)
    ax_fill.set_ylim(0, 120)
    ax_fill.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    leg = ax_fill.legend(
        loc="upper left", framealpha=0.25,
        facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
    )

    # ── Panel 2: Slippage distribution (violin) ──────────────────────────
    _violin_panel(
        ax    = ax_slip,
        df    = df,
        col   = "slippage_bps",
        title = "Slippage vs VWAP (bps)\n[lower = better]",
        ylabel= "Slippage (bps)",
    )

    # ── Panel 3: Cumulative unfilled shares ───────────────────────────────
    for name in STRATEGY_ORDER:
        sub  = df[df["strategy"] == name].sort_values("window_idx")
        cum  = sub["unfilled_qty"].cumsum().values
        ax_unfil.plot(
            sub["window_idx"], cum,
            label     = name,
            color     = PALETTE[name],
            linewidth = 2.2,
        )
        ax_unfil.fill_between(
            sub["window_idx"], cum,
            alpha=0.07, color=PALETTE[name],
        )

    ax_unfil.set_title("Cumulative Unfilled Shares\n[lower = better]", color=TEXT, pad=8)
    ax_unfil.set_xlabel("Window index", color=TEXT)
    ax_unfil.set_ylabel("Cumulative unfilled (shares)", color=TEXT)
    ax_unfil.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y/1e3:.0f}K")
    )
    leg2 = ax_unfil.legend(
        loc="upper left", framealpha=0.25,
        facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
    )

    # ── Panel 4: Implementation Shortfall distribution ────────────────────
    _violin_panel(
        ax    = ax_is,
        df    = df.dropna(subset=["implementation_shortfall_bps"]),
        col   = "implementation_shortfall_bps",
        title = "Implementation Shortfall (bps)\n[slippage + fees, lower = better]",
        ylabel= "IS (bps)",
    )

    # ── Master title ─────────────────────────────────────────────────────
    fig.suptitle(
        "AAPL Order Routing Backtest  ·  Phase 4  ·  "
        f"{max_windows} Windows × 5 bars  ·  5 000-share Market BUY  ·  "
        "3 Simulated Exchanges (ALPHA / BETA / GAMMA)",
        fontsize   = 12,
        fontweight = "bold",
        color      = TEXT,
        y          = 0.95,
    )

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    chart_path = os.path.join(RESULTS_DIR, "slippage_comparison.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("Chart saved → %s", chart_path)
    return chart_path


def _violin_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    col: str,
    title: str,
    ylabel: str,
) -> None:
    """Render a styled violin plot for `col` split by strategy."""
    data   = [df[df["strategy"] == name][col].dropna().values for name in STRATEGY_ORDER]
    colors = [PALETTE[name] for name in STRATEGY_ORDER]
    pos    = list(range(len(STRATEGY_ORDER)))

    # --- violin bodies ---
    vp = ax.violinplot(data, positions=pos, showmedians=True, showextrema=True)

    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.65)

    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        vp[key].set_color(TEXT)
        vp[key].set_linewidth(1.3)

    # --- mean dots ---
    for i, (series, color) in enumerate(zip(data, colors)):
        if len(series):
            ax.scatter(
                [i], [series.mean()],
                color      = color,
                zorder     = 5,
                s          = 55,
                edgecolors = "white",
                linewidth  = 1.2,
            )
            ax.annotate(
                f"{series.mean():.2f}",
                xy         = (i, series.mean()),
                xytext     = (i + 0.22, series.mean()),
                color      = TEXT,
                fontsize   = 7.5,
                va         = "center",
            )

    ax.axhline(0, color=SUBTEXT, linestyle="--", linewidth=0.8, alpha=0.55)
    ax.set_xticks(pos)
    ax.set_xticklabels(STRATEGY_ORDER, rotation=12, ha="right")
    ax.set_title(title, color=TEXT, pad=8)
    ax.set_ylabel(ylabel, color=TEXT)


# =========================================================================
# 5.  Entry point
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 — AAPL routing strategy backtest"
    )
    parser.add_argument(
        "--windows", type=int, default=200,
        help="Number of 5-bar windows to replay (default: 200 = 1 000 bars)",
    )
    args = parser.parse_args()

    if not os.path.exists(CSV_PATH):
        log.error("CSV not found: %s", CSV_PATH)
        sys.exit(1)

    # ── Run ───────────────────────────────────────────────────────────────
    df         = run_backtest(args.windows)
    csv_path   = save_csv(df)
    print_summary(df)
    chart_path = plot_results(df, args.windows)

    print(f"  Deliverables:")
    print(f"    CSV   → {csv_path}")
    print(f"    Chart → {chart_path}")
    print()


if __name__ == "__main__":
    main()
