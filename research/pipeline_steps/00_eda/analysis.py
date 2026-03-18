"""
Pipeline EDA — Full Phase 1 Data Exploration

Runs the full Phase 1 pipeline (universe scan -> OHLCV download -> technical
calculation), then explores the enriched DataFrame BEFORE any filtering.

Outputs:
  - enriched_snapshot.parquet  (reusable by later per-step analyses)
  - metadata_snapshot.parquet
  - eda_report.md              (self-contained markdown report with embedded figures)
"""

from __future__ import annotations

import base64
import sys
import warnings
from dataclasses import asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stock_analyzer.config import DEFAULT_CONFIG, MEGA_CAP_BLACKLIST  # noqa: E402
from stock_analyzer.screener.market_data import MarketDataFetcher  # noqa: E402
from stock_analyzer.screener.technical import TechnicalCalculator  # noqa: E402
from stock_analyzer.screener.universe import UniverseLoader  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

EDA_DIR = Path(__file__).resolve().parent
SNAPSHOT_PATH = EDA_DIR / "enriched_snapshot.parquet"
META_SNAPSHOT_PATH = EDA_DIR / "metadata_snapshot.parquet"
REPORT_PATH = EDA_DIR / "eda_report.md"

cfg = DEFAULT_CONFIG

# ---------------------------------------------------------------------------
# Report infrastructure  (per report_generation_policy.md)
# ---------------------------------------------------------------------------
report_data: dict = {}
report_figures: dict[str, str] = {}
analysis_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def capture_figure(fig: plt.Figure, name: str) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    report_figures[name] = img_b64
    buf.close()
    plt.close(fig)
    return img_b64


def df_to_md_table(df: pd.DataFrame, index: bool = False) -> str:
    lines: list[str] = []
    headers = ([""] if index else []) + [str(c) for c in df.columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for idx, row in df.iterrows():
        vals = ([str(idx)] if index else []) + [
            str(v) if pd.notna(v) else "" for v in row
        ]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


# ============================================================================
# SECTION 1 — Data Loading
# ============================================================================
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if SNAPSHOT_PATH.exists():
        print("[Section 1] Loading cached enriched snapshot...")
        enriched = pd.read_parquet(SNAPSHOT_PATH)
        metadata = (
            pd.read_parquet(META_SNAPSHOT_PATH)
            if META_SNAPSHOT_PATH.exists()
            else pd.DataFrame()
        )
        print(f"  Loaded {len(enriched)} tickers from snapshot")
        return enriched, metadata

    print("[Section 1] No snapshot — running full Phase 1 pipeline...")
    print(f"  Config: market_cap {cfg.market_cap_min/1e6:.0f}M-{cfg.market_cap_max/1e9:.0f}B, "
          f"min_vol_3m={cfg.min_avg_volume_3m:,}")

    tickers, metadata = UniverseLoader(cfg).load()
    print(f"  Universe: {len(tickers)} unique tickers")

    bundle = MarketDataFetcher(cfg).fetch(tickers, metadata)
    print(f"  OHLCV: {len(bundle.ohlcv)} tickers")

    enriched = TechnicalCalculator(cfg).compute(bundle)
    print(f"  Enriched: {len(enriched)} tickers with full technical features")

    enriched.to_parquet(SNAPSHOT_PATH)
    metadata.to_parquet(META_SNAPSHOT_PATH)
    print(f"  Snapshots saved to {EDA_DIR.name}/")
    return enriched, metadata


# ============================================================================
# SECTION 2 — Feature Distributions
# ============================================================================
def analyze_distributions(enriched: pd.DataFrame) -> None:
    print("[Section 2] Computing feature distributions...")

    numeric_cols = enriched.select_dtypes(include=[np.number]).columns.tolist()
    bool_cols = [c for c in enriched.columns if enriched[c].dtype == "bool"
                 or set(enriched[c].dropna().unique()).issubset({True, False, 0, 1})]
    bool_cols = [c for c in bool_cols if c in ["above_50dma", "above_200dma",
                                                 "breakout_detected", "trend_50_above_200"]]

    stats = enriched[numeric_cols].describe().T
    stats["missing_%"] = (enriched[numeric_cols].isna().mean() * 100).round(1)
    report_data["describe"] = stats.round(2).to_dict()

    # --- 2a. Price / MA features -----------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Price / Moving Average Features", fontsize=16, fontweight="bold")

    _hist(axes[0, 0], enriched["last_close"], "Last Close ($)", clip_q=0.99)
    _hist(axes[0, 1], enriched["sma_50"], "SMA 50", clip_q=0.99)
    _hist(axes[0, 2], enriched["sma_200"], "SMA 200", clip_q=0.99)
    _hist_thresh(axes[1, 0], enriched["pct_from_52wk_high"],
                 "% from 52-wk High", thresh=-cfg.max_pct_from_52wk_high,
                 thresh_label=f"Threshold = -{cfg.max_pct_from_52wk_high}%")
    _bool_bar(axes[1, 1], enriched["above_50dma"], "Above 50 DMA")
    _bool_bar(axes[1, 2], enriched["above_200dma"], "Above 200 DMA")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    capture_figure(fig, "price_ma_features")

    pct_above_50 = (enriched["above_50dma"] == True).mean() * 100  # noqa: E712
    pct_above_200 = (enriched["above_200dma"] == True).mean() * 100  # noqa: E712
    pct_golden = (enriched["trend_50_above_200"] == True).mean() * 100 if "trend_50_above_200" in enriched.columns else 0  # noqa: E712
    pct_within_52wk = (enriched["pct_from_52wk_high"] >= -cfg.max_pct_from_52wk_high).mean() * 100

    report_data["price_ma"] = {
        "pct_above_50dma": round(pct_above_50, 1),
        "pct_above_200dma": round(pct_above_200, 1),
        "pct_golden_cross": round(pct_golden, 1),
        "pct_within_52wk_high": round(pct_within_52wk, 1),
        "median_pct_from_52wk": round(float(enriched["pct_from_52wk_high"].median()), 2),
    }

    # --- 2b. Volume features ---------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Volume Features", fontsize=16, fontweight="bold")

    _hist_thresh(axes[0], enriched["rel_volume_latest"].clip(upper=10),
                 "Relative Volume (latest)", thresh=cfg.min_relative_volume,
                 thresh_label=f"Threshold = {cfg.min_relative_volume}x")
    _hist_thresh(axes[1], enriched["rel_volume_3d_max"].clip(upper=10),
                 "Relative Volume (3d max)", thresh=cfg.min_relative_volume,
                 thresh_label=f"Threshold = {cfg.min_relative_volume}x")
    _hist_thresh(axes[2], enriched["avg_daily_dollar_volume"],
                 "Avg Daily $ Volume", thresh=cfg.min_avg_daily_dollar_volume,
                 thresh_label=f"Threshold = ${cfg.min_avg_daily_dollar_volume/1e6:.0f}M",
                 log_scale=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    capture_figure(fig, "volume_features")

    pct_rel_vol = (enriched["rel_volume_3d_max"] >= cfg.min_relative_volume).mean() * 100
    pct_dollar_vol = (enriched["avg_daily_dollar_volume"] >= cfg.min_avg_daily_dollar_volume).mean() * 100
    report_data["volume"] = {
        "pct_pass_rel_volume": round(pct_rel_vol, 1),
        "pct_pass_dollar_volume": round(pct_dollar_vol, 1),
        "median_rel_vol_3d_max": round(float(enriched["rel_volume_3d_max"].median()), 2),
        "median_dollar_vol": round(float(enriched["avg_daily_dollar_volume"].median()), 0),
    }

    # --- 2c. Breakout features -------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Breakout Features", fontsize=16, fontweight="bold")

    _bool_bar(axes[0, 0], enriched["breakout_detected"], "Breakout Detected")

    bo = enriched[enriched["breakout_detected"] == True]  # noqa: E712
    if not bo.empty:
        _hist_thresh(axes[0, 1], bo["pct_above_breakout"].clip(upper=100),
                     "% Above Breakout (detected only)",
                     thresh=cfg.max_pct_above_breakout,
                     thresh_label=f"Threshold = {cfg.max_pct_above_breakout}%")
        tf_counts = bo["breakout_timeframe"].value_counts().sort_index()
        axes[1, 0].bar([str(int(x)) for x in tf_counts.index], tf_counts.values,
                       color=sns.color_palette("muted", len(tf_counts)), edgecolor="white")
        for i, v in enumerate(tf_counts.values):
            axes[1, 0].text(i, v + max(tf_counts.values)*0.02,
                            f"{v:,}", ha="center", fontsize=10)
        axes[1, 0].set_title("Breakout Timeframe Distribution", fontweight="bold")
        axes[1, 0].set_xlabel("Lookback days")
        axes[1, 0].set_ylabel("Count")

        _hist_thresh(axes[1, 1], bo["breakout_volume_ratio"].dropna().clip(upper=10),
                     "Breakout Volume Ratio (detected only)",
                     thresh=cfg.min_breakout_volume_ratio,
                     thresh_label=f"Threshold = {cfg.min_breakout_volume_ratio}x")
    else:
        for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
            ax.text(0.5, 0.5, "No breakouts detected", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    capture_figure(fig, "breakout_features")

    pct_breakout = (enriched["breakout_detected"] == True).mean() * 100  # noqa: E712
    report_data["breakout"] = {
        "pct_breakout_detected": round(pct_breakout, 1),
        "n_breakouts": int(bo.shape[0]) if not bo.empty else 0,
        "timeframe_counts": bo["breakout_timeframe"].value_counts().to_dict() if not bo.empty else {},
    }

    # --- 2d. Quality / Structure features --------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Quality / Structure Features", fontsize=16, fontweight="bold")

    _hist(axes[0, 0], enriched["atr_contraction_ratio"].dropna().clip(upper=3),
          "ATR Contraction Ratio")
    _hist(axes[0, 1], enriched["base_depth_pct"].dropna().clip(upper=60),
          "Base Depth %")
    _hist(axes[1, 0], enriched["return_20d"].clip(-50, 50), "20-day Return %")
    _hist_thresh(axes[1, 1], enriched["sector_relative_strength"].clip(-30, 30),
                 "Sector-Relative Strength",
                 thresh=0, thresh_label="Threshold = 0 (outperform)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    capture_figure(fig, "quality_features")

    pct_sector_pos = (enriched["sector_relative_strength"] > 0).mean() * 100
    report_data["quality"] = {
        "median_atr_contraction": round(float(enriched["atr_contraction_ratio"].median()), 2)
            if enriched["atr_contraction_ratio"].notna().any() else None,
        "median_base_depth": round(float(enriched["base_depth_pct"].median()), 2)
            if enriched["base_depth_pct"].notna().any() else None,
        "median_return_20d": round(float(enriched["return_20d"].median()), 2),
        "pct_sector_outperform": round(pct_sector_pos, 1),
    }

    # --- 2e. Metadata ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Metadata: Market Cap & Sector", fontsize=16, fontweight="bold")

    mc = enriched["market_cap"].dropna() / 1e9
    axes[0].hist(mc, bins=50, edgecolor="white", alpha=0.8)
    axes[0].set_title("Market Cap Distribution", fontweight="bold")
    axes[0].set_xlabel("Market Cap ($B)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(cfg.market_cap_min / 1e9, color="red", ls="--", label=f"Min = ${cfg.market_cap_min/1e9:.1f}B")
    axes[0].axvline(cfg.market_cap_max / 1e9, color="red", ls="--", label=f"Max = ${cfg.market_cap_max/1e9:.0f}B")
    axes[0].legend()

    sector_counts = enriched["sector"].value_counts()
    colors = sns.color_palette("muted", len(sector_counts))
    axes[1].barh(sector_counts.index[::-1], sector_counts.values[::-1], color=colors[::-1], edgecolor="white")
    axes[1].set_title("Sector Breakdown", fontweight="bold")
    axes[1].set_xlabel("Count")
    for i, v in enumerate(sector_counts.values[::-1]):
        axes[1].text(v + max(sector_counts.values)*0.01, i,
                     f"{v:,} ({v/len(enriched)*100:.1f}%)", va="center", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    capture_figure(fig, "metadata_features")

    report_data["metadata_dist"] = {
        "sector_counts": sector_counts.to_dict(),
        "n_sectors": len(sector_counts),
        "top_sector": sector_counts.index[0],
        "top_sector_pct": round(sector_counts.iloc[0] / len(enriched) * 100, 1),
    }
    print("  Done.")


# ============================================================================
# SECTION 3 — Funnel / Filter Cascade Analysis
# ============================================================================
def analyze_funnel(enriched: pd.DataFrame) -> None:
    print("[Section 3] Replaying filter cascade...")

    df = enriched.copy()
    initial = len(df)

    filter_steps: list[tuple[str, pd.Series]] = [
        ("Price > 50 DMA",
         df["above_50dma"] == True),  # noqa: E712
        ("Price > 200 DMA (or within 5%)",
         (df["above_200dma"] == True) | ((df["last_close"] / df["sma_200"] - 1) * 100 >= -cfg.max_pct_below_200dma)),  # noqa: E712
        (f"Within {cfg.max_pct_from_52wk_high}% of 52-wk high",
         df["pct_from_52wk_high"] >= -cfg.max_pct_from_52wk_high),
        (f"Relative volume (3d max) >= {cfg.min_relative_volume}",
         df["rel_volume_3d_max"] >= cfg.min_relative_volume),
        (f"Avg daily $ vol >= ${cfg.min_avg_daily_dollar_volume/1e6:.0f}M",
         df["avg_daily_dollar_volume"] >= cfg.min_avg_daily_dollar_volume),
        ("Breakout detected",
         df["breakout_detected"] == True),  # noqa: E712
        (f"Distance from breakout <= {cfg.max_pct_above_breakout}%",
         df["pct_above_breakout"].fillna(999) <= cfg.max_pct_above_breakout),
        ("Sector-relative strength > 0",
         df["sector_relative_strength"] > 0),
    ]
    if "breakout_volume_ratio" in df.columns:
        filter_steps.append(
            (f"Breakout volume >= {cfg.min_breakout_volume_ratio}x avg",
             df["breakout_volume_ratio"].fillna(0) >= cfg.min_breakout_volume_ratio))

    rows = []
    for name, mask in filter_steps:
        before = len(df)
        df = df.loc[mask]
        after = len(df)
        killed = before - after
        kill_rate = killed / before * 100 if before > 0 else 0
        cumulative = after / initial * 100
        rows.append({
            "Filter": name,
            "In": f"{before:,}",
            "Out": f"{after:,}",
            "Killed": f"{killed:,}",
            "Kill Rate": f"{kill_rate:.1f}%",
            "Cumulative Survival": f"{cumulative:.2f}%",
        })

    funnel_df = pd.DataFrame(rows)
    report_data["funnel"] = {
        "table": rows,
        "initial": initial,
        "final": len(df),
        "overall_survival_pct": round(len(df) / initial * 100, 3),
    }

    # Bar chart of kill counts
    kills = [int(r["Killed"].replace(",", "")) for r in rows]
    names_short = [r["Filter"][:35] for r in rows]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(names_short[::-1], kills[::-1], color=sns.color_palette("Reds_r", len(kills)),
                   edgecolor="white")
    ax.set_title("Stocks Killed per Filter Step", fontsize=14, fontweight="bold")
    ax.set_xlabel("Stocks removed")
    for bar, k in zip(bars, kills[::-1]):
        ax.text(bar.get_width() + max(kills)*0.01, bar.get_y() + bar.get_height()/2,
                f"{k:,}", va="center", fontsize=10)
    fig.tight_layout()
    capture_figure(fig, "funnel_bar")

    # Cumulative survival line
    survivals = [initial]
    for r in rows:
        survivals.append(int(r["Out"].replace(",", "")))
    labels = ["Start"] + [r["Filter"][:28] for r in rows]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(len(survivals)), survivals, "o-", color="#2c3e50", linewidth=2, markersize=8)
    ax.fill_between(range(len(survivals)), survivals, alpha=0.15, color="#3498db")
    ax.set_xticks(range(len(survivals)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Stocks remaining")
    ax.set_title("Cumulative Filter Funnel", fontsize=14, fontweight="bold")
    for i, s in enumerate(survivals):
        ax.annotate(f"{s:,}", (i, s), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)
    fig.tight_layout()
    capture_figure(fig, "funnel_cumulative")
    print("  Done.")


# ============================================================================
# SECTION 4 — Correlation & Redundancy
# ============================================================================
def analyze_correlations(enriched: pd.DataFrame) -> None:
    print("[Section 4] Computing correlations...")

    numeric = enriched.select_dtypes(include=[np.number]).copy()
    drop_cols = [c for c in ["breakout_level", "market_cap"] if c in numeric.columns]
    numeric = numeric.drop(columns=drop_cols, errors="ignore")

    corr_pearson = numeric.corr(method="pearson")
    corr_spearman = numeric.corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle("Feature Correlations", fontsize=16, fontweight="bold")
    mask = np.triu(np.ones_like(corr_pearson, dtype=bool))
    sns.heatmap(corr_pearson, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, ax=axes[0], annot_kws={"size": 7},
                xticklabels=True, yticklabels=True)
    axes[0].set_title("Pearson Correlation", fontweight="bold")
    axes[0].tick_params(labelsize=7)

    sns.heatmap(corr_spearman, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, ax=axes[1], annot_kws={"size": 7},
                xticklabels=True, yticklabels=True)
    axes[1].set_title("Spearman Correlation", fontweight="bold")
    axes[1].tick_params(labelsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    capture_figure(fig, "correlation_heatmaps")

    # Flag high correlations
    high_corr = []
    cols = corr_spearman.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r_s = corr_spearman.iloc[i, j]
            r_p = corr_pearson.iloc[i, j]
            if abs(r_s) > 0.7 or abs(r_p) > 0.7:
                high_corr.append({
                    "Feature A": cols[i],
                    "Feature B": cols[j],
                    "Pearson": round(r_p, 3),
                    "Spearman": round(r_s, 3),
                })
    report_data["correlations"] = {
        "high_corr_pairs": high_corr,
        "n_high_corr": len(high_corr),
    }

    # Specific redundancy questions
    def _corr(a: str, b: str) -> tuple[float, float]:
        valid = enriched[[a, b]].dropna()
        if len(valid) < 10:
            return (np.nan, np.nan)
        return (round(float(valid[a].corr(valid[b])), 3),
                round(float(valid[a].corr(valid[b], method="spearman")), 3))

    redundancy = {}
    redundancy["rel_vol_latest_vs_3d_max"] = _corr("rel_volume_latest", "rel_volume_3d_max")
    redundancy["return_20d_vs_sector_rel"] = _corr("return_20d", "sector_relative_strength")

    above_50 = enriched["above_50dma"] == True  # noqa: E712
    above_200 = enriched["above_200dma"] == True  # noqa: E712
    redundancy["above_200_is_subset_of_50"] = {
        "both_true": int((above_50 & above_200).sum()),
        "above_200_but_not_50": int((~above_50 & above_200).sum()),
        "pct_200_implies_50": round(
            (above_50 & above_200).sum() / max(above_200.sum(), 1) * 100, 1),
    }

    report_data["redundancy"] = redundancy
    print("  Done.")


# ============================================================================
# SECTION 5 — Data Quality
# ============================================================================
def analyze_quality(enriched: pd.DataFrame) -> None:
    print("[Section 5] Assessing data quality...")

    missing = enriched.isna().mean().sort_values(ascending=False) * 100
    missing = missing[missing > 0]

    fig, ax = plt.subplots(figsize=(12, max(4, len(missing) * 0.4)))
    if not missing.empty:
        ax.barh(missing.index[::-1], missing.values[::-1],
                color=sns.color_palette("YlOrRd", len(missing))[::-1], edgecolor="white")
        for i, v in enumerate(missing.values[::-1]):
            ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=10)
        ax.set_title("Missing Values by Column", fontweight="bold")
        ax.set_xlabel("% Missing")
    else:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_title("Missing Values by Column", fontweight="bold")
    fig.tight_layout()
    capture_figure(fig, "missing_values")

    report_data["quality"] = {
        "missing_cols": {k: round(v, 1) for k, v in missing.to_dict().items()},
        "n_cols_with_missing": len(missing),
        "total_tickers": len(enriched),
    }

    # Outliers (>3 std)
    numeric = enriched.select_dtypes(include=[np.number])
    outlier_info = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) < 10:
            continue
        mean, std = s.mean(), s.std()
        if std == 0:
            continue
        n_out = int(((s - mean).abs() > 3 * std).sum())
        if n_out > 0:
            outlier_info[col] = {"n_outliers": n_out, "pct": round(n_out / len(s) * 100, 2)}
    report_data["quality"]["outliers"] = outlier_info

    # Metadata completeness
    meta_cols = ["sector", "industry", "market_cap"]
    meta_missing = {}
    for c in meta_cols:
        if c in enriched.columns:
            if enriched[c].dtype == object:
                n_miss = int((enriched[c].isna() | (enriched[c] == "")).sum())
            else:
                n_miss = int(enriched[c].isna().sum())
            meta_missing[c] = n_miss
    report_data["quality"]["metadata_missing"] = meta_missing
    print("  Done.")


# ============================================================================
# SECTION 6 — Survivor Profile
# ============================================================================
def analyze_survivors(enriched: pd.DataFrame) -> None:
    print("[Section 6] Building survivor profile...")

    df = enriched.copy()

    # Replay all filters to get survivor mask on ORIGINAL df
    mask = pd.Series(True, index=df.index)
    mask &= df["above_50dma"] == True  # noqa: E712
    mask &= ((df["above_200dma"] == True) |  # noqa: E712
             ((df["last_close"] / df["sma_200"] - 1) * 100 >= -cfg.max_pct_below_200dma))
    mask &= df["pct_from_52wk_high"] >= -cfg.max_pct_from_52wk_high
    mask &= df["rel_volume_3d_max"] >= cfg.min_relative_volume
    mask &= df["avg_daily_dollar_volume"] >= cfg.min_avg_daily_dollar_volume
    mask &= df["breakout_detected"] == True  # noqa: E712
    mask &= df["pct_above_breakout"].fillna(999) <= cfg.max_pct_above_breakout
    mask &= df["sector_relative_strength"] > 0
    if "breakout_volume_ratio" in df.columns:
        mask &= df["breakout_volume_ratio"].fillna(0) >= cfg.min_breakout_volume_ratio

    survivors = df[mask]
    eliminated = df[~mask]

    report_data["survivors"] = {
        "n_survivors": len(survivors),
        "n_eliminated": len(eliminated),
        "survival_pct": round(len(survivors) / len(df) * 100, 3),
    }

    compare_cols = ["last_close", "pct_from_52wk_high", "rel_volume_3d_max",
                    "avg_daily_dollar_volume", "return_20d", "sector_relative_strength",
                    "atr_contraction_ratio", "base_depth_pct"]
    compare_cols = [c for c in compare_cols if c in df.columns]

    # Side-by-side comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Survivors (n={len(survivors)}) vs Eliminated (n={len(eliminated)})",
                 fontsize=16, fontweight="bold")
    for i, col in enumerate(compare_cols):
        ax = axes[i // 4, i % 4]
        s_data = survivors[col].dropna()
        e_data = eliminated[col].dropna()

        clip_lo = min(e_data.quantile(0.02) if len(e_data) > 10 else e_data.min(),
                      s_data.quantile(0.02) if len(s_data) > 10 else (s_data.min() if not s_data.empty else 0))
        clip_hi = max(e_data.quantile(0.98) if len(e_data) > 10 else e_data.max(),
                      s_data.quantile(0.98) if len(s_data) > 10 else (s_data.max() if not s_data.empty else 1))

        bins = np.linspace(clip_lo, clip_hi, 40)
        ax.hist(e_data.clip(clip_lo, clip_hi), bins=bins, alpha=0.5, label="Eliminated", density=True)
        if not s_data.empty:
            ax.hist(s_data.clip(clip_lo, clip_hi), bins=bins, alpha=0.7, label="Survivors", density=True)
        ax.set_title(col.replace("_", " ").title(), fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    capture_figure(fig, "survivor_comparison")

    # Sector composition comparison
    surv_sectors = survivors["sector"].value_counts(normalize=True).round(3) * 100
    all_sectors = enriched["sector"].value_counts(normalize=True).round(3) * 100

    sector_comp = pd.DataFrame({
        "Universe %": all_sectors,
        "Survivors %": surv_sectors,
    }).fillna(0).sort_values("Universe %", ascending=False)
    sector_comp["Delta"] = (sector_comp["Survivors %"] - sector_comp["Universe %"]).round(1)
    report_data["survivors"]["sector_comparison"] = sector_comp.to_dict()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(sector_comp))
    w = 0.35
    ax.bar(x - w/2, sector_comp["Universe %"], w, label="Universe", alpha=0.8)
    ax.bar(x + w/2, sector_comp["Survivors %"], w, label="Survivors", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sector_comp.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("% of group")
    ax.set_title("Sector Composition: Universe vs Survivors", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    capture_figure(fig, "sector_composition")

    # Market cap comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    mc_all = enriched["market_cap"].dropna() / 1e9
    mc_surv = survivors["market_cap"].dropna() / 1e9
    bins = np.linspace(0, cfg.market_cap_max / 1e9, 40)
    ax.hist(mc_all, bins=bins, alpha=0.5, label="Universe", density=True)
    if not mc_surv.empty:
        ax.hist(mc_surv, bins=bins, alpha=0.7, label="Survivors", density=True)
    ax.set_title("Market Cap: Universe vs Survivors", fontweight="bold")
    ax.set_xlabel("Market Cap ($B)")
    ax.legend()
    fig.tight_layout()
    capture_figure(fig, "market_cap_comparison")
    print("  Done.")


# ============================================================================
# Helper plot functions
# ============================================================================
def _hist(ax, series, title, clip_q=None, bins=50):
    data = series.dropna()
    if clip_q:
        data = data[data <= data.quantile(clip_q)]
    ax.hist(data, bins=bins, edgecolor="white", alpha=0.8)
    ax.set_title(title, fontweight="bold", fontsize=11)
    stats = f"n={len(data):,}  med={data.median():.2f}"
    ax.text(0.97, 0.93, stats, transform=ax.transAxes, fontsize=8,
            va="top", ha="right", bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))


def _hist_thresh(ax, series, title, thresh, thresh_label="", bins=50, log_scale=False):
    data = series.dropna()
    ax.hist(data, bins=bins, edgecolor="white", alpha=0.8)
    ax.axvline(thresh, color="red", ls="--", lw=2, label=thresh_label)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    if log_scale:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"${x/1e6:.0f}M" if x >= 1e6 else f"${x:,.0f}"))
    pct_pass = (series.dropna() >= thresh).mean() * 100
    ax.text(0.97, 0.93, f"{pct_pass:.1f}% pass", transform=ax.transAxes, fontsize=9,
            va="top", ha="right", bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))


def _bool_bar(ax, series, title):
    counts = series.value_counts().sort_index()
    labels = ["False", "True"] if len(counts) == 2 else [str(x) for x in counts.index]
    colors = ["#e74c3c", "#2ecc71"][:len(counts)]
    ax.bar(labels, counts.values, color=colors, edgecolor="white")
    for i, v in enumerate(counts.values):
        pct = v / len(series) * 100
        ax.text(i, v + len(series)*0.01, f"{v:,}\n({pct:.1f}%)", ha="center", fontsize=10)
    ax.set_title(title, fontweight="bold", fontsize=11)


# ============================================================================
# REPORT ASSEMBLY
# ============================================================================
def assemble_report() -> str:
    print("[Report] Assembling markdown report...")

    _pm = report_data.get("price_ma", {})
    _vol = report_data.get("volume", {})
    _bo = report_data.get("breakout", {})
    _qual = report_data.get("quality", {})
    _fun = report_data.get("funnel", {})
    _corr = report_data.get("correlations", {})
    _surv = report_data.get("survivors", {})
    _qd = report_data.get("quality", {})
    _md = report_data.get("metadata_dist", {})

    lines: list[str] = []
    L = lines.append

    L("# Pipeline EDA Report — Full Phase 1 Data Exploration\n")
    L(f"**Generated:** {analysis_timestamp}  ")
    L(f"**Config:** market_cap {cfg.market_cap_min/1e6:.0f}M–{cfg.market_cap_max/1e9:.0f}B, "
      f"min_vol_3m={cfg.min_avg_volume_3m:,}, "
      f"breakout_window={cfg.recent_breakout_window}d\n")
    L("---\n")

    # --- Executive Summary ---
    L("## 1. Executive Summary\n")
    L("| Metric | Value |")
    L("|--------|-------|")
    L(f"| Universe size (after OHLCV + technicals) | **{_qd.get('total_tickers', 'N/A'):,}** |")
    L(f"| % above 50 DMA | {_pm.get('pct_above_50dma', 'N/A')}% |")
    L(f"| % above 200 DMA | {_pm.get('pct_above_200dma', 'N/A')}% |")
    L(f"| % with golden cross (50 > 200 DMA) | {_pm.get('pct_golden_cross', 'N/A')}% |")
    L(f"| % within {cfg.max_pct_from_52wk_high}% of 52-wk high | {_pm.get('pct_within_52wk_high', 'N/A')}% |")
    L(f"| % with rel volume 3d max >= {cfg.min_relative_volume}x | {_vol.get('pct_pass_rel_volume', 'N/A')}% |")
    L(f"| % with avg $ vol >= ${cfg.min_avg_daily_dollar_volume/1e6:.0f}M | {_vol.get('pct_pass_dollar_volume', 'N/A')}% |")
    L(f"| % with breakout detected | {_bo.get('pct_breakout_detected', 'N/A')}% |")
    L(f"| % outperforming sector (20d) | {_qual.get('pct_sector_outperform', 'N/A') if isinstance(_qual, dict) else 'N/A'}% |")
    L(f"| Survivors (pass all filters) | **{_surv.get('n_survivors', 'N/A')}** ({_surv.get('survival_pct', 'N/A')}%) |")
    L(f"| Sectors represented | {_md.get('n_sectors', 'N/A')} |")
    L(f"| Dominant sector | {_md.get('top_sector', 'N/A')} ({_md.get('top_sector_pct', 'N/A')}%) |")
    L("")

    L("---\n")

    # --- Section 2: Feature Distributions ---
    L("## 2. Feature Distributions\n")

    L("### 2.1 Price / Moving Average Features\n")
    L(f"- **{_pm.get('pct_above_50dma', '?')}%** of the universe trades above its 50 DMA")
    L(f"- **{_pm.get('pct_above_200dma', '?')}%** trades above its 200 DMA")
    L(f"- **{_pm.get('pct_golden_cross', '?')}%** have a golden cross (50 DMA > 200 DMA)")
    L(f"- **{_pm.get('pct_within_52wk_high', '?')}%** are within {cfg.max_pct_from_52wk_high}% of their 52-week high")
    L(f"- Median distance from 52-wk high: **{_pm.get('median_pct_from_52wk', '?')}%**\n")
    if "price_ma_features" in report_figures:
        L(f"![Price / MA Features](data:image/png;base64,{report_figures['price_ma_features']})\n")

    L("### 2.2 Volume Features\n")
    L(f"- **{_vol.get('pct_pass_rel_volume', '?')}%** have 3-day max relative volume >= {cfg.min_relative_volume}x")
    L(f"- **{_vol.get('pct_pass_dollar_volume', '?')}%** have avg daily dollar volume >= ${cfg.min_avg_daily_dollar_volume/1e6:.0f}M")
    L(f"- Median 3d-max relative volume: **{_vol.get('median_rel_vol_3d_max', '?')}x**")
    L(f"- Median avg daily dollar volume: **${_vol.get('median_dollar_vol', 0)/1e6:.1f}M**\n")
    if "volume_features" in report_figures:
        L(f"![Volume Features](data:image/png;base64,{report_figures['volume_features']})\n")

    L("### 2.3 Breakout Features\n")
    L(f"- **{_bo.get('pct_breakout_detected', '?')}%** of the universe ({_bo.get('n_breakouts', '?')} stocks) have a recent breakout")
    tf = _bo.get("timeframe_counts", {})
    if tf:
        L("- Breakout timeframe distribution:")
        for tf_key in sorted(tf.keys()):
            L(f"  - {int(tf_key)}-day lookback: **{tf[tf_key]}** breakouts")
    L("")
    if "breakout_features" in report_figures:
        L(f"![Breakout Features](data:image/png;base64,{report_figures['breakout_features']})\n")

    L("### 2.4 Quality / Structure Features\n")
    q = report_data.get("quality", {})
    if isinstance(q, dict) and "median_atr_contraction" in q:
        L(f"- Median ATR contraction ratio: **{q.get('median_atr_contraction', '?')}** (< 1.0 = consolidation)")
        L(f"- Median base depth: **{q.get('median_base_depth', '?')}%**")
        L(f"- Median 20-day return: **{q.get('median_return_20d', '?')}%**")
        L(f"- **{q.get('pct_sector_outperform', '?')}%** outperform their sector over 20 days\n")
    if "quality_features" in report_figures:
        L(f"![Quality Features](data:image/png;base64,{report_figures['quality_features']})\n")

    L("### 2.5 Metadata (Market Cap & Sector)\n")
    if "metadata_features" in report_figures:
        L(f"![Metadata](data:image/png;base64,{report_figures['metadata_features']})\n")

    sector_counts = _md.get("sector_counts", {})
    if sector_counts:
        L("| Sector | Count | % |")
        L("|--------|------:|---:|")
        total = sum(sector_counts.values())
        for sec, cnt in sorted(sector_counts.items(), key=lambda x: -x[1]):
            L(f"| {sec} | {cnt:,} | {cnt/total*100:.1f}% |")
        L("")

    L("---\n")

    # --- Section 3: Funnel ---
    L("## 3. Filter Funnel Analysis\n")
    funnel_rows = _fun.get("table", [])
    if funnel_rows:
        L("| Filter | In | Out | Killed | Kill Rate | Cumulative Survival |")
        L("|--------|---:|----:|-------:|----------:|--------------------:|")
        for r in funnel_rows:
            L(f"| {r['Filter']} | {r['In']} | {r['Out']} | {r['Killed']} | {r['Kill Rate']} | {r['Cumulative Survival']} |")
        L("")
        L(f"**Overall survival:** {_fun.get('initial', '?'):,} → {_fun.get('final', '?'):,} "
          f"(**{_fun.get('overall_survival_pct', '?')}%**)\n")

    if "funnel_bar" in report_figures:
        L(f"![Funnel Bar](data:image/png;base64,{report_figures['funnel_bar']})\n")
    if "funnel_cumulative" in report_figures:
        L(f"![Funnel Cumulative](data:image/png;base64,{report_figures['funnel_cumulative']})\n")

    # Identify hardest filters and near-redundant
    if funnel_rows:
        sorted_by_kill = sorted(funnel_rows, key=lambda r: -int(r["Killed"].replace(",", "")))
        L("**Top 3 hardest filters (most stocks killed):**\n")
        for i, r in enumerate(sorted_by_kill[:3], 1):
            L(f"{i}. **{r['Filter']}** — killed {r['Killed']} ({r['Kill Rate']})")
        L("")

        near_redundant = [r for r in funnel_rows if int(r["Killed"].replace(",", "")) <= 5]
        if near_redundant:
            L("**Near-redundant filters (killed <= 5 stocks):**\n")
            for r in near_redundant:
                L(f"- **{r['Filter']}** — killed only {r['Killed']} ({r['Kill Rate']})")
            L("")

    L("---\n")

    # --- Section 4: Correlations ---
    L("## 4. Correlation & Redundancy Analysis\n")
    if "correlation_heatmaps" in report_figures:
        L(f"![Correlations](data:image/png;base64,{report_figures['correlation_heatmaps']})\n")

    high_corr = _corr.get("high_corr_pairs", [])
    if high_corr:
        L(f"**{len(high_corr)} feature pairs with |correlation| > 0.7:**\n")
        L("| Feature A | Feature B | Pearson | Spearman |")
        L("|-----------|-----------|--------:|---------:|")
        for p in sorted(high_corr, key=lambda x: -abs(x["Spearman"])):
            L(f"| {p['Feature A']} | {p['Feature B']} | {p['Pearson']} | {p['Spearman']} |")
        L("")
    else:
        L("No feature pairs with |correlation| > 0.7 found.\n")

    redundancy = report_data.get("redundancy", {})
    L("### Specific Redundancy Checks\n")

    rv = redundancy.get("rel_vol_latest_vs_3d_max", (None, None))
    L(f"- `rel_volume_latest` vs `rel_volume_3d_max`: Pearson={rv[0]}, Spearman={rv[1]}")

    rd = redundancy.get("return_20d_vs_sector_rel", (None, None))
    L(f"- `return_20d` vs `sector_relative_strength`: Pearson={rd[0]}, Spearman={rd[1]}")

    sub = redundancy.get("above_200_is_subset_of_50", {})
    L(f"- `above_200dma` ⊂ `above_50dma`? "
      f"{sub.get('pct_200_implies_50', '?')}% of stocks above 200 DMA are also above 50 DMA "
      f"(only {sub.get('above_200_but_not_50', '?')} exceptions)\n")

    L("---\n")

    # --- Section 5: Data Quality ---
    L("## 5. Data Quality\n")
    if "missing_values" in report_figures:
        L(f"![Missing Values](data:image/png;base64,{report_figures['missing_values']})\n")

    missing_cols = _qd.get("missing_cols", {})
    if missing_cols:
        L("| Column | % Missing |")
        L("|--------|----------:|")
        for col, pct in sorted(missing_cols.items(), key=lambda x: -x[1]):
            L(f"| {col} | {pct}% |")
        L("")
    else:
        L("No missing values found across all columns.\n")

    outliers = _qd.get("outliers", {})
    if outliers:
        L(f"**Outliers (> 3 std from mean):** {len(outliers)} columns have outliers\n")
        L("| Column | # Outliers | % |")
        L("|--------|----------:|---:|")
        for col, info in sorted(outliers.items(), key=lambda x: -x[1]["n_outliers"]):
            L(f"| {col} | {info['n_outliers']} | {info['pct']}% |")
        L("")

    meta_miss = _qd.get("metadata_missing", {})
    if meta_miss:
        L("**Metadata completeness:**\n")
        L("| Field | # Missing |")
        L("|-------|----------:|")
        for c, n in meta_miss.items():
            L(f"| {c} | {n} |")
        L("")

    L("---\n")

    # --- Section 6: Survivor Profile ---
    L("## 6. Survivor Profile\n")
    L(f"- **{_surv.get('n_survivors', '?')}** stocks pass all quantitative filters "
      f"(**{_surv.get('survival_pct', '?')}%** of {_qd.get('total_tickers', '?'):,})")
    L(f"- **{_surv.get('n_eliminated', '?'):,}** eliminated\n")

    if "survivor_comparison" in report_figures:
        L(f"![Survivor vs Eliminated](data:image/png;base64,{report_figures['survivor_comparison']})\n")
    if "sector_composition" in report_figures:
        L(f"![Sector Composition](data:image/png;base64,{report_figures['sector_composition']})\n")
    if "market_cap_comparison" in report_figures:
        L(f"![Market Cap Comparison](data:image/png;base64,{report_figures['market_cap_comparison']})\n")

    L("---\n")
    L(f"*Report generated at {analysis_timestamp}*\n")

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    print("=" * 60)
    print("PIPELINE EDA — Full Phase 1 Data Exploration")
    print("=" * 60)

    enriched, metadata = load_data()

    report_data["metadata"] = {
        "timestamp": analysis_timestamp,
        "config": {k: str(v) for k, v in asdict(cfg).items()},
        "n_tickers": len(enriched),
        "n_columns": enriched.shape[1],
    }

    analyze_distributions(enriched)
    analyze_funnel(enriched)
    analyze_correlations(enriched)
    analyze_quality(enriched)
    analyze_survivors(enriched)

    report_md = assemble_report()
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    print(f"\nReport saved to: {REPORT_PATH}")
    print(f"  Lines: {len(report_md.splitlines())}")
    print(f"  Characters: {len(report_md):,}")
    print(f"  Figures embedded: {len(report_figures)}")
    print("Done.")


if __name__ == "__main__":
    main()
