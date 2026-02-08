"""Growth analysis: rate calculation, prediction, remaining life, risk scoring.

Computes corrosion growth metrics from matched anomaly pairs across ILI runs
and predicts future wall loss. Provides remaining-life estimates, risk scoring,
multi-run trend extrapolation, and summary statistics for the Growth Analysis
dashboard.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from config import MAX_PLAUSIBLE_GROWTH_RATE, WALL_LOSS_REPAIR_THRESHOLD


def calculate_growth_rates(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive growth metrics to matches DataFrame.

    Enriches the matched-anomaly pairs with:
      - remaining_wall_pct: percentage of wall still intact
      - remaining_life_years: years until depth reaches 80% wall loss
        (the repair threshold) at the current growth rate
      - growth_class: categorical label for the growth rate band
      - risk_score: composite 0-100 urgency score
      - risk_category: categorical label derived from risk_score
    """
    if matches_df.empty:
        return matches_df

    df = matches_df.copy()

    # Remaining wall (% intact)
    df["remaining_wall_pct"] = 100.0 - df["later_depth_pct"]

    # Remaining life at current growth rate (years until 80% wall loss)
    df["remaining_life_years"] = df.apply(_remaining_life, axis=1)

    # Growth classification
    df["growth_class"] = df["depth_growth_rate"].apply(_classify_growth)

    # Risk score (0-100): higher = more urgent
    df["risk_score"] = df.apply(_compute_risk_score, axis=1)

    # Risk category
    df["risk_category"] = df["risk_score"].apply(_classify_risk)

    return df


def _remaining_life(row) -> float:
    """Estimate years until depth reaches repair threshold.

    Calculates years until the anomaly depth reaches 80% wall loss (the
    repair threshold) at the current growth rate. Returns NaN for stable
    or shrinking anomalies (rate <= 0) since they are not progressing
    toward the threshold.
    """
    rate = row.get("depth_growth_rate", np.nan)
    current = row.get("later_depth_pct", np.nan)
    if np.isnan(rate) or np.isnan(current) or rate <= 0:
        return np.nan
    remaining_capacity = WALL_LOSS_REPAIR_THRESHOLD - current
    if remaining_capacity <= 0:
        return 0.0
    return round(remaining_capacity / rate, 1)


def _classify_growth(rate) -> str:
    """Classify growth rate into categories.

    Classification bands (%/yr):
      rate < 0  -> Apparent Shrinkage (likely a measurement artifact)
      rate == 0 -> Stable
      0 < rate <= 1 -> Low
      1 < rate <= 3 -> Moderate
      3 < rate <= 5 -> High
      rate > 5      -> Severe
    """
    if pd.isna(rate):
        return "Unknown"
    if rate < 0:
        return "Apparent Shrinkage"
    if rate == 0:
        return "Stable"
    if rate <= 1.0:
        return "Low"
    if rate <= 3.0:
        return "Moderate"
    if rate <= MAX_PLAUSIBLE_GROWTH_RATE:
        return "High"
    return "Severe"


def _compute_risk_score(row) -> float:
    """Compute risk score (0-100) based on current depth and growth rate.

    Composite score with two equally-weighted components:
      - depth_component (0-50): current depth normalized by the 80% wall
        loss repair threshold; deeper anomalies score higher.
      - rate_component (0-50): growth rate normalized by the 5%/yr
        plausible-growth cap; faster-growing anomalies score higher.
    Each component is clamped to its 0-50 range before summing.
    """
    depth = row.get("later_depth_pct", 0)
    rate = row.get("depth_growth_rate", 0)
    if pd.isna(depth):
        depth = 0
    if pd.isna(rate) or rate < 0:
        rate = 0

    # Depth component (0-50): deeper = riskier
    depth_score = min(50, depth * 50 / WALL_LOSS_REPAIR_THRESHOLD)

    # Growth rate component (0-50): faster = riskier
    rate_score = min(50, rate * 50 / MAX_PLAUSIBLE_GROWTH_RATE)

    return round(depth_score + rate_score, 1)


def _classify_risk(score) -> str:
    """Classify the numeric risk score into a named category.

    Thresholds:
      score >= 70 -> Critical
      score >= 50 -> High
      score >= 30 -> Medium
      score <  30 -> Low
    """
    if pd.isna(score):
        return "Unknown"
    if score >= 70:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 30:
        return "Medium"
    return "Low"


# ── Multi-run growth trend prediction ─────────────────────────────────────────

def predict_growth_trends(multi_run_matches: pd.DataFrame) -> pd.DataFrame:
    """For anomalies tracked across 3 runs, fit growth models and predict.

    Fits a linear regression on 3-run depth data (2007, 2015, 2022) to
    extrapolate projected depth to 2030 and 2035. When all three data
    points are available, also fits a quadratic curve and flags the anomaly
    as accelerating if the quadratic coefficient is positive (i.e. the
    growth rate is increasing over time).

    Expects columns: depth_2007, depth_2015, depth_2022 (from multi_run chaining).
    """
    if multi_run_matches.empty:
        return multi_run_matches

    df = multi_run_matches.copy()
    years = np.array([2007, 2015, 2022], dtype=float)

    predictions = []
    for _, row in df.iterrows():
        depths = np.array([
            row.get("depth_2007", np.nan),
            row.get("depth_2015", np.nan),
            row.get("depth_2022", np.nan),
        ])

        valid = ~np.isnan(depths)
        n_valid = valid.sum()

        pred = {
            "linear_rate": np.nan,
            "linear_r2": np.nan,
            "predicted_2030": np.nan,
            "predicted_2035": np.nan,
            "is_accelerating": False,
        }

        if n_valid >= 2:
            y_valid = years[valid]
            d_valid = depths[valid]

            # Linear fit
            slope, intercept, r_value, _, _ = sp_stats.linregress(y_valid, d_valid)
            pred["linear_rate"] = round(slope, 4)
            pred["linear_r2"] = round(r_value ** 2, 4)
            pred["predicted_2030"] = round(slope * 2030 + intercept, 1)
            pred["predicted_2035"] = round(slope * 2035 + intercept, 1)

        if n_valid == 3:
            # Check for acceleration (quadratic fit)
            coeffs = np.polyfit(years[valid], depths[valid], 2)
            pred["is_accelerating"] = coeffs[0] > 0  # Positive quadratic = acceleration

        predictions.append(pred)

    pred_df = pd.DataFrame(predictions, index=df.index)
    return pd.concat([df, pred_df], axis=1)


def growth_summary_stats(matches_df: pd.DataFrame) -> dict:
    """Compute aggregate growth statistics for the dashboard.

    Returns a dict of summary metrics (count, mean, median, std, min/max,
    and percentage breakdowns by severity) used to populate the Growth
    Analysis dashboard cards and charts.
    """
    if matches_df.empty:
        return {}

    rates = matches_df["depth_growth_rate"].dropna()
    return {
        "count": len(rates),
        "mean_rate": round(rates.mean(), 3),
        "median_rate": round(rates.median(), 3),
        "std_rate": round(rates.std(), 3),
        "min_rate": round(rates.min(), 3),
        "max_rate": round(rates.max(), 3),
        "pct_negative": round((rates < 0).mean() * 100, 1),
        "pct_high": round((rates > 3.0).mean() * 100, 1),
        "pct_severe": round((rates > MAX_PLAUSIBLE_GROWTH_RATE).mean() * 100, 1),
    }


def top_concerns(matches_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return the top N highest-risk anomalies.

    Selects the N anomalies with the highest risk_score and returns a
    subset of key columns (location, depth, growth rate, remaining life,
    risk info, and confidence) for display in the dashboard table.
    """
    if matches_df.empty or "risk_score" not in matches_df.columns:
        return pd.DataFrame()
    return matches_df.nlargest(n, "risk_score")[[
        "later_joint", "later_distance", "later_clock",
        "later_depth_pct", "depth_growth_rate", "remaining_life_years",
        "risk_score", "risk_category", "confidence_label",
    ]].reset_index(drop=True)
