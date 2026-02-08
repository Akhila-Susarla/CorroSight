"""Virtual ILI: Predict future inspection findings from historical growth trends.

"Virtual ILI" is an industry concept where historical corrosion growth data from
prior inline inspections is used to predict what a future inspection would find,
without physically running the ILI tool through the pipeline. This avoids the
cost and operational disruption of an actual run while still informing integrity
management decisions such as dig scheduling and repair prioritization.
"""

import numpy as np
import pandas as pd

from config import WALL_LOSS_REPAIR_THRESHOLD, MAX_PLAUSIBLE_GROWTH_RATE


def predict_future_inspection(cache: dict, target_year: int) -> dict:
    """Simulate what a future ILI run would find at target_year.

    Workflow:
      1. Uses 2015-2022 pairwise matches as the primary growth-rate source
         (most recent inspection pair, so rates reflect current conditions).
      2. For triple-tracked anomalies (matched across all three runs), prefers
         the 3-run linear regression rate over the pairwise rate because more
         data points yield a better trend estimate.
      3. Filters out negative growth rates -- these are measurement artifacts
         (tool tolerance, reporting differences), not real wall recovery.
      4. Extrapolates depth forward:
             predicted_depth = current_depth + (rate * years_forward)
         and clamps the result to [0, 100] (physical wall-loss percentage bounds).
      5. Classifies predicted risk:
             Critical >= 70%, High >= 50%, Medium >= 30%, Low < 30%.
      6. Computes threshold crossings: how many anomalies newly exceed 50/60/70/80%
         between now (2022) and the target year -- useful for dig-planning.
      7. Returns summary stats, risk distribution, depth-distribution histogram
         bins, the top 20 concerns by predicted depth, and all predictions.
    """
    results = cache.get("results", {})
    pairwise = results.get("pairwise", {})
    chain = results.get("chain", {})
    triple = chain.get("triple_matches", pd.DataFrame())

    # Use 2015-2022 matches as the primary source (most recent growth data)
    key = (2015, 2022)
    if key not in pairwise:
        key = list(pairwise.keys())[-1] if pairwise else None
    if key is None:
        return {"error": "No match data available"}

    matches = pairwise[key].get("matches", pd.DataFrame())
    if matches.empty:
        return {"error": "No matches to extrapolate"}

    base_year = 2022
    years_forward = target_year - base_year
    if years_forward <= 0:
        return {"error": "Target year must be after 2022"}

    predictions = []
    for _, row in matches.iterrows():
        current_depth = row.get("later_depth_pct", np.nan)
        growth_rate = row.get("depth_growth_rate", np.nan)

        if pd.isna(current_depth) or pd.isna(growth_rate):
            continue

        # Use triple-match refined rate if available -- the linear regression
        # over three runs is more reliable than a single pairwise delta.
        refined_rate = growth_rate
        is_triple = False
        if not triple.empty:
            j = row.get("later_joint")
            tm = triple[triple["joint_2022"] == j]
            if not tm.empty:
                lr = tm.iloc[0].get("linear_rate", np.nan)
                if not pd.isna(lr):
                    refined_rate = lr
                    is_triple = True

        # Skip anomalies with negative growth rates (measurement artifacts,
        # not real wall recovery -- wall loss does not reverse in practice)
        if refined_rate < 0:
            continue

        # Extrapolate: predicted_depth = current + rate * years, clamped to [0, 100]
        predicted_depth = current_depth + (refined_rate * years_forward)
        predicted_depth = max(0, min(100, predicted_depth))

        # Estimate how many years until the anomaly reaches the repair threshold
        remaining_capacity = WALL_LOSS_REPAIR_THRESHOLD - predicted_depth
        if refined_rate > 0:
            years_to_threshold = remaining_capacity / refined_rate if remaining_capacity > 0 else 0
        else:
            years_to_threshold = None

        # Classify predicted risk: Critical >=70%, High >=50%, Medium >=30%, Low <30%
        if predicted_depth >= 70:
            risk = "Critical"
        elif predicted_depth >= 50:
            risk = "High"
        elif predicted_depth >= 30:
            risk = "Medium"
        else:
            risk = "Low"

        predictions.append({
            "joint": _safe(row.get("later_joint")),
            "distance_ft": _safe(row.get("later_distance")),
            "clock": _safe(row.get("later_clock")),
            "current_depth_2022": round(current_depth, 1),
            "growth_rate": round(refined_rate, 3),
            "predicted_depth": round(predicted_depth, 1),
            "predicted_risk": risk,
            "years_to_80pct": round(years_to_threshold, 1) if years_to_threshold is not None else None,
            "event_type": row.get("later_event_type", row.get("event_type", "Metal Loss")),
            "confidence": row.get("confidence_label", "Unknown"),
            "is_triple_tracked": is_triple,
        })

    if not predictions:
        return {"error": "No predictions could be generated"}

    pred_df = pd.DataFrame(predictions)

    # Threshold crossings: count anomalies that are currently below each
    # threshold but predicted to exceed it by the target year -- these are
    # the "newly actionable" features that drive dig-plan changes.
    thresholds = {}
    for thresh in [50, 60, 70, 80]:
        currently_below = pred_df["current_depth_2022"] < thresh
        predicted_above = pred_df["predicted_depth"] >= thresh
        newly_crossing = currently_below & predicted_above
        thresholds[f"crossing_{thresh}pct"] = int(newly_crossing.sum())

    # Risk distribution
    risk_dist = pred_df["predicted_risk"].value_counts().to_dict()

    # Top concerns (sorted by predicted depth)
    top_concerns = pred_df.nlargest(20, "predicted_depth").to_dict(orient="records")

    # Summary stats
    summary = {
        "target_year": target_year,
        "years_forward": years_forward,
        "total_predicted": len(pred_df),
        "mean_predicted_depth": round(pred_df["predicted_depth"].mean(), 1),
        "max_predicted_depth": round(pred_df["predicted_depth"].max(), 1),
        "critical_count": int((pred_df["predicted_risk"] == "Critical").sum()),
        "high_count": int((pred_df["predicted_risk"] == "High").sum()),
        "medium_count": int((pred_df["predicted_risk"] == "Medium").sum()),
        "low_count": int((pred_df["predicted_risk"] == "Low").sum()),
        "needing_repair_by_target": int((pred_df["predicted_depth"] >= WALL_LOSS_REPAIR_THRESHOLD).sum()),
    }

    # Depth distribution for histogram
    depth_bins = {
        "0-20%": int((pred_df["predicted_depth"] < 20).sum()),
        "20-40%": int(((pred_df["predicted_depth"] >= 20) & (pred_df["predicted_depth"] < 40)).sum()),
        "40-60%": int(((pred_df["predicted_depth"] >= 40) & (pred_df["predicted_depth"] < 60)).sum()),
        "60-80%": int(((pred_df["predicted_depth"] >= 60) & (pred_df["predicted_depth"] < 80)).sum()),
        "80-100%": int((pred_df["predicted_depth"] >= 80).sum()),
    }

    return {
        "summary": summary,
        "risk_distribution": risk_dist,
        "threshold_crossings": thresholds,
        "depth_distribution": depth_bins,
        "top_concerns": top_concerns,
        "all_predictions": pred_df.to_dict(orient="records"),
    }


def _safe(v):
    """Convert numpy scalar types to native Python types for JSON serialization.

    NumPy integers (np.int64, etc.) and floats (np.float64, etc.) are not
    natively JSON-serializable. This helper converts them to plain int/float
    so the prediction results can be returned as JSON API responses. NaN
    floats are mapped to None.
    """
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if np.isnan(f) else round(f, 2)
    return v
