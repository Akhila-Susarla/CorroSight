"""Multi-run analysis: chain matching across 3 ILI runs and trend analysis.

Chains pairwise matches (e.g. 2007->2015, 2015->2022) to track individual
anomalies across all 3 inspections, enabling lifecycle categorization and
depth-growth trend analysis over the full inspection history.
"""

import numpy as np
import pandas as pd

from config import YEARS_BETWEEN
from matching import match_anomalies
from growth import predict_growth_trends


def run_full_analysis(
    corrected_runs: dict[int, pd.DataFrame],
) -> dict:
    """Run complete multi-run matching and growth analysis.

    Workflow:
      1. Pairwise matching for consecutive runs (e.g. 2007-2015, 2015-2022)
      2. Direct first-to-last matching (e.g. 2007-2022) for validation
      3. Chain pairwise results into 3-run tracking (triple matches)
      4. Predict growth trends for anomalies tracked across all 3 runs

    Returns dict with all pairwise match results and chained 3-run tracking.
    """
    years = sorted(corrected_runs.keys())
    results = {}

    # --- Step 1: Pairwise matching for each consecutive run pair ---
    pairwise = {}
    for i in range(len(years) - 1):
        y_early = years[i]
        y_later = years[i + 1]
        yb = YEARS_BETWEEN.get((y_early, y_later), y_later - y_early)
        match_result = match_anomalies(
            corrected_runs[y_later],
            corrected_runs[y_early],
            yb,
        )
        pairwise[(y_early, y_later)] = match_result

    results["pairwise"] = pairwise

    # --- Step 2: Direct first-to-last matching (skip middle run) for validation ---
    if len(years) >= 3:
        y_first, y_last = years[0], years[-1]
        yb_direct = YEARS_BETWEEN.get((y_first, y_last), y_last - y_first)
        direct_result = match_anomalies(
            corrected_runs[y_last],
            corrected_runs[y_first],
            yb_direct,
        )
        results["direct_first_last"] = direct_result

    # --- Step 3: Chain pairwise matches into 3-run tracking ---
    if len(years) >= 3:
        chain = chain_three_runs(pairwise, years)
        results["chain"] = chain

        # Step 4: Predict depth-growth trends for anomalies found in all 3 runs
        if not chain["triple_matches"].empty:
            chain["triple_matches"] = predict_growth_trends(chain["triple_matches"])

    return results


def chain_three_runs(
    pairwise: dict,
    years: list[int],
) -> dict:
    """Chain pairwise matches to create 3-run tracking.

    Join logic uses earlier_row_idx as the key:
      - From pair (2015, 2022): anomaly A in 2022 matched to B in 2015
      - From pair (2007, 2015): anomaly B in 2015 matched to C in 2007
      - B's later_row_idx in matches_12 == B's earlier_row_idx in matches_23
      - When both exist, C -> B -> A forms a triple match.
    """
    y1, y2, y3 = years[0], years[1], years[2]

    matches_12 = pairwise.get((y1, y2), {}).get("matches", pd.DataFrame())
    matches_23 = pairwise.get((y2, y3), {}).get("matches", pd.DataFrame())

    if matches_12.empty or matches_23.empty:
        return {
            "triple_matches": pd.DataFrame(),
            "lifecycle_summary": pd.DataFrame(),
        }

    # Build lookup: map each y2 anomaly (by row_idx) to its y1 match from pair 1-2.
    # matches_23: later = y3 run, earlier = y2 run
    # matches_12: later = y2 run, earlier = y1 run

    # Key: y2 row_idx -> y1 match info (the row from matches_12)
    y2_to_y1 = {}
    for _, m12 in matches_12.iterrows():
        y2_row = m12["later_row_idx"]  # This is the y2 anomaly
        y2_to_y1[y2_row] = m12

    # Walk through pair 2-3 matches; when the y2 side also exists in the
    # y2_to_y1 lookup, we have a triple match spanning all 3 runs.
    triple_records = []
    for _, m23 in matches_23.iterrows():
        y2_row = m23["earlier_row_idx"]  # The y2 anomaly in this match

        if y2_row in y2_to_y1:
            m12 = y2_to_y1[y2_row]
            # Triple match found: y1 → y2 → y3
            record = {
                # Tracking
                "lifecycle": "Tracked All 3 Runs",
                # Y1 (earliest) data
                f"joint_{y1}": m12["earlier_joint"],
                f"distance_{y1}": m12["earlier_distance"],
                f"clock_{y1}": m12["earlier_clock"],
                f"depth_{y1}": m12["earlier_depth_pct"],
                f"length_{y1}": m12["earlier_length_in"],
                f"width_{y1}": m12["earlier_width_in"],
                f"row_idx_{y1}": m12["earlier_row_idx"],
                # Y2 (middle) data
                f"joint_{y2}": m23["earlier_joint"],
                f"distance_{y2}": m23["earlier_distance"],
                f"clock_{y2}": m23["earlier_clock"],
                f"depth_{y2}": m23["earlier_depth_pct"],
                f"length_{y2}": m23["earlier_length_in"],
                f"width_{y2}": m23["earlier_width_in"],
                f"row_idx_{y2}": m23["earlier_row_idx"],
                # Y3 (latest) data
                f"joint_{y3}": m23["later_joint"],
                f"distance_{y3}": m23["later_distance"],
                f"clock_{y3}": m23["later_clock"],
                f"depth_{y3}": m23["later_depth_pct"],
                f"length_{y3}": m23["later_length_in"],
                f"width_{y3}": m23["later_width_in"],
                f"row_idx_{y3}": m23["later_row_idx"],
                # Confidence (min of both matches)
                "confidence_12": m12["confidence"],
                "confidence_23": m23["confidence"],
                "min_confidence": min(m12["confidence"], m23["confidence"]),
                # Overall growth
                "total_depth_growth": _safe_sub(m23["later_depth_pct"], m12["earlier_depth_pct"]),
                "total_years": YEARS_BETWEEN.get((y1, y3), y3 - y1),
            }
            triple_records.append(record)

    triple_df = pd.DataFrame(triple_records)
    if not triple_df.empty:
        # Annualized depth growth rate across the full span (y1 to y3)
        triple_df["overall_growth_rate"] = (
            triple_df["total_depth_growth"] / triple_df["total_years"]
        ).round(3)

    # Lifecycle summary
    lifecycle = _build_lifecycle_summary(pairwise, years, triple_df)

    return {
        "triple_matches": triple_df,
        "lifecycle_summary": lifecycle,
    }


def _build_lifecycle_summary(
    pairwise: dict,
    years: list[int],
    triple_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize anomaly lifecycle across runs into 5 categories.

    Categories:
      - Tracked All 3 Runs: anomaly matched in both pairwise comparisons
      - New in y2 (tracked to y3): first appeared in y2, still present in y3
      - New in y3: first appeared in the latest run only
      - Disappeared after y1: present in y1 but not matched in y2
      - Disappeared after y2: present in y2 but not matched in y3
    """
    y1, y2, y3 = years[0], years[1], years[2]

    m12 = pairwise.get((y1, y2), {})
    m23 = pairwise.get((y2, y3), {})

    records = [
        {"Category": "Tracked All 3 Runs", "Count": len(triple_df)},
        # Matched in pair 2-3 but not chained to y1 -> first seen in y2
        {"Category": f"New in {y2} (tracked to {y3})",
         "Count": len(m23.get("matches", pd.DataFrame())) - len(triple_df)},
        # Present only in the latest run (unmatched in pair 2-3)
        {"Category": f"New in {y3}",
         "Count": len(m23.get("new_anomalies", pd.DataFrame()))},
        # Present in y1 but not matched to anything in y2
        {"Category": f"Disappeared after {y1}",
         "Count": len(m12.get("missing_anomalies", pd.DataFrame()))
                  if isinstance(m12, dict) else 0},
        # Present in y2 but not matched to anything in y3
        {"Category": f"Disappeared after {y2}",
         "Count": len(m23.get("missing_anomalies", pd.DataFrame()))},
    ]
    return pd.DataFrame(records)


def _safe_sub(a, b) -> float:
    """Subtract b from a, returning NaN if either value is missing."""
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a) - float(b)


def export_results(results: dict, output_path: str):
    """Export all results to a multi-sheet Excel workbook.

    Sheets written:
      - Matches_{early}_{later}  : pairwise matched anomalies per run pair
      - New_{later}              : anomalies with no match in the earlier run
      - Missing_{early}          : anomalies from earlier run not found later
      - Multi_Run_Tracking       : triple-matched anomalies across all 3 runs
      - Lifecycle_Summary        : counts per lifecycle category
      - Summary_Stats            : aggregate match statistics per run pair
    """
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        # --- Pairwise match sheets (one set per consecutive run pair) ---
        for (y_early, y_later), match_result in results.get("pairwise", {}).items():
            matches = match_result.get("matches", pd.DataFrame())
            if not matches.empty:
                matches.to_excel(writer, sheet_name=f"Matches_{y_early}_{y_later}", index=False)

            new = match_result.get("new_anomalies", pd.DataFrame())
            if not new.empty:
                cols = [c for c in ["event_type", "corrected_distance", "clock_hours",
                                     "depth_pct", "length_in", "width_in", "joint_number",
                                     "id_od", "comments"] if c in new.columns]
                new[cols].to_excel(writer, sheet_name=f"New_{y_later}", index=False)

            missing = match_result.get("missing_anomalies", pd.DataFrame())
            if not missing.empty:
                cols = [c for c in ["event_type", "corrected_distance", "clock_hours",
                                     "depth_pct", "length_in", "width_in", "joint_number",
                                     "id_od", "comments"] if c in missing.columns]
                missing[cols].to_excel(writer, sheet_name=f"Missing_{y_early}", index=False)

        # --- 3-run chained results ---
        chain = results.get("chain", {})
        triple = chain.get("triple_matches", pd.DataFrame())
        if not triple.empty:
            triple.to_excel(writer, sheet_name="Multi_Run_Tracking", index=False)

        lifecycle = chain.get("lifecycle_summary", pd.DataFrame())
        if not lifecycle.empty:
            lifecycle.to_excel(writer, sheet_name="Lifecycle_Summary", index=False)

        # --- Aggregate stats for each pairwise comparison ---
        stats_records = []
        for (y_early, y_later), match_result in results.get("pairwise", {}).items():
            s = match_result.get("stats", {})
            s["pair"] = f"{y_early}-{y_later}"
            stats_records.append(s)
        if stats_records:
            pd.DataFrame(stats_records).to_excel(writer, sheet_name="Summary_Stats", index=False)
