"""Integrity analytics: segment risk heatmap, ASME B31G interaction,
automated dig list, and population growth analytics.

This module provides four industry-standard integrity analytics that pipeline
operators require for regulatory compliance (49 CFR 192 / 195) and repair
planning after in-line inspection (ILI) runs:

  1. Segment Risk Analysis   -- spatial risk heatmap across the pipeline
  2. Interaction Assessment   -- ASME B31G / RSTRENG defect interaction check
  3. Automated Dig List       -- prioritized repair schedule (the #1 deliverable)
  4. Population Analytics     -- systemic corrosion pattern identification

Each function accepts the pairwise-matched anomaly DataFrame produced by the
alignment engine and returns JSON-serializable results for the frontend.
"""

import numpy as np
import pandas as pd

from config import WALL_LOSS_REPAIR_THRESHOLD, MAX_PLAUSIBLE_GROWTH_RATE


# ── 1. Segment Risk Heatmap ─────────────────────────────────────────────────
#
# Divides the full pipeline (~57,000 ft) into fixed-length segments (default
# 1,000 ft) and computes a composite risk score (0-100) for each segment.
# This gives operators a spatial "heat map" showing where corrosion risk is
# concentrated, so they can plan field work geographically.

def segment_risk_analysis(
    matches_df: pd.DataFrame,
    corrected_runs: dict,
    segment_length_ft: float = 1000.0,
) -> list[dict]:
    """Divide pipeline into segments and compute composite risk score per segment.

    Composite score (0-100) = anomaly_density (25) + max_depth (35) +
                              avg_growth_rate (25) + critical_count (15)

    Score components and their full-credit thresholds:
      - density     (25 pts): 5 anomalies in one segment = full credit.
            Measures clustering of defects in a localized area.
      - max_depth   (35 pts): 80% wall loss (repair threshold) = full credit.
            The single deepest defect drives near-term failure risk.
      - avg_growth  (25 pts): 3 %/yr average growth rate = full credit.
            Fast-growing segments need earlier re-inspection.
      - critical_count (15 pts): 3 critical-category anomalies = full credit.
            Multiple critical defects compound the segment's risk.
    """
    # Determine pipeline extent from the most recent ILI run
    latest_year = max(corrected_runs.keys())
    latest_run = corrected_runs[latest_year]
    max_dist = latest_run["corrected_distance"].max()
    n_segments = int(np.ceil(max_dist / segment_length_ft))

    # Build anomaly list with growth data from the most recent pair.
    # If no pairwise matches exist (e.g., only one run loaded), fall back
    # to raw anomalies from the latest run without growth information.
    if matches_df.empty:
        # Fall back to raw anomalies without growth data
        anom = latest_run[latest_run["is_anomaly"]].copy()
        anom = anom.rename(columns={
            "corrected_distance": "distance",
            "depth_pct": "depth",
        })
        anom["growth_rate"] = np.nan
        anom["risk_category"] = "Unknown"
    else:
        anom = matches_df[["later_distance", "later_depth_pct",
                           "depth_growth_rate", "risk_category"]].copy()
        anom.columns = ["distance", "depth", "growth_rate", "risk_category"]

    segments = []
    for i in range(n_segments):
        start = i * segment_length_ft
        end = start + segment_length_ft
        mid = start + segment_length_ft / 2

        seg_anom = anom[(anom["distance"] >= start) & (anom["distance"] < end)]
        count = len(seg_anom)

        # Empty segments get a zero risk score
        if count == 0:
            segments.append({
                "segment": i + 1,
                "start_ft": round(start, 1),
                "end_ft": round(end, 1),
                "midpoint_ft": round(mid, 1),
                "anomaly_count": 0,
                "max_depth_pct": 0,
                "avg_growth_rate": 0,
                "critical_count": 0,
                "risk_score": 0,
            })
            continue

        max_depth = seg_anom["depth"].max() if not seg_anom["depth"].isna().all() else 0
        avg_rate = seg_anom["growth_rate"].dropna()
        avg_rate = avg_rate[avg_rate >= 0].mean() if len(avg_rate[avg_rate >= 0]) > 0 else 0
        critical_count = (seg_anom["risk_category"] == "Critical").sum()

        # Density score (0-25): linear scale, 5 anomalies per segment = full 25 pts
        density_score = min(25, count * 25 / 5)

        # Max depth score (0-35): linear scale, 80% wall loss = full 35 pts
        depth_score = min(35, (max_depth / WALL_LOSS_REPAIR_THRESHOLD) * 35) if not np.isnan(max_depth) else 0

        # Growth rate score (0-25): linear scale, 3 %/yr avg = full 25 pts
        rate_score = min(25, (avg_rate / 3.0) * 25) if not np.isnan(avg_rate) else 0

        # Critical count score (0-15): linear scale, 3 criticals = full 15 pts
        crit_score = min(15, critical_count * 15 / 3)

        risk_score = round(density_score + depth_score + rate_score + crit_score, 1)

        segments.append({
            "segment": i + 1,
            "start_ft": round(start, 1),
            "end_ft": round(end, 1),
            "midpoint_ft": round(mid, 1),
            "anomaly_count": int(count),
            "max_depth_pct": round(float(max_depth), 1) if not np.isnan(max_depth) else 0,
            "avg_growth_rate": round(float(avg_rate), 3) if not np.isnan(avg_rate) else 0,
            "critical_count": int(critical_count),
            "risk_score": risk_score,
        })

    return segments



# ── 2. ASME B31G Anomaly Interaction Assessment ─────────────────────────────
#
# Per ASME B31G and RSTRENG (Remaining Strength of Corroded Pipe), corrosion
# anomalies that are close together axially can "interact" -- meaning the pipe
# sees them as one larger effective defect rather than two independent ones.
# This reduces the calculated burst pressure below what either defect alone
# would produce.  The standard interaction criterion is: two anomalies interact
# when their axial spacing is less than 6 x wall_thickness.
#
# This function implements a forward-chaining algorithm: anomalies are sorted
# by distance, then for each unvisited anomaly we walk forward, adding any
# neighbor within the 6xWT threshold to the current cluster.  The chain
# continues from the last added anomaly so that three or more defects can
# form a single interaction group.

def interaction_assessment(matches_df: pd.DataFrame) -> list[dict]:
    """Detect anomalies that may interact per ASME B31G / RSTRENG rules.

    Per ASME B31G, two anomalies interact if their axial spacing is less than
    6 x wall_thickness. When anomalies interact, their combined effective
    length is greater than either alone, reducing the pipe's burst pressure.

    Severity classification for each interaction cluster:
      HIGH   -- max depth >= 60% wall loss  OR  >= 4 anomalies in the cluster
      MEDIUM -- max depth >= 40% wall loss  OR  >= 3 anomalies in the cluster
      LOW    -- everything else (still flagged because interaction exists)
    """
    if matches_df.empty:
        return []

    # Work with the latest-run anomaly data
    required = ["later_distance", "later_depth_pct", "later_wall_thickness",
                 "later_joint", "later_clock", "later_length_in",
                 "depth_growth_rate", "risk_score"]
    available = [c for c in required if c in matches_df.columns]
    df = matches_df[available].dropna(subset=["later_distance", "later_depth_pct"]).copy()
    # Sort by distance so the forward-chaining walk processes anomalies in order
    df = df.sort_values("later_distance").reset_index(drop=True)

    if len(df) < 2:
        return []

    interactions = []
    used = set()  # Track anomalies already assigned to a cluster

    for i in range(len(df)):
        if i in used:
            continue
        row_i = df.iloc[i]
        wt = row_i.get("later_wall_thickness", 0.3)
        if pd.isna(wt) or wt <= 0:
            wt = 0.3  # Default wall thickness

        # Interaction threshold: 6 x wall thickness (in inches), convert to feet
        # because distances in the dataframe are in feet
        threshold_ft = (6 * wt) / 12.0

        # Start a new cluster with the current anomaly, then chain forward:
        # each successive anomaly within threshold of the *last added* member
        # extends the cluster.
        cluster = [i]
        # Find all anomalies within the threshold distance (chain forward)
        j = i + 1
        last_dist = row_i["later_distance"]
        while j < len(df):
            next_dist = df.iloc[j]["later_distance"]
            spacing = next_dist - last_dist

            # Subtract the anomaly's own length so we measure clear spacing
            # (edge-to-edge) rather than center-to-center
            length_i = row_i.get("later_length_in", 0)
            if pd.isna(length_i):
                length_i = 0
            clear_spacing = spacing - (length_i / 12.0)

            if clear_spacing <= threshold_ft:
                cluster.append(j)
                used.add(j)
                last_dist = next_dist
                j += 1
            else:
                break

        # Only report clusters with 2+ anomalies (single anomalies cannot interact)
        if len(cluster) >= 2:
            used.add(i)
            members = df.iloc[cluster]

            # Combined effective length: full span plus the longest individual anomaly
            total_span_ft = members["later_distance"].max() - members["later_distance"].min()
            member_lengths = members.get("later_length_in", pd.Series([0] * len(members)))
            total_length_in = total_span_ft * 12 + member_lengths.fillna(0).max()

            max_depth = members["later_depth_pct"].max()
            avg_depth = members["later_depth_pct"].mean()
            max_growth = members.get("depth_growth_rate", pd.Series()).max()
            max_risk = members.get("risk_score", pd.Series()).max()

            # Classify interaction severity based on combined depth and count
            if max_depth >= 60 or len(cluster) >= 4:
                severity = "HIGH"
            elif max_depth >= 40 or len(cluster) >= 3:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            interactions.append({
                "cluster_id": len(interactions) + 1,
                "anomaly_count": len(cluster),
                "start_distance_ft": round(float(members["later_distance"].min()), 2),
                "end_distance_ft": round(float(members["later_distance"].max()), 2),
                "span_ft": round(float(total_span_ft), 2),
                "effective_length_in": round(float(total_length_in), 1),
                "max_depth_pct": round(float(max_depth), 1),
                "avg_depth_pct": round(float(avg_depth), 1),
                "max_growth_rate": round(float(max_growth), 3) if not pd.isna(max_growth) else None,
                "max_risk_score": round(float(max_risk), 1) if not pd.isna(max_risk) else None,
                "joint": int(members["later_joint"].iloc[0]) if "later_joint" in members.columns and not pd.isna(members["later_joint"].iloc[0]) else None,
                "wall_thickness_in": round(float(wt), 3),
                "interaction_threshold_in": round(float(6 * wt), 2),
                "severity": severity,
            })

    return interactions



# ── 3. Automated Dig List / Repair Prioritization ───────────────────────────
#
# The dig list is the #1 deliverable that pipeline operators need from an ILI
# analysis.  It tells field crews exactly where to excavate, what they will
# find, and how urgently each location needs repair.  Regulatory bodies
# (PHMSA) require operators to remediate critical defects within strict
# timelines, so correct prioritization is essential.
#
# Each anomaly receives an urgency score (0-100) built from three components:
#   - Current depth   (40% weight): deeper defects are closer to failure.
#         80% wall loss = full 40 pts  (matches the repair threshold).
#   - Growth rate      (30% weight): fast-growing defects need earlier action.
#         5 %/yr = full 30 pts  (MAX_PLAUSIBLE_GROWTH_RATE from config).
#   - Remaining life   (30% weight): years until the defect reaches 80%.
#         0 years = full 30 pts, 15+ years = 0 pts  (linear interpolation).
#
# Anomalies are then categorized:
#   IMMEDIATE -- urgency >= 75  OR  depth >= 70%  OR  remaining life < 3 yr
#       Must be repaired within days/weeks per regulatory requirements.
#   SCHEDULED -- urgency >= 50  OR  depth >= 50%  OR  remaining life < 7 yr
#       Repair within the current operating year.
#   MONITOR   -- everything else that still shows growth or depth concern.
#       Track at next inspection; no excavation needed now.

def generate_dig_list(matches_df: pd.DataFrame) -> list[dict]:
    """Generate prioritized repair schedule with IMMEDIATE / SCHEDULED / MONITOR.

    Urgency score = current_depth_component (40) + growth_rate_component (30) +
                    remaining_life_component (30)

    Categories:
      IMMEDIATE: urgency >= 75 or depth >= 70% or remaining_life < 3 years
      SCHEDULED: urgency >= 50 or depth >= 50% or remaining_life < 7 years
      MONITOR:   everything else with growth or depth concern
    """
    if matches_df.empty:
        return []

    required = ["later_distance", "later_depth_pct", "later_joint", "later_clock",
                 "depth_growth_rate", "remaining_life_years", "risk_score",
                 "risk_category", "later_event_type", "later_id_od",
                 "later_wall_thickness", "confidence_label"]
    available = [c for c in required if c in matches_df.columns]
    df = matches_df[available].copy()
    df = df.dropna(subset=["later_depth_pct"])

    dig_items = []
    for _, row in df.iterrows():
        depth = row.get("later_depth_pct", 0)
        rate = row.get("depth_growth_rate", 0)
        rem_life = row.get("remaining_life_years", np.nan)

        if pd.isna(depth):
            depth = 0
        if pd.isna(rate) or rate < 0:
            rate = 0
        if pd.isna(rem_life):
            rem_life = 999

        # Skip low-concern anomalies: shallow (<20%) with negligible growth
        # (<= 0.5 %/yr) -- these do not warrant a dig site visit.
        if depth < 20 and rate <= 0.5:
            continue

        # Depth component (0-40): linear, 80% wall loss = full 40 pts
        depth_score = min(40, (depth / WALL_LOSS_REPAIR_THRESHOLD) * 40)

        # Growth rate component (0-30): linear, 5 %/yr = full 30 pts
        rate_score = min(30, (rate / MAX_PLAUSIBLE_GROWTH_RATE) * 30)

        # Remaining life component (0-30): inverse linear,
        # 0 yr remaining = full 30 pts, 15+ yr remaining = 0 pts
        if rem_life <= 0:
            life_score = 30
        elif rem_life >= 15:
            life_score = 0
        else:
            life_score = 30 * (1 - rem_life / 15)

        urgency = round(depth_score + rate_score + life_score, 1)

        # Categorize by urgency score and hard thresholds on depth / life
        if urgency >= 75 or depth >= 70 or rem_life < 3:
            category = "IMMEDIATE"
            priority = 1
        elif urgency >= 50 or depth >= 50 or rem_life < 7:
            category = "SCHEDULED"
            priority = 2
        else:
            category = "MONITOR"
            priority = 3

        dig_items.append({
            "joint": int(row.get("later_joint", 0)) if not pd.isna(row.get("later_joint")) else None,
            "distance_ft": round(float(row.get("later_distance", 0)), 2) if not pd.isna(row.get("later_distance")) else None,
            "clock": round(float(row.get("later_clock", 0)), 1) if not pd.isna(row.get("later_clock", np.nan)) else None,
            "depth_pct": round(float(depth), 1),
            "growth_rate": round(float(rate), 3),
            "remaining_life_years": round(float(rem_life), 1) if rem_life < 999 else None,
            "event_type": row.get("later_event_type", ""),
            "id_od": row.get("later_id_od", ""),
            "wall_thickness_in": round(float(row.get("later_wall_thickness", 0)), 3) if not pd.isna(row.get("later_wall_thickness")) else None,
            "urgency_score": urgency,
            "category": category,
            "priority": priority,
            "risk_category": row.get("risk_category", ""),
            "confidence": row.get("confidence_label", ""),
        })

    # Sort by priority (1=IMMEDIATE first) then by urgency descending within
    # each priority tier, so the most critical digs appear at the top.
    dig_items.sort(key=lambda x: (x["priority"], -x["urgency_score"]))
    return dig_items



# ── 4. Population Growth Analytics ──────────────────────────────────────────
#
# Groups anomaly growth rates by clock-position quadrant, ID/OD classification,
# and depth band to reveal systemic corrosion patterns across the pipeline.
#
# Clock quadrants divide the pipe cross-section:
#   Top    (10-2 o'clock) -- crown of the pipe
#   Right  ( 2-4 o'clock) -- springline (right side)
#   Bottom ( 4-8 o'clock) -- invert / 6 o'clock
#   Left   ( 8-10 o'clock) -- springline (left side)
#
# Domain interpretation of quadrant + ID/OD combinations:
#   Bottom-of-pipe + Internal (ID) = water or liquid settling to the low point,
#       causing accelerated internal corrosion in liquid pipelines.
#   Bottom-of-pipe + External (OD) = soil-side corrosion from poor drainage,
#       coating disbondment, or cathodic protection shielding.
#   Top-of-pipe + Internal (ID) = gas-phase or vapor-phase corrosion
#       (e.g., CO2, H2S) attacking the crown in multiphase flow.
#   Top-of-pipe + External (OD) = coating degradation, atmospheric corrosion
#       at supports, or CP current attenuation.
#
# When growth rates cluster in a specific quadrant + ID/OD combination, it
# indicates a systemic mechanism rather than random pitting, which changes the
# remediation strategy from spot repairs to system-wide mitigation.

def population_analytics(matches_df: pd.DataFrame) -> dict:
    """Analyze growth patterns by clock quadrant, ID/OD, and depth band.

    Clock quadrants:
      Top (10-2 o'clock), Right (2-4), Bottom (4-8), Left (8-10)

    Reveals systemic corrosion patterns:
      - Bottom-of-pipe = water settling (internal), soil-side (external)
      - Top-of-pipe = gas phase corrosion (internal), coating failure (external)
    """
    if matches_df.empty:
        return {"by_quadrant": [], "by_id_od": [], "by_depth_band": [],
                "quadrant_id_od": []}

    df = matches_df.copy()
    rates = df["depth_growth_rate"].copy()

    # Only use valid positive growth rates; negative rates (depth decrease)
    # are physically implausible and indicate measurement noise.
    valid_mask = rates.notna() & (rates >= 0)
    df_valid = df[valid_mask].copy()

    if df_valid.empty:
        return {"by_quadrant": [], "by_id_od": [], "by_depth_band": [],
                "quadrant_id_od": []}

    # Map clock position (0-12 o'clock) to pipe cross-section quadrant
    def _clock_quadrant(clock):
        if pd.isna(clock):
            return "Unknown"
        clock = clock % 12
        if clock >= 10 or clock < 2:
            return "Top (10-2)"
        elif clock >= 2 and clock < 4:
            return "Right (2-4)"
        elif clock >= 4 and clock < 8:
            return "Bottom (4-8)"
        else:
            return "Left (8-10)"

    df_valid["quadrant"] = df_valid["later_clock"].apply(_clock_quadrant)

    # Bin current depth into severity bands for population-level comparison
    def _depth_band(depth):
        if pd.isna(depth):
            return "Unknown"
        if depth < 20:
            return "0-20%"
        elif depth < 40:
            return "20-40%"
        elif depth < 60:
            return "40-60%"
        else:
            return "60%+"

    df_valid["depth_band"] = df_valid["later_depth_pct"].apply(_depth_band)

    # ── By Quadrant ── growth rate statistics per pipe cross-section quadrant
    by_quadrant = []
    for quad, group in df_valid.groupby("quadrant"):
        rates_g = group["depth_growth_rate"]
        by_quadrant.append({
            "quadrant": quad,
            "count": int(len(group)),
            "mean_growth_rate": round(float(rates_g.mean()), 3),
            "median_growth_rate": round(float(rates_g.median()), 3),
            "max_growth_rate": round(float(rates_g.max()), 3),
            "pct_high_growth": round(float((rates_g > 3.0).mean() * 100), 1),
            "avg_depth": round(float(group["later_depth_pct"].mean()), 1),
        })

    # ── By ID/OD ── Internal vs External corrosion growth comparison
    by_id_od = []
    id_od_col = "later_id_od" if "later_id_od" in df_valid.columns else None
    if id_od_col:
        for label, group in df_valid.groupby(id_od_col):
            if pd.isna(label) or label == "":
                label = "Unknown"
            rates_g = group["depth_growth_rate"]
            by_id_od.append({
                "type": str(label),
                "count": int(len(group)),
                "mean_growth_rate": round(float(rates_g.mean()), 3),
                "median_growth_rate": round(float(rates_g.median()), 3),
                "max_growth_rate": round(float(rates_g.max()), 3),
                "avg_depth": round(float(group["later_depth_pct"].mean()), 1),
            })

    # ── By Depth Band ── do deeper defects grow faster? (acceleration check)
    by_depth_band = []
    for band, group in df_valid.groupby("depth_band"):
        rates_g = group["depth_growth_rate"]
        by_depth_band.append({
            "band": band,
            "count": int(len(group)),
            "mean_growth_rate": round(float(rates_g.mean()), 3),
            "median_growth_rate": round(float(rates_g.median()), 3),
        })

    # ── Cross-tab: Quadrant x ID/OD ── the most diagnostic view: identifies
    # specific corrosion mechanisms (e.g., bottom-of-pipe internal = water settling)
    quadrant_id_od = []
    if id_od_col:
        for (quad, idod), group in df_valid.groupby(["quadrant", id_od_col]):
            if pd.isna(idod) or idod == "":
                idod = "Unknown"
            rates_g = group["depth_growth_rate"]
            quadrant_id_od.append({
                "quadrant": quad,
                "id_od": str(idod),
                "count": int(len(group)),
                "mean_growth_rate": round(float(rates_g.mean()), 3),
                "avg_depth": round(float(group["later_depth_pct"].mean()), 1),
            })

    return {
        "by_quadrant": by_quadrant,
        "by_id_od": by_id_od,
        "by_depth_band": by_depth_band,
        "quadrant_id_od": quadrant_id_od,
    }



# ── Combined Dashboard Endpoint ─────────────────────────────────────────────
#
# This is the single entry point called by the API layer.  It retrieves the
# best available pairwise match data from the analysis cache (preferring the
# most recent pair: 2015-2022, then 2007-2022, then 2007-2015), and runs all
# four analytics functions above.  The result is a consolidated JSON payload
# with a summary section plus the full output of each analytic.

def compute_integrity_dashboard(cache: dict) -> dict:
    """Compute all integrity analytics from the cached analysis results.

    Selects the best pairwise match dataset (preferring 2015-2022 for the
    shortest inter-run interval and most recent data), then delegates to:
      - segment_risk_analysis()  -> spatial risk heatmap
      - interaction_assessment() -> ASME B31G defect clusters
      - generate_dig_list()      -> prioritized repair schedule
      - population_analytics()   -> systemic corrosion patterns
    """
    results = cache.get("results", {})
    corrected_runs = cache.get("corrected_runs", {})

    # Select the best pairwise match set.  Preference order:
    # (2015, 2022) -- most recent pair with shortest interval (7 yr)
    # (2007, 2022) -- longest interval (15 yr), useful if 2015 missing
    # (2007, 2015) -- oldest pair, fallback only
    pairwise = results.get("pairwise", {})
    best_key = None
    for key in [(2015, 2022), (2007, 2022), (2007, 2015)]:
        if key in pairwise:
            best_key = key
            break

    matches = pd.DataFrame()
    if best_key and "matches" in pairwise[best_key]:
        matches = pairwise[best_key]["matches"]

    # Run all four analytics
    segments = segment_risk_analysis(matches, corrected_runs)
    interactions = interaction_assessment(matches)
    dig_list = generate_dig_list(matches)
    population = population_analytics(matches)

    # Aggregate summary counts for the dashboard header
    immediate_count = sum(1 for d in dig_list if d["category"] == "IMMEDIATE")
    scheduled_count = sum(1 for d in dig_list if d["category"] == "SCHEDULED")
    monitor_count = sum(1 for d in dig_list if d["category"] == "MONITOR")

    high_risk_segments = sum(1 for s in segments if s["risk_score"] >= 60)

    return {
        "summary": {
            "total_dig_items": len(dig_list),
            "immediate_count": immediate_count,
            "scheduled_count": scheduled_count,
            "monitor_count": monitor_count,
            "interaction_clusters": len(interactions),
            "high_risk_segments": high_risk_segments,
            "total_segments": len(segments),
            "match_pair": f"{best_key[0]}-{best_key[1]}" if best_key else "N/A",
        },
        "segments": segments,
        "interactions": interactions,
        "dig_list": dig_list,
        "population": population,
    }
