"""Configuration for the CorroSight pipeline integrity / ILI alignment system.

This module centralises every tunable constant used by the backend:
column-name mappings for each ILI vendor spreadsheet, event-type
normalisation rules, anomaly-matching tolerances, similarity weights,
confidence thresholds, and corrosion-growth limits.  It is imported by
the ingestion, alignment, matching, and growth-analysis layers so that
a single edit here propagates to the entire pipeline.
"""

# ── Run metadata ──────────────────────────────────────────────────────────────
# Three inline-inspection (ILI) runs performed on this pipeline segment.
RUN_YEARS = [2007, 2015, 2022]
# Pre-computed year gaps between every pair of runs (used for growth-rate
# calculations: depth-change / years-between = annualised growth rate).
YEARS_BETWEEN = {(2007, 2015): 8, (2015, 2022): 7, (2007, 2022): 15}

# ── Column name mappings (raw → canonical) ────────────────────────────────────
# Each ILI vendor delivers data with different header names and column counts.
#   2007 – Rosen        (15 columns, basic feature-level report)
#   2015 – Baker Hughes (34 columns, adds burst-pressure calcs & engineering)
#   2022 – Baker Hughes (43 columns, adds dent depth, seam info, pipe type)
# COLUMN_MAP translates every vendor-specific header to a unified canonical
# name so downstream code never has to know which vendor produced the data.
COLUMN_MAP = {
    # ── 2007 / Rosen (15 raw columns) ────────────────────────────────────
    2007: {
        "J. no.": "joint_number",
        "J. len [ft]": "joint_length_ft",
        "t [in]": "wall_thickness_in",
        "to u/s w. [ft]": "dist_to_us_weld_ft",
        "log dist. [ft]": "log_distance_ft",
        "Height [ft]": "elevation_ft",
        "event": "event_type",
        "depth [%]": "depth_pct",
        "ID Reduction [%]": "id_reduction_pct",
        "length [in]": "length_in",
        "width [in]": "width_in",
        "P2 Burst / MOP": "burst_mop_ratio",
        "o'clock": "clock_position",
        "internal": "id_od",              # YES/NO flag; see ID_OD_MAP_2007
        "comment": "comments",
    },
    # ── 2015 / Baker Hughes (34 raw columns) ─────────────────────────────
    2015: {
        "J. no.": "joint_number",
        "J. len [ft]": "joint_length_ft",
        "Wt [in]": "wall_thickness_in",
        "to u/s w. [ft]": "dist_to_us_weld_ft",
        "to d/s w. [ft]": "dist_to_ds_weld_ft",
        "Log Dist. [ft]": "log_distance_ft",
        "Event Description": "event_type",
        "ID/OD": "id_od",                 # already "ID" or "OD"
        "Depth [%]": "depth_pct",
        "Depth [in]": "depth_in",
        "OD Reduction [%]": "od_reduction_pct",
        "OD Reduction [in]": "od_reduction_in",
        "Length [in]": "length_in",
        "Width [in]": "width_in",
        "O'clock": "clock_position",
        "Comments": "comments",
        "Anomalies per Joint": "anomalies_per_joint",
        "Tool Velocity [ft/s]": "tool_velocity",
        "Elevation [ft]": "elevation_ft",
        "MOP [PSI]": "mop_psi",
        "SMYS [PSI]": "smys_psi",
        "Pdesign [PSI]": "pdesign_psi",
        "B31G Psafe [PSI]": "b31g_psafe_psi",
        "B31G Pburst [PSI]": "b31g_pburst_psi",
        "Mod B31G Psafe [PSI]": "mod_b31g_psafe_psi",
        "Mod B31G Pburst [PSI]": "mod_b31g_pburst_psi",
        "Effective Area Psafe [PSI]": "eff_area_psafe_psi",
        "Effective Area Pburst [PSI]": "eff_area_pburst_psi",
        "ERF": "erf",
        "RPR": "rpr",
    },
    # ── 2022 / Baker Hughes (43 raw columns) ─────────────────────────────
    2022: {
        "Joint Number": "joint_number",
        "Joint Length [ft]": "joint_length_ft",
        "WT [in]": "wall_thickness_in",
        "Distance to U/S GW \n[ft]": "dist_to_us_weld_ft",
        "Distance to D/S GW \n[ft]": "dist_to_ds_weld_ft",
        "ILI Wheel Count \n[ft.]": "log_distance_ft",
        "Event Description": "event_type",
        "ID/OD": "id_od",                 # already "ID" or "OD"
        "Metal Loss Depth \n[%]": "depth_pct",
        "Metal Loss Depth \n[in]": "depth_in",
        "Metal Loss Depth + Tolerance\n[%]": "depth_plus_tol_pct",
        "Dimension Classification": "dimension_class",
        "Dent Depth\n [%]": "dent_depth_pct",
        "Dent Depth\n [in]": "dent_depth_in",
        "Length [in]": "length_in",
        "Width [in]": "width_in",
        "O'clock\n[hh:mm]": "clock_position",
        "Comments": "comments",
        "Anomalies per Joint": "anomalies_per_joint",
        "Elevation [ft]": "elevation_ft",
        "Seam Position\n[hh:mm]": "seam_position",
        "Distance To Seam Weld \n[in]": "dist_to_seam_in",
        "Tool": "tool",
        "Evaluation Pressure [PSI]": "eval_pressure_psi",
        "SMYS [PSI]": "smys_psi",
        "Pipe Type": "pipe_type",
        "Pipe Diameter (O.D.) \n[in.]": "pipe_od_in",
        "Pdesign [PSI]": "pdesign_psi",
        "Mod B31G Psafe \n[PSI]": "mod_b31g_psafe_psi",
        "Mod B31G Pburst [PSI]": "mod_b31g_pburst_psi",
        "Effective Area Psafe [PSI]": "eff_area_psafe_psi",
        "Effective Area Pburst [PSI]": "eff_area_pburst_psi",
        "ERF": "erf",
        "RPR": "rpr",
    },
}

# ── Event type normalization ──────────────────────────────────────────────────
# The three vendor reports contain 59+ distinct raw event-type strings.
# EVENT_TYPE_MAP collapses them into 28 canonical types so that the matcher
# and growth analyser can compare features across runs without worrying
# about spelling, capitalisation, or vendor-specific naming conventions.
EVENT_TYPE_MAP = {
    # Metal loss — the primary corrosion defect type
    "metal loss": "Metal Loss",
    "Metal Loss": "Metal Loss",
    # Cluster — group of closely spaced pits; 2007 Rosen reported clusters
    # that later runs decomposed into individual metal-loss features
    "Cluster": "Cluster",
    "cluster": "Cluster",
    # Manufacturing anomalies — defects originating from pipe fabrication
    "metal loss-manufacturing anomaly": "Metal Loss Manufacturing",
    "metal loss manufacturing": "Metal Loss Manufacturing",
    "Metal Loss Manufacturing Anomaly": "Metal Loss Manufacturing",
    "Seam Weld Manufacturing Anomaly": "Seam Weld Manufacturing",
    "Seam Weld Anomaly - B": "Seam Weld Anomaly",
    "Seam Weld Dent": "Seam Weld Dent",
    # Girth weld — circumferential welds joining pipe joints
    "Girth Weld": "Girth Weld",
    "GirthWeld": "Girth Weld",
    "Girth Weld Anomaly": "Girth Weld Anomaly",
    # Dent — mechanical deformation of the pipe wall
    "Dent": "Dent",
    # Structural / pipeline components
    "Bend": "Bend",
    "Field Bend": "Bend",
    "Tap": "Tap",
    "Valve": "Valve",
    "Tee": "Tee",
    "Stopple Tee": "Tee",
    "Flange": "Flange",
    "Attachment": "Attachment",
    # Reference / survey markers
    "Above Ground Marker": "AGM",
    "AGM": "AGM",
    "Magnet": "Magnet",
    "Support": "Support",
    "Cathodic Protection Point": "CP Point",
    # Area start/end markers — paired features that bracket repairs,
    # casings, sleeves, composite wraps, and pipeline facilities
    "Area Start Launcher": "Launcher Start",
    "Area End Launcher": "Launcher End",
    "Area End Launch Trap": "Launcher End",
    "Area Start Receiver": "Receiver Start",
    "Area End Receiver": "Receiver End",
    "Area Start Receive Trap": "Receiver Start",
    "Area Start Installation": "Installation Start",
    "Area End Installation": "Installation End",
    "Area Start Sleeve": "Sleeve Start",
    "Area End Sleeve": "Sleeve End",
    "Start Sleeve": "Sleeve Start",
    "End Sleeve": "Sleeve End",
    "Area Start Composite Wrap": "Composite Wrap Start",
    "Area End Composite Wrap": "Composite Wrap End",
    "Start Composite Wrap": "Composite Wrap Start",
    "End Composite Wrap": "Composite Wrap End",
    "Area Start Casing": "Casing Start",
    "Area End Casing": "Casing End",
    "Start Casing": "Casing Start",
    "End Casing": "Casing End",
    "Area Start Tee": "Tee Start",
    "Area End Tee": "Tee End",
    "Start Repair Marker": "Repair Start",
    "End Repair Marker": "Repair End",
    "Start Recoat": "Recoat Start",
    "End Recoat": "Recoat End",
}

# Feature types classified as corrosion or defect anomalies.
# Only these types participate in cross-run matching and growth analysis;
# structural features (bends, valves, etc.) are excluded.
ANOMALY_TYPES = {
    "Metal Loss", "Cluster", "Metal Loss Manufacturing",
    "Dent", "Seam Weld Manufacturing", "Seam Weld Anomaly",
    "Seam Weld Dent", "Girth Weld Anomaly",
}

# Girth welds serve as fixed reference points for joint-level alignment.
# Because every ILI tool reliably detects girth welds, they anchor the
# coordinate system so that anomaly positions can be compared across runs.
REFERENCE_TYPES = {"Girth Weld"}

# Cross-run type compatibility for anomaly matching.
# Some types are interchangeable because vendors report them differently:
#   - Metal Loss <-> Cluster: 2007 Rosen reported clusters that Baker Hughes
#     later decomposed into individual metal-loss indications.
#   - Metal Loss Mfg <-> Seam Weld Mfg: same root cause, different labels.
#   - Dent <-> Seam Weld Dent: dent near a seam weld vs. plain dent.
COMPATIBLE_TYPES = {
    "Metal Loss": {"Metal Loss", "Cluster"},
    "Cluster": {"Metal Loss", "Cluster"},
    "Metal Loss Manufacturing": {"Metal Loss Manufacturing", "Seam Weld Manufacturing"},
    "Seam Weld Manufacturing": {"Metal Loss Manufacturing", "Seam Weld Manufacturing"},
    "Dent": {"Dent", "Seam Weld Dent"},
    "Seam Weld Dent": {"Dent", "Seam Weld Dent"},
    "Seam Weld Anomaly": {"Seam Weld Anomaly"},
    "Girth Weld Anomaly": {"Girth Weld Anomaly"},
}

# ── ID/OD normalization for 2007 ─────────────────────────────────────────────
# The 2007 Rosen report uses a YES/NO "internal" flag instead of the
# explicit "ID"/"OD" labels used by 2015 and 2022 Baker Hughes reports.
# This map converts the 2007 convention to a unified Internal/External form.
ID_OD_MAP_2007 = {
    "YES": "Internal",
    "NO": "External",
    "N/A": "Unknown",
}

# ── Matching tolerances ───────────────────────────────────────────────────────
# Maximum allowable differences when deciding whether two features from
# different ILI runs could be the same physical anomaly.
DISTANCE_TOLERANCE_FT = 3.0   # max axial offset along the pipe (feet)
CLOCK_TOLERANCE_HOURS = 1.0   # max circumferential offset (clock-hours; 1 hr = 30 deg)
DEPTH_TOLERANCE_PCT = 15.0    # max wall-thickness depth difference (% points)
LENGTH_TOLERANCE_IN = 3.0     # max axial length difference (inches)
WIDTH_TOLERANCE_IN = 3.0      # max circumferential width difference (inches)

# ── Similarity weights ───────────────────────────────────────────────────────
# Relative importance of each feature attribute when scoring a candidate match.
# Distance gets the highest weight because the odometer-based position is the
# most consistently accurate measurement across all three vendor tools.
# Clock position is the next strongest physical constraint (circumferential
# location rarely shifts).  Depth, dimensions, and type compatibility carry
# lower weight because they are expected to change with corrosion growth and
# because vendor sizing methodologies differ.
WEIGHT_DISTANCE = 0.35   # axial position along pipe
WEIGHT_CLOCK = 0.25      # circumferential position on pipe wall
WEIGHT_DEPTH = 0.20      # wall-loss depth percentage
WEIGHT_DIMENSIONS = 0.10 # length + width combined
WEIGHT_TYPE = 0.10       # event-type compatibility bonus

# ── Confidence thresholds ────────────────────────────────────────────────────
# Weighted similarity scores are classified into tiers to guide engineering
# review: HIGH matches are near-certain (auto-accept), MEDIUM need quick
# review, and anything below LOW is rejected as no match.
HIGH_CONFIDENCE = 0.85   # >= 0.85 — high-confidence match
MEDIUM_CONFIDENCE = 0.60 # >= 0.60 — medium-confidence, flag for review
LOW_CONFIDENCE = 0.40    # >= 0.40 — low-confidence, likely coincidental

# ── Growth thresholds ────────────────────────────────────────────────────────
# Domain-specific limits for corrosion growth analysis.
MAX_PLAUSIBLE_GROWTH_RATE = 5.0  # %/yr; rates above this are likely sizing
                                 # errors or mis-matches, not real corrosion
WALL_LOSS_REPAIR_THRESHOLD = 80.0  # % of wall thickness; at or above this
                                   # depth the feature triggers mandatory
                                   # repair / dig-verification per ASME B31.8S

# ── Canonical columns expected after normalization ───────────────────────────
# After ingestion and column mapping, every row — regardless of source year —
# must contain at least these columns.  The alignment and matching layers
# depend on this guaranteed schema.
CANONICAL_COLS = [
    "joint_number", "joint_length_ft", "wall_thickness_in",
    "dist_to_us_weld_ft", "log_distance_ft", "event_type",
    "depth_pct", "length_in", "width_in", "clock_position",
    "id_od", "comments", "elevation_ft", "run_year",
    "is_anomaly", "is_girth_weld",
]
