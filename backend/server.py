"""REST API layer for CorroSight — the ILI Pipeline Integrity Analysis Platform.

This FastAPI server loads and caches all analysis results on startup (ILI runs,
girth-weld alignment, anomaly matching, growth-rate calculations, and multi-run
chain analysis), then serves ~20 JSON endpoints consumed by the Angular frontend.

Endpoints cover: run summaries, data quality, EDA distributions, alignment stats,
pairwise match results, corrosion growth / risk, multi-run trajectories, pipeline
schematic data, Excel export, AI-powered chat/report/insights, integrity
dashboard analytics, runtime AI configuration, and file-upload pipeline re-runs.
"""

import os
import threading
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File as FastAPIFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from data_ingestion import (
    load_all_runs, load_summary, get_anomalies, get_girth_welds,
    data_quality_report, column_completeness,
)
from alignment import match_girth_welds, apply_distance_correction, compute_alignment_stats
from matching import match_anomalies
from growth import calculate_growth_rates, growth_summary_stats, top_concerns
from multi_run import run_full_analysis, export_results
from config import YEARS_BETWEEN, RUN_YEARS
from ai_service import (
    chat, generate_executive_report, generate_anomaly_narratives,
    generate_chart_insights, generate_nl_chart,
    set_api_key, get_api_key_status,
)
from virtual_ili import predict_future_inspection
from integrity_analytics import compute_integrity_dashboard

app = FastAPI(title="CorroSight API")

# CORS middleware: allow the Angular dev server (localhost:4200/4201) to call
# this API without browser cross-origin errors during local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:4201", "http://127.0.0.1:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state (computed on startup) ────────────────────────────────────────
# Path to the default ILI Excel dataset bundled with the project.
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ILIDataV2.xlsx")

# In-memory cache that stores all computed analysis results (runs, alignment,
# matches, growth metrics, etc.) so every API response is an instant dict lookup
# instead of re-computing on each request.
cache = {}

# Tracks progress of a background pipeline re-run triggered via /api/run-pipeline.
# The Angular frontend polls /api/run-pipeline/status to update a progress bar.
pipeline_progress = {
    "status": "idle",  # idle | running | completed | error
    "step": "",
    "step_number": 0,
    "total_steps": 6,
    "stats": {},
    "error": None,
}


def _nan_to_none(obj):
    """Recursively convert NaN/NaT and numpy scalars to Python-native types.

    Critical for JSON serialization: pandas/numpy NaN and NaT are not valid
    JSON values, and numpy int64/float64/bool_ aren't natively serializable.
    This walks dicts/lists and converts everything to None, int, float, or bool.
    """
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a JSON-serializable list of dicts.

    Replaces NaN with None first (for clean JSON null values), then applies
    the recursive _nan_to_none pass to catch numpy scalar types.
    """
    if df is None or df.empty:
        return []
    records = df.replace({np.nan: None}).to_dict(orient="records")
    return [_nan_to_none(r) for r in records]


@app.on_event("startup")
def startup():
    """Load data and run the full 5-step analysis pipeline on startup.

    1. Load all ILI runs and summary metadata from the Excel file.
    2. Match girth welds across runs to establish a common reference frame.
    3. Apply distance correction using the girth-weld alignment offsets.
    4. Run the full analysis suite (pairwise matching, multi-run chaining).
    5. Calculate growth rates for every pairwise match set.

    All results are stored in `cache` so API endpoints return instantly.
    """
    runs = load_all_runs(DATA_PATH)
    summary = load_summary(DATA_PATH)
    gw_alignment = match_girth_welds(runs)
    corrected_runs = apply_distance_correction(runs, gw_alignment)
    results = run_full_analysis(corrected_runs)

    # Add growth metrics to pairwise matches
    for key in results.get("pairwise", {}):
        matches = results["pairwise"][key].get("matches", pd.DataFrame())
        if not matches.empty:
            results["pairwise"][key]["matches"] = calculate_growth_rates(matches)

    cache["runs"] = runs
    cache["summary"] = summary
    cache["gw_alignment"] = gw_alignment
    cache["corrected_runs"] = corrected_runs
    cache["results"] = results
    cache["alignment_stats"] = compute_alignment_stats(gw_alignment)


# ── API Endpoints ─────────────────────────────────────────────────────────────

# --- Run Summary: metadata for the frontend header cards ---

@app.get("/api/summary")
def get_summary():
    """Run metadata and high-level counts."""
    runs = cache["runs"]
    summary = cache["summary"]
    run_info = []
    for year in sorted(runs.keys()):
        df = runs[year]
        anom = get_anomalies(df)
        gw = get_girth_welds(df)
        info = {
            "year": year,
            "total_rows": len(df),
            "anomaly_count": len(anom),
            "girth_weld_count": len(gw),
        }
        row = summary[summary["Start Date"].dt.year == year]
        if not row.empty:
            info["vendor"] = str(row.iloc[0].get("ILI Vendor", ""))
            info["tool_type"] = str(row.iloc[0].get("Tool Type", ""))
            info["start_date"] = str(row.iloc[0].get("Start Date", ""))
        run_info.append(info)
    return {"runs": run_info}


# --- Data Quality: heatmap of completeness and issue counts per run ---

@app.get("/api/quality")
def get_quality():
    """Data quality report and column completeness."""
    runs = cache["runs"]
    dq = data_quality_report(runs)
    comp = column_completeness(runs)
    return {
        "quality_report": _df_to_records(dq),
        "completeness": _df_to_records(comp),
    }


# --- Exploratory Data Analysis: distributions for depth, length, width, clock ---

@app.get("/api/eda")
def get_eda():
    """EDA data: event distributions, depth/length/width histograms, clock distributions."""
    runs = cache["runs"]
    result = {"event_types": {}, "depth_data": {}, "length_data": {}, "width_data": {}, "clock_data": {}}

    for year in sorted(runs.keys()):
        df = runs[year]
        anom = get_anomalies(df)

        # Event type counts
        evt_counts = df[df["is_anomaly"]]["event_type"].value_counts()
        result["event_types"][str(year)] = evt_counts.to_dict()

        # Distributions (raw values for frontend to bin)
        result["depth_data"][str(year)] = anom["depth_pct"].dropna().tolist()
        result["length_data"][str(year)] = anom["length_in"].dropna().tolist()
        result["width_data"][str(year)] = anom["width_in"].dropna().tolist()
        result["clock_data"][str(year)] = anom["clock_hours"].dropna().tolist()

    return _nan_to_none(result)


# --- Alignment: girth-weld alignment table and drift statistics ---

@app.get("/api/alignment")
def get_alignment():
    """Girth weld alignment table and drift statistics."""
    gw = cache["gw_alignment"]
    stats = cache["alignment_stats"]
    return {
        "alignment_table": _df_to_records(gw),
        "stats": _nan_to_none(stats),
    }


# --- Pairwise Matches: anomaly match results with filtering by run pair ---

@app.get("/api/matches/{pair}")
def get_matches(pair: str):
    """Match results for a run pair (e.g., '2007-2015')."""
    parts = pair.split("-")
    if len(parts) != 2:
        return {"error": "Use format YYYY-YYYY, e.g. 2007-2015"}

    y_early, y_later = int(parts[0]), int(parts[1])
    results = cache["results"]
    pairwise = results.get("pairwise", {})
    key = (y_early, y_later)

    if key in pairwise:
        match_result = pairwise[key]
    elif "direct_first_last" in results and key == (2007, 2022):
        match_result = results["direct_first_last"]
    else:
        return {"error": f"No match data for {pair}"}

    return {
        "stats": _nan_to_none(match_result.get("stats", {})),
        "matches": _df_to_records(match_result.get("matches", pd.DataFrame())),
        "new_anomalies_count": len(match_result.get("new_anomalies", pd.DataFrame())),
        "missing_anomalies_count": len(match_result.get("missing_anomalies", pd.DataFrame())),
    }


# --- Growth Analysis: growth rates, risk matrix, remaining life, top concerns ---

@app.get("/api/growth/{pair}")
def get_growth(pair: str):
    """Growth analysis for a run pair."""
    parts = pair.split("-")
    if len(parts) != 2:
        return {"error": "Use format YYYY-YYYY"}

    y_early, y_later = int(parts[0]), int(parts[1])
    results = cache["results"]
    pairwise = results.get("pairwise", {})
    key = (y_early, y_later)

    if key not in pairwise:
        return {"error": f"No data for {pair}"}

    matches = pairwise[key].get("matches", pd.DataFrame())
    if matches.empty:
        return {"stats": {}, "top_concerns": [], "growth_rates": []}

    stats = growth_summary_stats(matches)
    tc = top_concerns(matches, n=20)

    # Growth rate values for histogram (exclude negative — measurement artifacts)
    rates_series = matches["depth_growth_rate"].dropna()
    rates = rates_series[rates_series >= 0].tolist()

    # Remaining life values (cap at 30 years for operational relevance)
    remaining_life = matches["remaining_life_years"].dropna()
    remaining_life = remaining_life[remaining_life < 30].tolist()

    # Risk matrix data
    risk_data = matches.dropna(subset=["later_depth_pct", "depth_growth_rate"])
    risk_points = risk_data[["later_depth_pct", "depth_growth_rate",
                             "risk_category", "later_joint", "later_distance",
                             "confidence_label"]].to_dict(orient="records") if not risk_data.empty else []

    return _nan_to_none({
        "stats": stats,
        "top_concerns": _df_to_records(tc),
        "growth_rates": rates,
        "remaining_life": remaining_life,
        "risk_points": risk_points,
    })


# --- Multi-Run: 3-run chain matches and growth trajectories ---

@app.get("/api/multirun")
def get_multirun():
    """Multi-run chain matching and growth trends."""
    results = cache["results"]
    chain = results.get("chain", {})

    triple = chain.get("triple_matches", pd.DataFrame())
    lifecycle = chain.get("lifecycle_summary", pd.DataFrame())

    # Extract trajectory data for top 10 fastest-growing
    trajectories = []
    if not triple.empty and "overall_growth_rate" in triple.columns:
        top10 = triple.nlargest(10, "overall_growth_rate")
        for _, row in top10.iterrows():
            traj = {
                "joint": row.get("joint_2022"),
                "depths": [row.get("depth_2007"), row.get("depth_2015"), row.get("depth_2022")],
                "years": [2007, 2015, 2022],
                "linear_rate": row.get("linear_rate"),
                "predicted_2030": row.get("predicted_2030"),
                "is_accelerating": row.get("is_accelerating", False),
            }
            trajectories.append(traj)

    return _nan_to_none({
        "triple_count": len(triple),
        "lifecycle": _df_to_records(lifecycle),
        "trajectories": trajectories,
        "triple_matches": _df_to_records(triple.head(200)),  # Cap for performance
    })


# --- Pipeline Schematic View: anomaly positions for the unrolled pipe visual ---

@app.get("/api/pipeline-view")
def get_pipeline_view():
    """All anomaly positions and match connections for the pipeline schematic."""
    corrected_runs = cache["corrected_runs"]
    results = cache["results"]

    # Anomalies per run
    anomalies_by_run = {}
    for year in sorted(corrected_runs.keys()):
        anom = get_anomalies(corrected_runs[year])
        anomalies_by_run[str(year)] = anom[[
            "corrected_distance", "clock_hours", "depth_pct",
            "event_type", "joint_number",
        ]].to_dict(orient="records")

    # Match connections (from pairwise results)
    connections = []
    for (ye, yl), mr in results.get("pairwise", {}).items():
        matches = mr.get("matches", pd.DataFrame())
        if matches.empty:
            continue
        for _, m in matches.iterrows():
            connections.append({
                "earlier_dist": m.get("earlier_distance"),
                "earlier_clock": m.get("earlier_clock"),
                "later_dist": m.get("later_distance"),
                "later_clock": m.get("later_clock"),
                "confidence": m.get("confidence_label"),
                "pair": f"{ye}-{yl}",
            })

    # Girth weld positions (2022 reference)
    gw = get_girth_welds(corrected_runs[2022])
    gw_positions = gw["corrected_distance"].tolist()

    return _nan_to_none({
        "anomalies": anomalies_by_run,
        "connections": connections,
        "girth_welds": gw_positions,
    })


# --- Export: download Excel workbook with all analysis results ---

@app.get("/api/export")
def export_excel():
    """Export results to Excel and return file."""
    output_path = os.path.join(os.path.dirname(__file__), "data", "ILI_Analysis_Results.xlsx")
    export_results(cache["results"], output_path)
    return FileResponse(
        output_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="ILI_Analysis_Results.xlsx",
    )


# ── AI-Powered Endpoints ────────────────────────────────────────────────────
# Chat, executive report, anomaly narratives, chart insights, natural-language
# chart generation, and virtual ILI prediction — all powered by the AI service.

# Pydantic request model for the AI copilot chat endpoint.
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


# Pydantic request model for chart-specific AI insight generation.
class InsightRequest(BaseModel):
    chart_type: str
    data: dict


# Pydantic request model for natural-language-to-Plotly chart generation.
class NLChartRequest(BaseModel):
    query: str


# --- AI Chat: conversational copilot for asking questions about the data ---

@app.post("/api/chat")
def api_chat(req: ChatRequest):
    """AI copilot chat — ask questions about pipeline data in plain English."""
    response = chat(req.message, req.history, cache)
    return {"response": response}


# --- AI Report: auto-generated executive summary of the full analysis ---

@app.get("/api/ai-report")
def api_ai_report():
    """Generate AI executive summary report."""
    report = generate_executive_report(cache)
    return {"report": report}


# --- AI Narratives: plain-English explanations for top concern anomalies ---

@app.get("/api/ai-narratives")
def api_ai_narratives():
    """Generate AI narrative cards for top concern anomalies."""
    narratives = generate_anomaly_narratives(cache)
    return _nan_to_none({"narratives": narratives})


# --- AI Insights: contextual annotations for any chart ---

@app.post("/api/ai-insights")
def api_ai_insights(req: InsightRequest):
    """Generate AI chart annotations/insights."""
    annotations = generate_chart_insights(req.chart_type, req.data, cache)
    return {"annotations": annotations}


# --- AI Natural-Language Chart: generate a Plotly chart spec from plain English ---

@app.post("/api/nl-chart")
def api_nl_chart(req: NLChartRequest):
    """Generate a Plotly chart from natural language."""
    result = generate_nl_chart(req.query, cache)
    return _nan_to_none(result)


# --- Virtual ILI: predict what a future inspection would find at a target year ---

@app.get("/api/virtual-ili/{year}")
def api_virtual_ili(year: int):
    """Predict what a future ILI inspection would find at target year."""
    result = predict_future_inspection(cache, year)
    return _nan_to_none(result)


# ── Integrity Dashboard ──────────────────────────────────────────────────────
# Segment-level risk heatmap, ASME B31G burst-pressure assessment, prioritised
# dig list, and population-level corrosion growth analytics.

@app.get("/api/integrity-dashboard")
def api_integrity_dashboard():
    """Segment risk heatmap, ASME B31G interaction assessment,
    automated dig list, and population growth analytics."""
    result = compute_integrity_dashboard(cache)
    return _nan_to_none(result)


# ── API Key Configuration ─────────────────────────────────────────────────────
# Allows the frontend to set/check the xAI API key at runtime without restarting.

# Pydantic request model for setting the AI service API key.
class ApiKeyRequest(BaseModel):
    key: str


@app.post("/api/set-api-key")
def api_set_api_key(req: ApiKeyRequest):
    """Set the xAI API key at runtime (validates before accepting)."""
    result = set_api_key(req.key)
    return {"status": "ok" if result.get("configured") else "error", **result}


@app.get("/api/ai-status")
def api_ai_status():
    """Get the current AI service status."""
    return get_api_key_status()


# ── Run Pipeline ───────────────────────────────────────────────────────────────
# File upload + background analysis re-run. The frontend uploads a new Excel file,
# the pipeline runs in a daemon thread, and the frontend polls for progress.


def _run_pipeline_background(file_path: str):
    """Run the full 6-step analysis pipeline in a background thread.

    Steps:
      1. Load ILI data from the uploaded Excel file.
      2. Match girth welds across runs (common reference frame).
      3. Apply distance correction using girth-weld offsets.
      4. Run anomaly matching (Hungarian algorithm) across all run pairs.
      5. Calculate growth rates and risk scores for matched anomalies.
      6. Finalize and atomically swap the global cache.

    Progress is written to `pipeline_progress` so the frontend can poll status.
    """
    global pipeline_progress
    try:
        # Step 1: Load data
        pipeline_progress.update({"step": "Loading ILI data from Excel...", "step_number": 1})
        pipeline_progress["stats"]["filename"] = os.path.basename(file_path)
        runs = load_all_runs(file_path)
        summary = load_summary(file_path)
        total_rows = sum(len(df) for df in runs.values())
        run_years = sorted(runs.keys())
        pipeline_progress["stats"]["total_rows"] = total_rows
        pipeline_progress["stats"]["runs_loaded"] = len(runs)

        # Step 2: Match girth welds
        pipeline_progress.update({"step": "Matching girth welds across runs...", "step_number": 2})
        gw_alignment = match_girth_welds(runs)
        pipeline_progress["stats"]["girth_welds_matched"] = len(gw_alignment)

        # Step 3: Apply distance correction
        pipeline_progress.update({"step": "Applying distance correction...", "step_number": 3})
        corrected_runs = apply_distance_correction(runs, gw_alignment)

        # Step 4: Run matching & analysis
        pipeline_progress.update({"step": "Running anomaly matching (Hungarian algorithm)...", "step_number": 4})
        results = run_full_analysis(corrected_runs)
        total_matches = sum(
            len(mr.get("matches", pd.DataFrame()))
            for mr in results.get("pairwise", {}).values()
        )
        pipeline_progress["stats"]["total_matches"] = total_matches

        # Step 5: Calculate growth rates
        pipeline_progress.update({"step": "Calculating growth rates & risk scores...", "step_number": 5})
        for key in results.get("pairwise", {}):
            matches = results["pairwise"][key].get("matches", pd.DataFrame())
            if not matches.empty:
                results["pairwise"][key]["matches"] = calculate_growth_rates(matches)

        # Step 6: Finalize — build new cache and swap atomically
        pipeline_progress.update({"step": "Finalizing results...", "step_number": 6})
        alignment_stats = compute_alignment_stats(gw_alignment)

        chain = results.get("chain", {})
        triple = chain.get("triple_matches", pd.DataFrame())
        pipeline_progress["stats"]["triple_matches"] = len(triple) if not triple.empty else 0

        # Atomically update cache
        new_cache = {
            "runs": runs,
            "summary": summary,
            "gw_alignment": gw_alignment,
            "corrected_runs": corrected_runs,
            "results": results,
            "alignment_stats": alignment_stats,
        }
        cache.update(new_cache)

        pipeline_progress.update({"status": "completed", "step": "Pipeline analysis complete!"})

    except Exception as e:
        pipeline_progress.update({"status": "error", "error": str(e), "step": f"Error: {str(e)}"})


# --- File Upload + Background Re-run: upload new Excel, re-run entire pipeline ---

@app.post("/api/run-pipeline")
async def api_run_pipeline(file: UploadFile = FastAPIFile(...)):
    """Trigger a full pipeline re-run using an uploaded Excel file."""
    global pipeline_progress
    if pipeline_progress["status"] == "running":
        return {"status": "already_running", "progress": pipeline_progress}

    # Save uploaded file
    upload_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    pipeline_progress = {
        "status": "running",
        "step": "Initializing...",
        "step_number": 0,
        "total_steps": 6,
        "stats": {"filename": file.filename},
        "error": None,
    }
    thread = threading.Thread(target=_run_pipeline_background, args=(file_path,), daemon=True)
    thread.start()
    return {"status": "started", "filename": file.filename}


# --- Pipeline Status: polled by the frontend to track background re-run progress ---

@app.get("/api/run-pipeline/status")
def api_pipeline_status():
    """Get the current pipeline run progress."""
    return pipeline_progress
