"""
AI service: LLM integration via xAI (Grok 4.1 Fast) for pipeline integrity analysis.

Provides AI-powered features including:
  - Chat copilot with full pipeline context awareness
  - Executive summary report generation
  - Anomaly narrative "story card" generation
  - Chart insight annotations
  - Natural-language-to-Plotly chart generation

All features gracefully degrade to regex-based / template-driven responses
when no XAI_API_KEY is configured, so the app remains functional without
an API key.
"""

import os
import re
import json
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ---------------------------------------------------------------------------
# API key loading -- checks .env file in the backend directory first, then
# falls back to the XAI_API_KEY environment variable. This two-tier lookup
# lets developers store the key in .env locally while still supporting
# deployment via environment variables.
# ---------------------------------------------------------------------------
def _load_env_key() -> str:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("XAI_API_KEY=") and not line.startswith("#"):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
    return os.environ.get("XAI_API_KEY", "")

# Module-level state: loaded once at import, but can be overwritten at
# runtime via set_api_key().
XAI_API_KEY = _load_env_key()
XAI_BASE_URL = "https://api.x.ai/v1"
MODEL = "grok-4-1-fast"


# ---------------------------------------------------------------------------
# Runtime API key configuration.
# Accepts a new key from the frontend settings panel, validates it by making
# a lightweight test completion call, and updates the module-level key only
# on success. If the test call fails with a non-auth error (e.g. network
# timeout, rate limit), the key is still accepted because it may be valid.
# ---------------------------------------------------------------------------
def set_api_key(key: str) -> dict:
    """Set the xAI API key at runtime after validating it with a test call."""
    global XAI_API_KEY
    if not key:
        return {"configured": False, "error": "No key provided"}
    if not HAS_OPENAI:
        XAI_API_KEY = key
        return {"configured": True, "model": MODEL, "error": None}

    # Validate by making a lightweight test call
    try:
        test_client = OpenAI(api_key=key, base_url=XAI_BASE_URL)
        test_client.chat.completions.create(
            model=MODEL,
            max_tokens=5,
            messages=[{"role": "user", "content": "Hi"}],
        )
        XAI_API_KEY = key
        return {"configured": True, "model": MODEL, "error": None}
    except Exception as e:
        err = str(e)
        # Only reject the key on clear authentication / bad-request errors
        if "Incorrect API key" in err or "invalid" in err.lower() or "401" in err or "400" in err:
            return {"configured": False, "error": "Invalid API key. Get yours at https://console.x.ai"}
        # Other errors (network, rate limit) — key might be fine, accept it
        XAI_API_KEY = key
        return {"configured": True, "model": MODEL, "error": None}


def get_api_key_status() -> dict:
    """Return whether the API key is configured and the model in use."""
    return {"configured": bool(XAI_API_KEY), "model": MODEL}


# ---------------------------------------------------------------------------
# Client factory.
# Creates a fresh OpenAI-compatible client on every call instead of caching
# a singleton, so that runtime key changes made via set_api_key() take
# effect immediately without restarting the server.
# ---------------------------------------------------------------------------
def _get_client() -> Optional["OpenAI"]:
    if not HAS_OPENAI or not XAI_API_KEY:
        return None
    return OpenAI(api_key=XAI_API_KEY, base_url=XAI_BASE_URL)

# ---------------------------------------------------------------------------
# Context builders -- these functions assemble plain-text summaries of the
# analysis cache that are injected into the LLM system prompt so the model
# can reason about the pipeline data without needing direct data access.
# ---------------------------------------------------------------------------


def _build_pipeline_context(cache: dict) -> str:
    """
    Assemble a structured text summary of all analysis results for the LLM.

    Covers: per-run row/anomaly/girth-weld counts, alignment drift stats,
    pairwise match counts with confidence breakdown and growth rate
    statistics, and multi-run chain tracking (triple matches, lifecycle).
    """
    parts = []

    # Summary
    runs = cache.get("runs", {})
    parts.append("## CorroSight - Pipeline Data Summary")
    for year in sorted(runs.keys()):
        df = runs[year]
        anom_count = df["is_anomaly"].sum() if "is_anomaly" in df.columns else 0
        gw_count = df["is_girth_weld"].sum() if "is_girth_weld" in df.columns else 0
        parts.append(f"- {year}: {len(df)} total rows, {anom_count} anomalies, {gw_count} girth welds")

    # Alignment stats
    astats = cache.get("alignment_stats", {})
    if astats:
        parts.append(f"\n## Alignment Stats")
        parts.append(f"- Matched girth welds: {astats.get('matched_joints', 'N/A')}")
        parts.append(f"- Max drift: {astats.get('max_drift_ft', 'N/A')} ft")
        parts.append(f"- Mean drift: {astats.get('mean_drift_ft', 'N/A')} ft")

    # Pairwise match stats (one section per year-pair)
    results = cache.get("results", {})
    pairwise = results.get("pairwise", {})
    for (ye, yl), mr in pairwise.items():
        stats = mr.get("stats", {})
        matches = mr.get("matches", pd.DataFrame())
        new_anom = mr.get("new_anomalies", pd.DataFrame())
        missing = mr.get("missing_anomalies", pd.DataFrame())
        parts.append(f"\n## Matching: {ye} to {yl}")
        parts.append(f"- Matches found: {len(matches)}")
        parts.append(f"- New anomalies in {yl}: {len(new_anom)}")
        parts.append(f"- Missing from {ye}: {len(missing)}")
        if not matches.empty and "confidence_label" in matches.columns:
            conf_counts = matches["confidence_label"].value_counts().to_dict()
            parts.append(f"- Confidence breakdown: {conf_counts}")
        if not matches.empty and "depth_growth_rate" in matches.columns:
            rates = matches["depth_growth_rate"].dropna()
            if len(rates) > 0:
                parts.append(f"- Mean growth rate: {rates.mean():.3f} %/yr")
                parts.append(f"- Max growth rate: {rates.max():.3f} %/yr")
                parts.append(f"- Negative growth (apparent shrinkage): {(rates < 0).sum()} anomalies")

    # Multi-run chain summary (anomalies tracked across all 3 inspections)
    chain = results.get("chain", {})
    triple = chain.get("triple_matches", pd.DataFrame())
    lifecycle = chain.get("lifecycle_summary", pd.DataFrame())
    if not triple.empty:
        parts.append(f"\n## Multi-Run Tracking (3 inspections)")
        parts.append(f"- Anomalies tracked across all 3 runs: {len(triple)}")
        if "overall_growth_rate" in triple.columns:
            parts.append(f"- Mean overall growth rate: {triple['overall_growth_rate'].mean():.3f} %/yr")
    if not lifecycle.empty:
        for _, row in lifecycle.iterrows():
            parts.append(f"- {row['Category']}: {row['Count']}")

    return "\n".join(parts)


def _build_top_concerns_context(cache: dict, n: int = 20) -> str:
    """
    Format the top N highest-risk anomalies as a numbered list for the LLM.

    Anomalies are ranked by risk_score from each pairwise match set and
    include joint number, distance, clock position, depth, growth rate,
    remaining life estimate, and risk category.
    """
    results = cache.get("results", {})
    pairwise = results.get("pairwise", {})
    parts = []

    for (ye, yl), mr in pairwise.items():
        matches = mr.get("matches", pd.DataFrame())
        if matches.empty or "risk_score" not in matches.columns:
            continue
        top_n = matches.nlargest(n, "risk_score")
        parts.append(f"\n## Top {n} Concerns ({ye}-{yl})")
        for i, (_, row) in enumerate(top_n.iterrows(), 1):
            parts.append(
                f"{i}. Joint {row.get('later_joint', '?')}, "
                f"Dist {row.get('later_distance', 0):.0f} ft, "
                f"Clock {row.get('later_clock', 0):.1f}, "
                f"Depth {row.get('later_depth_pct', 0):.1f}%, "
                f"Rate {row.get('depth_growth_rate', 0):.2f} %/yr, "
                f"Remaining life {row.get('remaining_life_years', 'N/A')} yr, "
                f"Risk: {row.get('risk_category', 'Unknown')}"
            )

    return "\n".join(parts)


def _build_trajectory_context(cache: dict) -> str:
    """
    Format the top 10 fastest-growing triple-match anomalies for the LLM.

    Each entry shows the 3-run depth history (2007 -> 2015 -> 2022),
    overall growth rate, 2030 prediction, and whether growth is
    accelerating between inspection intervals.
    """
    results = cache.get("results", {})
    chain = results.get("chain", {})
    triple = chain.get("triple_matches", pd.DataFrame())
    if triple.empty:
        return ""

    parts = ["## Multi-Run Growth Trajectories"]
    if "overall_growth_rate" in triple.columns:
        top10 = triple.nlargest(10, "overall_growth_rate")
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            parts.append(
                f"{i}. Joint {row.get('joint_2022', '?')}: "
                f"2007={row.get('depth_2007', '?')}% -> "
                f"2015={row.get('depth_2015', '?')}% -> "
                f"2022={row.get('depth_2022', '?')}%, "
                f"Rate: {row.get('overall_growth_rate', 0):.3f} %/yr, "
                f"Predicted 2030: {row.get('predicted_2030', 'N/A')}%, "
                f"Accelerating: {row.get('is_accelerating', False)}"
            )

    return "\n".join(parts)


# ── Public API ───────────────────────────────────────────────────────────────

def chat(message: str, history: list[dict], cache: dict) -> str:
    """
    Handle a chat message with full pipeline context.

    Builds a system prompt containing the complete pipeline analysis context
    (run summary, top concerns, growth trajectories), appends the last 10
    messages of chat history for conversational continuity, and sends
    everything to the LLM. Falls back to _fallback_chat() when no API key
    is configured, providing regex-routed data-driven answers instead.
    """
    client = _get_client()
    if client is None:
        return _fallback_chat(message, cache)

    context = _build_pipeline_context(cache)
    concerns = _build_top_concerns_context(cache)
    trajectories = _build_trajectory_context(cache)

    system_prompt = f"""You are an expert pipeline integrity engineer AI assistant. You have access to the complete analysis results from 3 In-Line Inspection (ILI) runs on a pipeline (2007, 2015, 2022).

Your role: Help engineers understand their pipeline data, identify risks, and make informed integrity decisions. Be precise with numbers and always cite specific data.

{context}

{concerns}

{trajectories}

Guidelines:
- Answer with specific data from the analysis when possible
- Highlight safety-critical findings proactively
- Use engineering terminology appropriately
- When discussing risk, reference the specific anomalies by joint number and location
- If asked about something not in the data, say so clearly
- Keep responses concise but thorough"""

    messages = [{"role": "system", "content": system_prompt}]
    # Include last 10 messages of history for conversational context
    for h in history[-10:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=2048,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI service error: {str(e)}"


def generate_executive_report(cache: dict) -> str:
    """
    Generate a comprehensive executive summary report.

    Sends a structured prompt to the LLM requesting a professional
    pipeline integrity assessment covering: executive summary, inspection
    overview, alignment/matching results, corrosion growth trends, critical
    findings with specific joint citations, prioritized repair actions,
    and next-inspection timing recommendations. Falls back to a plain
    template report when no API key is available.
    """
    client = _get_client()
    if client is None:
        return _fallback_report(cache)

    context = _build_pipeline_context(cache)
    concerns = _build_top_concerns_context(cache)
    trajectories = _build_trajectory_context(cache)

    prompt = f"""You are a senior pipeline integrity engineer writing an executive summary report for management.

Based on the following ILI analysis data, write a professional report covering:

1. **Executive Summary** - Key findings in 2-3 sentences
2. **Inspection Overview** - What was inspected, when, by whom
3. **Alignment & Matching Results** - How well the data aligned across runs
4. **Corrosion Growth Trends** - Overall growth patterns, acceleration, distribution
5. **Critical Findings** - Top risk areas requiring immediate attention (cite specific joints/locations)
6. **Recommended Actions** - Prioritized list of recommended repairs/monitoring
7. **Next Inspection Timing** - When should the next inspection be scheduled based on growth rates

Data:
{context}

{concerns}

{trajectories}

Format: Use markdown headers and bullet points. Include specific numbers. Keep the total report under 1500 words."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Report generation error: {str(e)}"


def generate_anomaly_narratives(cache: dict, n: int = 20) -> list[dict]:
    """
    Batch-generate AI narrative "story cards" for the top N anomalies.

    Collects the top 20 (by risk_score) anomalies from the latest pairwise
    match set, enriches each with 3-run depth history from triple matches
    when available, and sends all of them in a single LLM call. The LLM
    returns a JSON array of {joint, narrative} objects which are merged
    back onto the source anomaly data. Falls back to template-based
    narratives when no API key is configured.
    """
    client = _get_client()

    results = cache.get("results", {})
    pairwise = results.get("pairwise", {})
    chain = results.get("chain", {})
    triple = chain.get("triple_matches", pd.DataFrame())

    # Collect top concerns from the latest pair (2015-2022)
    key = (2015, 2022)
    if key not in pairwise:
        key = list(pairwise.keys())[-1] if pairwise else None
    if key is None:
        return []

    matches = pairwise[key].get("matches", pd.DataFrame())
    if matches.empty or "risk_score" not in matches.columns:
        return []

    top = matches.nlargest(n, "risk_score")
    anomaly_data = []
    for _, row in top.iterrows():
        d = {
            "joint": _safe_val(row.get("later_joint")),
            "distance_ft": _safe_val(row.get("later_distance")),
            "clock": _safe_val(row.get("later_clock")),
            "depth_pct": _safe_val(row.get("later_depth_pct")),
            "earlier_depth_pct": _safe_val(row.get("earlier_depth_pct")),
            "growth_rate": _safe_val(row.get("depth_growth_rate")),
            "remaining_life_years": _safe_val(row.get("remaining_life_years")),
            "risk_score": _safe_val(row.get("risk_score")),
            "risk_category": row.get("risk_category", "Unknown"),
            "confidence": row.get("confidence_label", "Unknown"),
            "event_type": row.get("later_event_type", row.get("event_type", "Metal Loss")),
        }
        # Enrich with 3-run history if this anomaly was tracked across all runs
        if not triple.empty:
            j = row.get("later_joint")
            triple_match = triple[triple["joint_2022"] == j]
            if not triple_match.empty:
                tm = triple_match.iloc[0]
                d["depth_2007"] = _safe_val(tm.get("depth_2007"))
                d["depth_2015"] = _safe_val(tm.get("depth_2015"))
                d["predicted_2030"] = _safe_val(tm.get("predicted_2030"))
                d["is_accelerating"] = bool(tm.get("is_accelerating", False))
        anomaly_data.append(d)

    if client is None:
        return _fallback_narratives(anomaly_data)

    # Build a single batched prompt
    anomaly_list_str = json.dumps(anomaly_data, indent=2, default=str)
    prompt = f"""You are a pipeline integrity engineer. For each of the following high-risk anomalies, write a 2-3 sentence narrative "story card" that explains:
- The anomaly's history and evolution across inspections
- Its current risk level and why it's concerning
- What action should be considered

Return a JSON array of objects with "joint" and "narrative" keys. Each narrative should be specific and data-driven.

Anomalies:
{anomaly_list_str}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract the JSON array from the response (may be wrapped in markdown)
        text = response.choices[0].message.content
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            narratives = json.loads(text[start:end])
        else:
            narratives = [{"joint": d["joint"], "narrative": "Narrative generation failed."} for d in anomaly_data]
        # Merge LLM-generated narratives back onto the source anomaly data
        result = []
        narrative_map = {str(n.get("joint")): n.get("narrative", "") for n in narratives}
        for d in anomaly_data:
            d["narrative"] = narrative_map.get(str(d["joint"]), "")
            result.append(d)
        return result

    except Exception as e:
        return _fallback_narratives(anomaly_data)


def generate_chart_insights(chart_type: str, data: dict, cache: dict) -> list[dict]:
    """
    Generate AI-powered annotations for a specific chart.

    Sends chart type and a truncated data summary to the LLM, asking it to
    identify the top 3-5 most important patterns (outliers, clusters,
    trends, danger zones). Returns a list of Plotly annotation objects
    with text, x/y positions, and arrow settings ready for overlay on the
    chart. Falls back to a placeholder annotation when no API key.
    """
    client = _get_client()
    if client is None:
        return _fallback_insights(chart_type, data, cache)

    data_summary = json.dumps(data, indent=2, default=str)[:3000]

    prompt = f"""You are a pipeline integrity data analyst. Analyze the following chart data and identify the top 3-5 most important patterns, anomalies, or insights.

Chart type: {chart_type}
Data:
{data_summary}

Return a JSON array of Plotly annotation objects. Each must have:
- "text": A concise insight (max 15 words)
- "x": The x-position for the annotation
- "y": The y-position for the annotation
- "showarrow": true or false

Focus on: outliers, clusters, trends, danger zones, and actionable findings."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return []
    except Exception:
        return _fallback_insights(chart_type, data, cache)


def generate_nl_chart(query: str, cache: dict) -> dict:
    """
    Generate a Plotly chart configuration from a natural-language request.

    Describes all available data columns (per-run and match columns) to
    the LLM and asks it to produce a complete Plotly.js-compatible JSON
    config ({data, layout, data_query}) that answers the user's question.
    Returns an error dict when no API key is available, since there is no
    meaningful regex-based fallback for arbitrary chart generation.
    """
    client = _get_client()
    if client is None:
        return {"error": "AI service not available. Set XAI_API_KEY environment variable."}

    # Describe available data columns so the LLM knows what it can plot
    runs = cache.get("runs", {})
    columns_info = {}
    for year in sorted(runs.keys()):
        columns_info[str(year)] = list(runs[year].columns)

    results = cache.get("results", {})
    pairwise = results.get("pairwise", {})
    match_cols = []
    for key, mr in pairwise.items():
        m = mr.get("matches", pd.DataFrame())
        if not m.empty:
            match_cols = list(m.columns)
            break

    prompt = f"""You are a data visualization expert. Generate a Plotly.js chart configuration from this natural language request.

User request: "{query}"

Available data columns:
- Run data per year (2007, 2015, 2022): {json.dumps(columns_info)}
- Match data columns: {json.dumps(match_cols[:30])}

Return a JSON object with:
- "data": array of Plotly trace objects
- "layout": Plotly layout object
- "data_query": a description of what data to extract (I'll handle the extraction)

Only return the JSON object, nothing else. Make the chart visually appealing with good colors and titles."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return {"error": "Could not parse chart config"}
    except Exception as e:
        return {"error": str(e)}


# == Fallback responses (when no API key) =====================================
# These functions provide useful data-driven answers without calling an LLM.
# They are used both as the primary response path when no API key is set
# and as error recovery when LLM calls fail.

def _fallback_chat(message: str, cache: dict) -> str:
    """
    Provide data-driven responses without LLM using regex-based routing.

    Routes the user's message by keyword matching:
      - "summary" / "overview" / "tell me about" -> full pipeline context
      - "risk" / "concern" / "critical" / "worst" -> top-N concerns
        (parses "top 5", "show me 10 worst", etc. for dynamic N)
      - "growth" / "trajectory" / "trend" / "predict" -> trajectory data
      - "match" / "align" / "pair" -> pairwise match counts
      - Anything else -> help menu listing available topics
    """
    msg_lower = message.lower()
    results = cache.get("results", {})
    pairwise = results.get("pairwise", {})
    chain = results.get("chain", {})

    # Route: summary keywords
    if any(w in msg_lower for w in ["summary", "overview", "tell me about"]):
        return _build_pipeline_context(cache)


    # Route: risk keywords -- also parses a requested count from the message
    if any(w in msg_lower for w in ["risk", "concern", "dangerous", "critical", "worst"]):
        # Parse requested count: "top 5 risks", "show me 10 worst", etc.
        n = 20  # default
        m = re.search(r'(?:top|show|list|give)\s*(?:me\s*)?(\d+)', msg_lower)
        if not m:
            m = re.search(r'(\d+)\s*(?:top|worst|biggest|highest|critical|risk|concern)', msg_lower)
        if m:
            n = max(1, min(100, int(m.group(1))))
        return _build_top_concerns_context(cache, n=n)

    # Route: growth / trajectory keywords

    if any(w in msg_lower for w in ["growth", "trajectory", "trend", "predict"]):
        return _build_trajectory_context(cache) or "No multi-run trajectory data available."

    # Route: match / alignment keywords

    if any(w in msg_lower for w in ["match", "align", "pair"]):
        parts = []
        for (ye, yl), mr in pairwise.items():
            matches = mr.get("matches", pd.DataFrame())
            parts.append(f"{ye}-{yl}: {len(matches)} matches found")

    # Default: show help menu with available query topics
        return "\n".join(parts) if parts else "No matching data available."

    return (
        "I can answer questions about this pipeline's inspection data. "
        "Try asking about:\n"
        "- Pipeline summary / overview\n"
        "- Top risk concerns\n"
        "- Growth trends and predictions\n"
        "- Match results between runs\n\n"
        "Note: For AI-powered responses, set the XAI_API_KEY environment variable."
    )


def _fallback_report(cache: dict) -> str:
    """Generate a basic template report without LLM, using raw context data."""
    context = _build_pipeline_context(cache)
    concerns = _build_top_concerns_context(cache)
    return f"""# CorroSight - Executive Summary

{context}

{concerns}

---
*Note: For AI-enhanced narrative, set the XAI_API_KEY environment variable.*
"""


def _fallback_narratives(anomaly_data: list[dict]) -> list[dict]:
    """Generate template-based narratives without LLM, using anomaly fields directly."""
    for d in anomaly_data:
        growth = d.get("growth_rate")
        depth = d.get("depth_pct")
        remaining = d.get("remaining_life_years")
        risk = d.get("risk_category", "Unknown")

        narrative = f"This {d.get('event_type', 'anomaly')} at joint {d.get('joint', '?')} "
        narrative += f"has a current depth of {depth}% wall loss "
        if growth is not None:
            narrative += f"and is growing at {growth:.2f}%/yr. "
        if remaining is not None and remaining < 20:
            narrative += f"Estimated remaining life is {remaining:.0f} years. "
        narrative += f"Risk classification: {risk}."

        if d.get("depth_2007") is not None:
            narrative += f" Tracked since 2007 ({d['depth_2007']}% -> {d.get('depth_2015', '?')}% -> {depth}%)."
        if d.get("is_accelerating"):
            narrative += " Growth appears to be ACCELERATING."

        d["narrative"] = narrative
    return anomaly_data


def _fallback_insights(chart_type: str, data: dict, cache: dict) -> list[dict]:
    """Return a placeholder annotation directing the user to configure an API key."""
    return [
        {"text": "AI insights require XAI_API_KEY", "x": 0.5, "y": 0.5,
         "xref": "paper", "yref": "paper", "showarrow": False,
         "font": {"color": "gray", "size": 12}}
    ]


def _safe_val(v):
    """
    Convert numpy scalar types to native Python types for JSON serialization.

    Handles np.integer -> int, np.floating -> float (with NaN -> None),
    and np.bool_ -> bool. Passes through all other types unchanged.
    """
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if np.isnan(f) else round(f, 2)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v
