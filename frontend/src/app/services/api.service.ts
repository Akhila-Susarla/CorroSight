import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

const BASE = 'http://localhost:8000/api';

/**
 * Central HTTP client service for the CorroSight frontend.
 *
 * Communicates with the FastAPI backend at localhost:8000. All REST calls
 * are routed through this single service so that endpoint URLs, payload
 * shapes, and HTTP methods are defined in one place.
 */
@Injectable({ providedIn: 'root' })
export class ApiService {
  constructor(private http: HttpClient) {}

  // ── Core Data Endpoints ─────────────────────────────────────────────────

  /** Fetches run metadata and anomaly/girth weld counts per year. GET /api/summary */
  getSummary(): Observable<any> {
    return this.http.get(`${BASE}/summary`);
  }

  /** Fetches data-quality metrics (completeness, validity, outliers) for each uploaded run. GET /api/quality */
  getQuality(): Observable<any> {
    return this.http.get(`${BASE}/quality`);
  }

  /** Fetches exploratory data analysis distributions and statistics. GET /api/eda */
  getEda(): Observable<any> {
    return this.http.get(`${BASE}/eda`);
  }

  /** Fetches alignment results showing how ILI runs were spatially matched. GET /api/alignment */
  getAlignment(): Observable<any> {
    return this.http.get(`${BASE}/alignment`);
  }

  /** Fetches matched anomaly pairs for a specific run-pair identifier. GET /api/matches/:pair */
  getMatches(pair: string): Observable<any> {
    return this.http.get(`${BASE}/matches/${pair}`);
  }

  /** Fetches corrosion growth rates and depth-change data for a run pair. GET /api/growth/:pair */
  getGrowth(pair: string): Observable<any> {
    return this.http.get(`${BASE}/growth/${pair}`);
  }

  /** Fetches multi-run trend analysis across all uploaded ILI inspections. GET /api/multirun */
  getMultirun(): Observable<any> {
    return this.http.get(`${BASE}/multirun`);
  }

  /** Fetches the pipeline schematic view with anomaly overlay positions. GET /api/pipeline-view */
  getPipelineView(): Observable<any> {
    return this.http.get(`${BASE}/pipeline-view`);
  }

  /** Opens a new browser tab to download the full analysis as an Excel workbook. GET /api/export */
  exportExcel(): void {
    window.open(`${BASE}/export`, '_blank');
  }

  // ── AI-Powered Endpoints ──────────────────────────────────────────────────

  /** Sends a user message and conversation history to the AI chat assistant. POST /api/chat */
  chat(message: string, history: { role: string; content: string }[]): Observable<any> {
    return this.http.post(`${BASE}/chat`, { message, history });
  }

  /** Generates a comprehensive AI-written integrity report for the current dataset. GET /api/ai-report */
  getAiReport(): Observable<any> {
    return this.http.get(`${BASE}/ai-report`);
  }

  /** Fetches AI-generated plain-language narratives summarising each analysis section. GET /api/ai-narratives */
  getAiNarratives(): Observable<any> {
    return this.http.get(`${BASE}/ai-narratives`);
  }

  /** Requests AI-driven insights for a specific chart type and its underlying data. POST /api/ai-insights */
  getAiInsights(chartType: string, data: any): Observable<any> {
    return this.http.post(`${BASE}/ai-insights`, { chart_type: chartType, data });
  }

  /** Converts a natural-language query into a chart specification and returns rendered data. POST /api/nl-chart */
  getNlChart(query: string): Observable<any> {
    return this.http.post(`${BASE}/nl-chart`, { query });
  }

  /** Fetches virtual ILI predictions for a given projection year. GET /api/virtual-ili/:year */
  getVirtualIli(year: number): Observable<any> {
    return this.http.get(`${BASE}/virtual-ili/${year}`);
  }

  /** Fetches the aggregated integrity dashboard with risk rankings and repair recommendations. GET /api/integrity-dashboard */
  getIntegrityDashboard(): Observable<any> {
    return this.http.get(`${BASE}/integrity-dashboard`);
  }

  // ── Configuration Endpoints ───────────────────────────────────────────────

  /** Stores the user-supplied OpenAI API key on the backend for AI features. POST /api/set-api-key */
  setApiKey(key: string): Observable<any> {
    return this.http.post(`${BASE}/set-api-key`, { key });
  }

  /** Checks whether an API key is configured and AI features are available. GET /api/ai-status */
  getAiStatus(): Observable<any> {
    return this.http.get(`${BASE}/ai-status`);
  }

  /** Uploads an ILI data file and triggers the full analysis pipeline. POST /api/run-pipeline */
  runPipeline(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post(`${BASE}/run-pipeline`, formData);
  }

  /** Polls the current pipeline execution status (running, completed, or failed). GET /api/run-pipeline/status */
  getPipelineStatus(): Observable<any> {
    return this.http.get(`${BASE}/run-pipeline/status`);
  }
}
