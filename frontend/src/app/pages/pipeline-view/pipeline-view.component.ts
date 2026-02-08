import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { PlotlyModule } from 'angular-plotly.js';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-pipeline-view',
  standalone: true,
  imports: [CommonModule, FormsModule, PlotlyModule],
  template: `
    <div class="page-container">
      <h1 class="page-title">Pipeline Schematic View</h1>

      <!-- Explanation card -->
      <div class="explanation-card">
        <h3>How to Read This Chart</h3>
        <p>This is an <strong>"unrolled pipe"</strong> visualization. Imagine slicing the pipe lengthwise and laying it flat:</p>
        <ul>
          <li><strong>X-axis (Pipeline Distance)</strong> — Distance in feet from the start of the pipeline. Each vertical dotted line marks a <em>girth weld</em> (a joint between two pipe segments).</li>
          <li><strong>Y-axis (Clock Position)</strong> — Position around the pipe circumference, expressed as a clock face (12:00 = top of pipe, 6:00 = bottom). Think of looking down the pipe from one end.</li>
          <li><strong>Dot Size</strong> — Proportional to the anomaly's depth (% of wall loss). Larger dots = deeper corrosion.</li>
          <li><strong>Dot Color</strong> — Each ILI inspection run has a distinct color: <span style="color:#636EFA;font-weight:600;">2007</span>, <span style="color:#EF553B;font-weight:600;">2015</span>, <span style="color:#00CC96;font-weight:600;">2022</span>.</li>
          <li><strong>Connecting Lines</strong> — Lines between dots show matched anomalies across runs. Color indicates match confidence: <span style="color:#2ecc71;font-weight:600;">HIGH</span>, <span style="color:#f39c12;font-weight:600;">MEDIUM</span>, <span style="color:#e74c3c;font-weight:600;">LOW</span>.</li>
        </ul>
        <p style="margin-top:8px;color:#546e7a;font-size:13px;">Use the sliders below to zoom into a specific pipeline segment. Toggle match connections on/off for clarity.</p>
      </div>

      <div class="controls">
        <label>Distance Range: {{distMin | number:'1.0-0'}} - {{distMax | number:'1.0-0'}} ft</label>
        <input type="range" min="0" max="58000" step="1000" [(ngModel)]="distMin" (ngModelChange)="buildChart()">
        <input type="range" min="0" max="58000" step="1000" [(ngModel)]="distMax" (ngModelChange)="buildChart()">
        <label>
          <input type="checkbox" [(ngModel)]="showConnections" (ngModelChange)="buildChart()"> Show match connections
        </label>
        <button class="ai-summary-btn" (click)="generateSummary()" [disabled]="summaryLoading">
          <span *ngIf="!summaryLoading">AI Summary</span>
          <span *ngIf="summaryLoading">Analyzing...</span>
        </button>
      </div>

      <!-- AI Summary -->
      <div class="ai-summary-card" *ngIf="aiSummary">
        <div class="ai-badge">AI Pipeline Segment Analysis</div>
        <button class="close-btn" (click)="aiSummary = null">&times;</button>
        <div class="summary-content" [innerHTML]="formatMarkdown(aiSummary)"></div>
      </div>

      <div class="chart-card" *ngIf="pipeChart.data.length">
        <plotly-plot [data]="pipeChart.data" [layout]="pipeChart.layout" [config]="{responsive:true}"></plotly-plot>
      </div>
    </div>
  `,
  styles: [`
    .explanation-card {
      background: white; border-radius: 10px; padding: 20px 24px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.10); margin-bottom: 16px;
      border-left: 4px solid #1a237e;
    }
    .explanation-card h3 { color: #1a237e; font-size: 15px; margin-bottom: 10px; }
    .explanation-card p { color: #444; font-size: 13px; line-height: 1.6; margin: 0 0 6px; }
    .explanation-card ul { padding-left: 20px !important; margin: 8px 0 !important; }
    .explanation-card li {
      display: list-item !important; list-style-type: disc !important;
      font-size: 13px; color: #444; line-height: 1.6; margin: 4px 0;
    }
    .controls {
      display: flex; gap: 16px; align-items: center; margin-bottom: 16px; flex-wrap: wrap;
      background: white; padding: 12px 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .controls label { font-size: 13px; color: #546e7a; }
    .controls input[type=range] { width: 200px; }
    .controls input[type=checkbox] { margin-right: 4px; }
    .ai-summary-btn {
      margin-left: auto; padding: 8px 16px;
      background: linear-gradient(135deg, #7c4dff, #1a237e); color: white;
      border: none; border-radius: 8px; font-size: 13px;
      cursor: pointer; font-weight: 500; transition: all 0.2s;
    }
    .ai-summary-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(124,77,255,0.4); }
    .ai-summary-btn:disabled { background: #ccc; cursor: default; transform: none; box-shadow: none; }
    .ai-summary-card {
      background: white; border-radius: 10px; padding: 20px 24px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15); margin-bottom: 16px;
      border-left: 4px solid #7c4dff; position: relative;
    }
    .ai-badge {
      background: linear-gradient(135deg, #7c4dff, #e040fb);
      color: white; padding: 4px 12px; border-radius: 12px;
      font-size: 11px; font-weight: 600; display: inline-block; margin-bottom: 12px;
    }
    .close-btn {
      position: absolute; top: 12px; right: 16px;
      background: none; border: none; font-size: 22px;
      color: #999; cursor: pointer;
    }
    .summary-content { font-size: 14px; line-height: 1.7; color: #333; }
  `]
})
/**
 * "Unrolled pipe" schematic visualization (distance x clock position).
 * Shows anomaly positions per run, match connection lines colored by confidence, and girth weld markers.
 */
export class PipelineViewComponent implements OnInit {
  rawData: any = null;
  distMin = 0;
  distMax = 58000;
  showConnections = true;
  pipeChart: any = { data: [], layout: {} };
  aiSummary: string | null = null;
  summaryLoading = false;

  constructor(private api: ApiService) {}

  /** Fetches pipeline view data (anomalies, connections, girth welds) on init. */
  ngOnInit(): void {
    this.api.getPipelineView().subscribe(d => {
      this.rawData = d;
      this.buildChart();
    });
  }

  /** Sends the current distance range to the AI copilot for a contextual segment summary. */
  generateSummary(): void {
    this.summaryLoading = true;
    const prompt = `Analyze the pipeline schematic view for the segment from ${this.distMin} ft to ${this.distMax} ft. ` +
      `Describe the anomaly density, any clusters of corrosion, clock position patterns, and highlight areas of concern. ` +
      `Mention how the 3 inspection runs (2007, 2015, 2022) compare in this segment.`;
    this.api.chat(prompt, []).subscribe({
      next: (res: any) => {
        this.aiSummary = res.response;
        this.summaryLoading = false;
      },
      error: () => {
        this.aiSummary = 'Failed to generate summary. Check backend connection.';
        this.summaryLoading = false;
      }
    });
  }
  /** Converts markdown text to HTML for the AI summary display. */

  formatMarkdown(text: string): string {
    if (!text) return '';
    return text
      .replace(/### (.*?)(\n|$)/g, '<h4 style="color:#1a237e;margin:12px 0 6px;">$1</h4>')
      .replace(/## (.*?)(\n|$)/g, '<h3 style="color:#1a237e;margin:16px 0 8px;">$1</h3>')
      .replace(/# (.*?)(\n|$)/g, '<h2 style="color:#1a237e;margin:20px 0 10px;">$1</h2>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/- (.*?)(\n|$)/g, '<li style="margin-left:20px;">$1</li>')
      .replace(/\n/g, '<br>');
  }

  /** Builds scatter traces per run year, match connection lines colored by confidence, and girth weld vertical markers. */
  buildChart(): void {
    if (!this.rawData) return;
    const traces: any[] = [];
    const colors: Record<string, string> = { '2007': '#636EFA', '2015': '#EF553B', '2022': '#00CC96' };

    // Anomaly scatter per run
    for (const [year, anomalies] of Object.entries(this.rawData.anomalies)) {
      const filtered = (anomalies as any[]).filter(
        a => a.corrected_distance >= this.distMin && a.corrected_distance <= this.distMax
      );
      if (!filtered.length) continue;

      traces.push({
        type: 'scatter', mode: 'markers', name: `Run ${year}`,
        x: filtered.map(a => a.corrected_distance),
        y: filtered.map(a => a.clock_hours),
        marker: {
          size: filtered.map(a => Math.max(3, Math.min(14, (a.depth_pct || 10) / 3))),
          color: colors[year] || '#999', opacity: 0.6,
        },
        customdata: filtered.map(a => [a.depth_pct, a.event_type, a.joint_number]),
        hovertemplate: `Run ${year}<br>Dist: %{x:.0f} ft<br>Clock: %{y:.1f}<br>Depth: %{customdata[0]}%<br>Type: %{customdata[1]}<br>Joint: %{customdata[2]}<extra></extra>`,
      });
    }

    // Match connections
    if (this.showConnections) {
      const connFiltered = this.rawData.connections.filter(
        (c: any) => c.later_dist >= this.distMin && c.later_dist <= this.distMax
      );
      // Group by confidence for coloring
      const confColors: Record<string, string> = { HIGH: '#2ecc71', MEDIUM: '#f39c12', LOW: '#e74c3c' };
      for (const conf of ['HIGH', 'MEDIUM', 'LOW']) {
        const subset = connFiltered.filter((c: any) => c.confidence === conf);
        if (!subset.length) continue;
        const x: (number | null)[] = [];
        const y: (number | null)[] = [];
        subset.forEach((c: any) => {
          x.push(c.earlier_dist, c.later_dist, null);
          y.push(c.earlier_clock, c.later_clock, null);
        });
        traces.push({
          type: 'scatter', mode: 'lines', name: `${conf} match`,
          x, y, line: { color: confColors[conf], width: 0.5 },
          opacity: 0.4, showlegend: true,
        });
      }
    }

    // Girth weld vertical lines (sparse)
    const gwFiltered = this.rawData.girth_welds.filter(
      (d: number) => d >= this.distMin && d <= this.distMax
    );
    const step = Math.max(1, Math.floor(gwFiltered.length / 50));
    const shapes = gwFiltered.filter((_: any, i: number) => i % step === 0).map((d: number) => ({
      type: 'line', x0: d, x1: d, y0: 0, y1: 12,
      line: { color: '#e0e0e0', width: 0.5, dash: 'dot' },
    }));

    this.pipeChart = {
      data: traces,
      layout: {
        height: 550, margin: { t: 10, b: 50, l: 60, r: 20 },
        xaxis: { title: { text: 'Pipeline Distance (ft)' }, range: [this.distMin, this.distMax] },
        yaxis: {
          title: { text: 'Clock Position' },
          range: [0, 12],
          tickvals: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
          ticktext: ['12:00', '1:00', '2:00', '3:00', '4:00', '5:00', '6:00', '7:00', '8:00', '9:00', '10:00', '11:00', '12:00'],
        },
        shapes,
        legend: { orientation: 'h', y: 1.05 },
      }
    };
  }
}
