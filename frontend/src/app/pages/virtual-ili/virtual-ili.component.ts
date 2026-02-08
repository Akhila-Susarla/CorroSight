import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { PlotlyModule } from 'angular-plotly.js';
import { MetricCardComponent } from '../../components/metric-card/metric-card.component';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-virtual-ili',
  standalone: true,
  imports: [CommonModule, FormsModule, PlotlyModule, MetricCardComponent],
  template: `
    <div class="page-container">
      <div class="page-header">
        <h1 class="page-title">Virtual ILI - Future Inspection Predictor</h1>
        <p class="page-subtitle">Simulate what a future inline inspection would find based on historical corrosion growth trends</p>
      </div>

      <!-- Year selector -->
      <div class="year-selector">
        <label>Predict inspection for year:</label>
        <input type="range" min="2025" max="2040" step="1" [(ngModel)]="targetYear" (ngModelChange)="loadPrediction()" />
        <span class="year-display">{{targetYear}}</span>
        <div class="year-labels">
          <span>2025</span><span>2030</span><span>2035</span><span>2040</span>
        </div>
      </div>

      <!-- Loading -->
      <div *ngIf="loading" class="loading">Generating predictions...</div>

      <!-- Summary cards -->
      <div class="cards-row" *ngIf="summary">
        <app-metric-card label="Total Predicted" [value]="summary.total_predicted" color="#1a237e"></app-metric-card>
        <app-metric-card label="Critical" [value]="summary.critical_count" subtitle="Depth > 70%" color="#e74c3c"></app-metric-card>
        <app-metric-card label="High Risk" [value]="summary.high_count" subtitle="Depth 50-70%" color="#f39c12"></app-metric-card>
        <app-metric-card label="Need Repair by {{targetYear}}" [value]="summary.needing_repair_by_target" subtitle="Depth > 80%" color="#c62828"></app-metric-card>
      </div>

      <!-- Charts row -->
      <div class="chart-row">
        <div class="chart-card" *ngIf="riskPieChart.data.length">
          <h3>Predicted Risk Distribution in {{targetYear}}</h3>
          <plotly-plot [data]="riskPieChart.data" [layout]="riskPieChart.layout" [config]="{responsive:true}"></plotly-plot>
        </div>
        <div class="chart-card" *ngIf="depthDistChart.data.length">
          <h3>Predicted Depth Distribution</h3>
          <plotly-plot [data]="depthDistChart.data" [layout]="depthDistChart.layout" [config]="{responsive:true}"></plotly-plot>
        </div>
      </div>

      <!-- Threshold crossings -->
      <div class="chart-card" *ngIf="thresholdChart.data.length">
        <h3>New Threshold Crossings by {{targetYear}}</h3>
        <p class="chart-desc">Anomalies that will cross critical depth thresholds between 2022 and {{targetYear}}</p>
        <plotly-plot [data]="thresholdChart.data" [layout]="thresholdChart.layout" [config]="{responsive:true}"></plotly-plot>
      </div>

      <!-- Predicted pipeline scatter -->
      <div class="chart-card" *ngIf="pipelineChart.data.length">
        <h3>Predicted Pipeline Condition in {{targetYear}}</h3>
        <p class="chart-desc">Anomaly depths projected forward — size and color indicate predicted severity</p>
        <plotly-plot [data]="pipelineChart.data" [layout]="pipelineChart.layout" [config]="{responsive:true}"></plotly-plot>
      </div>

      <!-- Top concerns table -->
      <div class="table-container" *ngIf="topConcerns.length">
        <h3 class="section-title">Top 20 Predicted High-Risk Anomalies in {{targetYear}}</h3>
        <table class="full-width-table">
          <thead>
            <tr>
              <th>Joint</th><th>Distance (ft)</th><th>Clock</th>
              <th>2022 Depth %</th><th>Growth %/yr</th>
              <th>Predicted {{targetYear}} Depth %</th><th>Predicted Risk</th>
              <th>Years to 80%</th><th>3-Run Tracked</th>
            </tr>
          </thead>
          <tbody>
            <tr *ngFor="let c of topConcerns" [class.danger-row]="c.predicted_risk === 'Critical'">
              <td>{{c.joint}}</td>
              <td>{{c.distance_ft?.toFixed(0)}}</td>
              <td>{{c.clock?.toFixed(1)}}</td>
              <td>{{c.current_depth_2022}}</td>
              <td>{{c.growth_rate?.toFixed(3)}}</td>
              <td class="predicted-depth" [class.critical]="c.predicted_depth >= 70" [class.high]="c.predicted_depth >= 50 && c.predicted_depth < 70">
                {{c.predicted_depth}}%
              </td>
              <td><span class="risk-badge" [class]="c.predicted_risk">{{c.predicted_risk}}</span></td>
              <td>{{c.years_to_80pct != null ? c.years_to_80pct.toFixed(1) : 'N/A'}}</td>
              <td>{{c.is_triple_tracked ? 'Yes' : 'No'}}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- AI narrative (if available) -->
      <div class="ai-narrative-card" *ngIf="aiNarrative">
        <div class="ai-badge">AI Generated</div>
        <h3>AI Prediction Summary</h3>
        <div [innerHTML]="formatMarkdown(aiNarrative)"></div>
      </div>
    </div>
  `,
  styles: [`
    .page-header { margin-bottom: 24px; }
    .page-subtitle { color: #666; font-size: 14px; margin-top: 4px; }

    .year-selector {
      background: white; border-radius: 12px; padding: 24px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12); margin-bottom: 24px;
      text-align: center;
    }
    .year-selector label { font-size: 16px; font-weight: 500; color: #37474f; display: block; margin-bottom: 12px; }
    .year-selector input[type="range"] {
      width: 80%; max-width: 500px; height: 6px; -webkit-appearance: none;
      background: linear-gradient(to right, #4caf50, #ff9800, #f44336);
      border-radius: 3px; outline: none;
    }
    .year-selector input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none; width: 24px; height: 24px;
      background: #1a237e; border-radius: 50%; cursor: pointer;
      box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }
    .year-display {
      display: inline-block; margin-left: 16px; font-size: 32px;
      font-weight: 700; color: #1a237e;
    }
    .year-labels {
      display: flex; justify-content: space-between;
      max-width: 500px; margin: 4px auto 0; font-size: 12px; color: #999;
      width: 80%;
    }

    .loading {
      text-align: center; padding: 40px; font-size: 16px;
      color: #7c4dff; font-weight: 500;
    }

    .chart-desc { font-size: 12px; color: #999; margin: -4px 0 8px; }

    table { border-collapse: collapse; font-size: 13px; }
    th { background: #eceff1; padding: 8px 10px; text-align: right; font-weight: 500; }
    td { padding: 8px 10px; border-bottom: 1px solid #eceff1; text-align: right; }
    .danger-row { background: #fff3e0; }

    .predicted-depth { font-weight: 600; }
    .predicted-depth.critical { color: #c62828; }
    .predicted-depth.high { color: #e65100; }

    .risk-badge { padding: 2px 8px; border-radius: 10px; font-size: 11px; }
    .risk-badge.Critical { background: #ffcdd2; color: #b71c1c; }
    .risk-badge.High { background: #ffe0b2; color: #e65100; }
    .risk-badge.Medium { background: #bbdefb; color: #1565c0; }
    .risk-badge.Low { background: #c8e6c9; color: #2e7d32; }

    .ai-narrative-card {
      background: white; border-radius: 12px; padding: 24px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12); margin-top: 24px;
      border-left: 4px solid #7c4dff; position: relative;
    }
    .ai-badge {
      position: absolute; top: 12px; right: 16px;
      background: linear-gradient(135deg, #7c4dff, #1a237e);
      color: white; padding: 4px 12px; border-radius: 12px;
      font-size: 11px; font-weight: 600;
    }
    .ai-narrative-card h3 { color: #1a237e; margin-bottom: 12px; }
  `]
})
/**
 * Virtual ILI future inspection predictor using growth trend extrapolation.
 * Simulates what a future inline inspection would find at the selected target year.
 */
export class VirtualIliComponent implements OnInit {
  targetYear = 2030;
  loading = false;
  summary: any = null;
  topConcerns: any[] = [];
  aiNarrative: string | null = null;

  riskPieChart: any = { data: [], layout: {} };
  depthDistChart: any = { data: [], layout: {} };
  thresholdChart: any = { data: [], layout: {} };
  pipelineChart: any = { data: [], layout: {} };

  constructor(private api: ApiService) {}

  /** Loads predictions for the default target year on init. */
  ngOnInit(): void {
    this.loadPrediction();
  }

  /** Fetches predictions for the target year and builds all charts (risk pie, depth dist, thresholds, pipeline). */
  loadPrediction(): void {
    this.loading = true;
    this.api.getVirtualIli(this.targetYear).subscribe({
      next: (d: any) => {
        if (d.error) {
          this.loading = false;
          return;
        }
        this.summary = d.summary;
        this.topConcerns = d.top_concerns || [];
        this.buildRiskPie(d.risk_distribution);
        this.buildDepthDist(d.depth_distribution);
        this.buildThresholdChart(d.threshold_crossings);
        this.buildPipelineChart(d.all_predictions);
        this.loading = false;
      },
      error: () => { this.loading = false; }
    });
  }

  /** Builds a donut chart of predicted risk distribution (Critical/High/Medium/Low). */
  buildRiskPie(dist: Record<string, number>): void {
    if (!dist) return;
    const labels = Object.keys(dist);
    const values = Object.values(dist);
    const colors: Record<string, string> = { Critical: '#e74c3c', High: '#f39c12', Medium: '#3498db', Low: '#2ecc71' };
    this.riskPieChart = {
      data: [{
        type: 'pie', labels, values,
        marker: { colors: labels.map(l => colors[l] || '#999') },
        hole: 0.4, textinfo: 'label+value',
      }],
      layout: { height: 300, margin: { t: 10, b: 10, l: 10, r: 10 }, showlegend: false }
    };
  }

  /** Builds a bar chart of predicted depth bins (e.g., 0-20%, 20-40%, etc.). */
  buildDepthDist(dist: Record<string, number>): void {
    if (!dist) return;
    const labels = Object.keys(dist);
    const values = Object.values(dist);
    const colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c'];
    this.depthDistChart = {
      data: [{
        type: 'bar', x: labels, y: values,
        marker: { color: colors },
      }],
      layout: {
        height: 300, margin: { t: 10, b: 50, l: 50, r: 20 },
        xaxis: { title: { text: 'Predicted Depth Range' } }, yaxis: { title: { text: 'Count' } },
      }
    };
  }

  /** Builds a bar chart of new threshold crossings (50/60/70/80%) between 2022 and target year. */
  buildThresholdChart(crossings: Record<string, number>): void {
    if (!crossings) return;
    const labels = ['50% Threshold', '60% Threshold', '70% Threshold', '80% Repair'];
    const keys = ['crossing_50pct', 'crossing_60pct', 'crossing_70pct', 'crossing_80pct'];
    const values = keys.map(k => crossings[k] || 0);
    const colors = ['#3498db', '#f39c12', '#e67e22', '#e74c3c'];
    this.thresholdChart = {
      data: [{
        type: 'bar', x: labels, y: values,
        marker: { color: colors },
        text: values.map(v => v.toString()), textposition: 'outside',
      }],
      layout: {
        height: 280, margin: { t: 10, b: 60, l: 60, r: 20 },
        xaxis: { title: { text: 'Depth Threshold' } },
        yaxis: { title: { text: 'Anomalies Newly Crossing' } },
      }
    };
  }

  /** Builds a scatter plot of predicted anomaly positions colored by predicted risk category. */
  buildPipelineChart(predictions: any[]): void {
    if (!predictions?.length) return;
    const colorMap: Record<string, string> = { Critical: '#e74c3c', High: '#f39c12', Medium: '#3498db', Low: '#2ecc71' };
    const groups: Record<string, any[]> = {};
    predictions.forEach(p => {
      const r = p.predicted_risk || 'Low';
      if (!groups[r]) groups[r] = [];
      groups[r].push(p);
    });
    const traces = Object.entries(groups).map(([risk, pts]) => ({
      type: 'scatter', mode: 'markers', name: risk,
      x: pts.map(p => p.distance_ft),
      y: pts.map(p => p.clock),
      marker: {
        color: colorMap[risk] || '#999',
        size: pts.map(p => Math.max(4, (p.predicted_depth || 0) / 5)),
        opacity: 0.6,
      },
      hovertemplate: 'Joint %{customdata[0]}<br>Dist: %{x:.0f} ft<br>Clock: %{y:.1f}<br>Predicted Depth: %{customdata[1]}%<extra>' + risk + '</extra>',
      customdata: pts.map(p => [p.joint, p.predicted_depth]),
    }));

    this.pipelineChart = {
      data: traces as any[],
      layout: {
        height: 350, margin: { t: 10, b: 50, l: 50, r: 20 },
        xaxis: { title: { text: 'Distance (ft)' } },
        yaxis: { title: { text: 'Clock Position (hr)' }, range: [0, 12] },
        shapes: [{
          type: 'rect', x0: 0, x1: 60000, y0: 0, y1: 12,
          fillcolor: 'rgba(0,0,0,0)', line: { width: 0 },
        }],
      }
    };
  }

  /** Converts markdown bold/italic to HTML for AI narrative rendering. */
  formatMarkdown(text: string): string {
    if (!text) return '';
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/\n/g, '<br>');
  }
}
