import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { PlotlyModule } from 'angular-plotly.js';
import { MetricCardComponent } from '../../components/metric-card/metric-card.component';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-growth',
  standalone: true,
  imports: [CommonModule, FormsModule, PlotlyModule, MetricCardComponent],
  template: `
    <div class="page-container">
      <h1 class="page-title">Growth Analysis</h1>

      <!-- Pair selector -->
      <div class="pair-selector">
        <label>Run Pair: </label>
        <select [(ngModel)]="selectedPair" (ngModelChange)="loadPair()">
          <option value="2007-2015">2007 vs 2015</option>
          <option value="2015-2022">2015 vs 2022</option>
        </select>

        <!-- AI Report button -->
        <button class="ai-report-btn" (click)="generateReport()" [disabled]="reportLoading">
          <span *ngIf="!reportLoading">Generate AI Report</span>
          <span *ngIf="reportLoading">Generating...</span>
        </button>
      </div>

      <!-- AI Executive Report -->
      <div class="ai-report-card" *ngIf="aiReport">
        <div class="ai-badge">AI Generated Executive Report</div>
        <button class="close-btn" (click)="aiReport = null">&times;</button>
        <div class="report-content" [innerHTML]="formatMarkdown(aiReport)"></div>
      </div>

      <!-- Stats cards -->
      <div class="cards-row" *ngIf="growthStats">
        <app-metric-card label="Mean Growth Rate" [value]="growthStats.mean_rate + ' %/yr'" color="#1a237e"></app-metric-card>
        <app-metric-card label="Median Growth Rate" [value]="growthStats.median_rate + ' %/yr'" color="#0d47a1"></app-metric-card>
        <app-metric-card label="Negative Growth" [value]="growthStats.pct_negative + '%'" subtitle="Apparent shrinkage" color="#f44336"></app-metric-card>
        <app-metric-card label="Severe Growth" [value]="growthStats.pct_severe + '%'" subtitle="> 5 %/yr" color="#e91e63"></app-metric-card>
      </div>

      <!-- Charts -->
      <div class="chart-row">
        <div class="chart-card" *ngIf="rateHistChart.data.length">
          <h3>Depth Growth Rate Distribution</h3>
          <plotly-plot [data]="rateHistChart.data" [layout]="rateHistChart.layout" [config]="{responsive:true}"></plotly-plot>
        </div>
        <div class="chart-card" *ngIf="riskMatrixChart.data.length">
          <h3>Risk Matrix: Depth vs Growth Rate</h3>
          <plotly-plot [data]="riskMatrixChart.data" [layout]="riskMatrixChart.layout" [config]="{responsive:true}"></plotly-plot>
        </div>
      </div>

      <!-- Remaining life -->
      <div class="chart-card" *ngIf="remainingLifeChart.data.length">
        <h3>Estimated Remaining Life Distribution</h3>
        <plotly-plot [data]="remainingLifeChart.data" [layout]="remainingLifeChart.layout" [config]="{responsive:true}"></plotly-plot>
      </div>

      <!-- Top concerns table -->
      <div class="table-container" *ngIf="topConcerns.length">
        <h3 class="section-title">Top 20 Highest-Risk Anomalies</h3>
        <table class="full-width-table">
          <thead>
            <tr>
              <th>Joint</th><th>Distance (ft)</th><th>Clock</th><th>Depth %</th>
              <th>Growth %/yr</th><th>Remaining Life (yr)</th><th>Risk Score</th><th>Risk</th>
            </tr>
          </thead>
          <tbody>
            <tr *ngFor="let c of topConcerns">
              <td>{{c.later_joint}}</td><td>{{c.later_distance?.toFixed(0)}}</td>
              <td>{{c.later_clock?.toFixed(1)}}</td><td>{{c.later_depth_pct}}</td>
              <td>{{c.depth_growth_rate?.toFixed(2)}}</td><td>{{c.remaining_life_years?.toFixed(1)}}</td>
              <td>{{c.risk_score?.toFixed(0)}}</td>
              <td><span class="risk-badge" [class]="c.risk_category">{{c.risk_category}}</span></td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- AI Narrative Cards -->
      <div *ngIf="narratives.length" style="margin-top: 32px;">
        <h2 class="section-title">
          AI Anomaly Intelligence Cards
          <span class="ai-tag">AI</span>
        </h2>
        <div class="narrative-grid">
          <div class="narrative-card" *ngFor="let n of narratives" [class]="'risk-border-' + n.risk_category">
            <div class="narrative-header">
              <span class="narrative-joint">Joint {{n.joint}}</span>
              <span class="risk-badge" [class]="n.risk_category">{{n.risk_category}}</span>
            </div>
            <div class="narrative-stats">
              <span>Depth: {{n.depth_pct}}%</span>
              <span>Rate: {{n.growth_rate?.toFixed(2)}} %/yr</span>
              <span *ngIf="n.remaining_life_years != null">Life: {{n.remaining_life_years?.toFixed(0)}} yr</span>
            </div>
            <p class="narrative-text">{{n.narrative}}</p>
            <div class="narrative-footer" *ngIf="n.depth_2007 != null">
              <small>History: {{n.depth_2007}}% (2007) &#8594; {{n.depth_2015}}% (2015) &#8594; {{n.depth_pct}}% (2022)</small>
            </div>
          </div>
        </div>
      </div>

      <!-- Load narratives button -->
      <div *ngIf="!narratives.length && !narrativesLoading" style="margin-top: 24px; text-align: center;">
        <button class="load-narratives-btn" (click)="loadNarratives()">Load AI Anomaly Stories</button>
      </div>
      <div *ngIf="narrativesLoading" style="text-align: center; margin-top: 24px; color: #7c4dff;">
        Generating AI narratives...
      </div>

      <!-- Multi-run section -->
      <div *ngIf="multirunData" style="margin-top: 32px;">
        <h2 class="section-title">Multi-Run Analysis (3 Inspections)</h2>

        <!-- Lifecycle summary -->
        <div class="chart-card" *ngIf="lifecycleChart.data.length">
          <h3>Anomaly Lifecycle Summary</h3>
          <plotly-plot [data]="lifecycleChart.data" [layout]="lifecycleChart.layout" [config]="{responsive:true}"></plotly-plot>
        </div>

        <!-- Growth trajectories -->
        <div class="chart-card" *ngIf="trajectoryChart.data.length">
          <h3>Growth Trajectories - Top 10 Fastest Growing (with 2030 Prediction)</h3>
          <plotly-plot [data]="trajectoryChart.data" [layout]="trajectoryChart.layout" [config]="{responsive:true}"></plotly-plot>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .pair-selector { margin-bottom: 20px; display: flex; align-items: center; gap: 12px; }
    .pair-selector select { padding: 8px 12px; border-radius: 4px; border: 1px solid #ccc; font-size: 14px; }
    .ai-report-btn {
      margin-left: auto; padding: 10px 20px;
      background: linear-gradient(135deg, #7c4dff, #1a237e); color: white;
      border: none; border-radius: 8px; font-size: 14px;
      cursor: pointer; font-weight: 500; transition: all 0.2s;
    }
    .ai-report-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(124,77,255,0.4); }
    .ai-report-btn:disabled { background: #ccc; cursor: default; transform: none; box-shadow: none; }

    .ai-report-card {
      background: white; border-radius: 12px; padding: 24px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15); margin-bottom: 24px;
      border-left: 4px solid #7c4dff; position: relative;
    }
    .ai-badge {
      background: linear-gradient(135deg, #7c4dff, #e040fb);
      color: white; padding: 4px 12px; border-radius: 12px;
      font-size: 11px; font-weight: 600; display: inline-block; margin-bottom: 16px;
    }
    .close-btn {
      position: absolute; top: 12px; right: 16px;
      background: none; border: none; font-size: 24px;
      color: #999; cursor: pointer;
    }
    .report-content { font-size: 14px; line-height: 1.7; color: #333; }

    table { border-collapse: collapse; font-size: 13px; }
    th { background: #eceff1; padding: 8px 10px; text-align: right; font-weight: 500; }
    td { padding: 8px 10px; border-bottom: 1px solid #eceff1; text-align: right; }
    .risk-badge { padding: 2px 8px; border-radius: 10px; font-size: 11px; }
    .risk-badge.Critical { background: #ffcdd2; color: #b71c1c; }
    .risk-badge.High { background: #ffe0b2; color: #e65100; }
    .risk-badge.Medium { background: #bbdefb; color: #1565c0; }
    .risk-badge.Low { background: #c8e6c9; color: #2e7d32; }

    .ai-tag {
      background: linear-gradient(135deg, #7c4dff, #e040fb);
      color: white; font-size: 10px; font-weight: 700;
      padding: 2px 8px; border-radius: 8px; margin-left: 8px; vertical-align: middle;
    }

    .narrative-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
      gap: 16px;
    }
    .narrative-card {
      background: white; border-radius: 10px; padding: 16px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.1); border-left: 4px solid #ccc;
    }
    .risk-border-Critical { border-left-color: #e74c3c; }
    .risk-border-High { border-left-color: #f39c12; }
    .risk-border-Medium { border-left-color: #3498db; }
    .risk-border-Low { border-left-color: #2ecc71; }

    .narrative-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .narrative-joint { font-weight: 600; font-size: 15px; color: #1a237e; }
    .narrative-stats { display: flex; gap: 12px; font-size: 12px; color: #666; margin-bottom: 8px; }
    .narrative-text { font-size: 13px; line-height: 1.6; color: #444; margin: 0; }
    .narrative-footer { margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee; }
    .narrative-footer small { color: #888; font-size: 11px; }

    .load-narratives-btn {
      padding: 12px 24px; background: #e8eaf6; color: #1a237e;
      border: 2px solid #c5cae9; border-radius: 8px; font-size: 14px;
      cursor: pointer; font-weight: 500; transition: all 0.2s;
    }
    .load-narratives-btn:hover { background: #c5cae9; }
  `]
})
/**
 * Growth analysis page with rate histograms, risk matrix, remaining life,
 * top concerns, AI report, narrative cards, and multi-run trajectories.
 */
export class GrowthComponent implements OnInit {
  selectedPair = '2015-2022';
  growthStats: any = null;
  topConcerns: any[] = [];
  multirunData: any = null;

  aiReport: string | null = null;
  reportLoading = false;
  narratives: any[] = [];
  narrativesLoading = false;

  rateHistChart: any = { data: [], layout: {} };
  riskMatrixChart: any = { data: [], layout: {} };
  remainingLifeChart: any = { data: [], layout: {} };
  lifecycleChart: any = { data: [], layout: {} };
  trajectoryChart: any = { data: [], layout: {} };

  constructor(private api: ApiService) {}

  /** Loads pair-specific growth data and multi-run data on init. */
  ngOnInit(): void {
    this.loadPair();
    this.loadMultirun();
  }

  /** Fetches growth stats, risk points, and remaining life for the selected run pair. */
  loadPair(): void {
    this.api.getGrowth(this.selectedPair).subscribe(d => {
      this.growthStats = d.stats;
      this.topConcerns = d.top_concerns || [];
      this.buildRateHist(d.growth_rates);
      this.buildRiskMatrix(d.risk_points);
      this.buildRemainingLife(d.remaining_life);
    });
  }

  /** Fetches multi-run (3 inspection) lifecycle and trajectory data. */
  loadMultirun(): void {
    this.api.getMultirun().subscribe(d => {
      this.multirunData = d;
      this.buildLifecycleChart(d.lifecycle);
      this.buildTrajectoryChart(d.trajectories);
    });
  }

  /** Requests an AI-generated executive report from the backend. */
  generateReport(): void {
    this.reportLoading = true;
    this.api.getAiReport().subscribe({
      next: (res: any) => {
        this.aiReport = res.report;
        this.reportLoading = false;
      },
      error: () => {
        this.aiReport = 'Failed to generate report. Please check the backend.';
        this.reportLoading = false;
      }
    });
  }

  /** Loads AI-generated per-anomaly narrative intelligence cards. */
  loadNarratives(): void {
    this.narrativesLoading = true;
    this.api.getAiNarratives().subscribe({
      next: (res: any) => {
        this.narratives = res.narratives || [];
        this.narrativesLoading = false;
      },
      error: () => {
        this.narrativesLoading = false;
      }
    });
  }

  /** Converts markdown text to HTML for the AI report display. */
  formatMarkdown(text: string): string {
    if (!text) return '';
    return text
      .replace(/### (.*?)(\n|$)/g, '<h4 style="color:#1a237e;margin:16px 0 8px;">$1</h4>')
      .replace(/## (.*?)(\n|$)/g, '<h3 style="color:#1a237e;margin:20px 0 10px;">$1</h3>')
      .replace(/# (.*?)(\n|$)/g, '<h2 style="color:#1a237e;margin:24px 0 12px;">$1</h2>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/- (.*?)(\n|$)/g, '<li style="margin-left:20px;">$1</li>')
      .replace(/\n/g, '<br>');
  }

  /** Builds a growth rate histogram with a severe threshold marker at 5%/yr. */
  buildRateHist(rates: number[]): void {
    this.rateHistChart = {
      data: [{ type: 'histogram', x: rates, nbinsx: 50, marker: { color: '#636EFA' } }],
      layout: {
        height: 320, margin: { t: 10, b: 50, l: 50, r: 20 },
        xaxis: { title: { text: 'Depth Growth Rate (%/yr)' } }, yaxis: { title: { text: 'Count' } },
        shapes: [
          { type: 'line', x0: 0, x1: 0, y0: 0, y1: 1, yref: 'paper', line: { dash: 'dash', color: 'gray' } },
          { type: 'line', x0: 5, x1: 5, y0: 0, y1: 1, yref: 'paper', line: { dash: 'dash', color: 'red' } },
        ],
        annotations: [{ x: 5, y: 1, yref: 'paper', text: 'Severe', showarrow: false, font: { color: 'red', size: 11 } }],
      }
    };
  }

  /** Builds a scatter plot of depth vs growth rate, colored by risk category, with a danger zone overlay. */
  buildRiskMatrix(points: any[]): void {
    if (!points?.length) return;
    const colorMap: Record<string, string> = { Critical: '#e74c3c', High: '#f39c12', Medium: '#3498db', Low: '#2ecc71' };
    const groups: Record<string, any[]> = {};
    points.forEach(p => {
      const cat = p.risk_category || 'Low';
      if (!groups[cat]) groups[cat] = [];
      groups[cat].push(p);
    });
    const traces: any[] = Object.entries(groups).map(([cat, pts]) => ({
      type: 'scatter', mode: 'markers', name: cat,
      x: pts.map(p => p.later_depth_pct), y: pts.map(p => p.depth_growth_rate),
      marker: { color: colorMap[cat] || '#999', size: 6, opacity: 0.6 },
      hovertemplate: 'Depth: %{x}%<br>Rate: %{y:.2f} %/yr<extra>' + cat + '</extra>',
    }));

    this.riskMatrixChart = {
      data: traces,
      layout: {
        height: 350, margin: { t: 10, b: 50, l: 60, r: 20 },
        xaxis: { title: { text: 'Current Depth (%)' } }, yaxis: { title: { text: 'Growth Rate (%/yr)' } },
        shapes: [{ type: 'rect', x0: 50, x1: 100, y0: 3, y1: 10, fillcolor: 'red', opacity: 0.05, line: { width: 0 } }],
      }
    };
  }

  /** Builds a remaining life histogram with a 10-year threshold annotation line. */
  buildRemainingLife(values: number[]): void {
    if (!values?.length) return;
    this.remainingLifeChart = {
      data: [{ type: 'histogram', x: values, nbinsx: 40, marker: { color: '#e74c3c' } }],
      layout: {
        height: 300, margin: { t: 10, b: 50, l: 50, r: 20 },
        xaxis: { title: { text: 'Remaining Life (years)' } }, yaxis: { title: { text: 'Count' } },
        shapes: [{ type: 'line', x0: 10, x1: 10, y0: 0, y1: 1, yref: 'paper', line: { dash: 'dash', color: 'orange' } }],
        annotations: [{ x: 10, y: 1, yref: 'paper', text: '10yr threshold', showarrow: false, font: { color: 'orange' } }],
      }
    };
  }

  /** Builds a bar chart summarizing anomaly lifecycle categories (e.g., new, growing, stable). */
  buildLifecycleChart(lifecycle: any[]): void {
    if (!lifecycle?.length) return;
    this.lifecycleChart = {
      data: [{
        type: 'bar', x: lifecycle.map(l => l.Category), y: lifecycle.map(l => l.Count),
        marker: { color: ['#1a237e', '#2196f3', '#4caf50', '#ff9800', '#f44336'] },
      }],
      layout: { height: 300, margin: { t: 10, b: 100, l: 60, r: 20 },
        xaxis: { title: { text: 'Lifecycle Category' }, tickangle: -30 }, yaxis: { title: { text: 'Number of Anomalies' } } }
    };
  }

  /** Builds multi-run growth trajectories with dashed 2030 prediction extrapolation and 80% repair threshold. */
  buildTrajectoryChart(trajectories: any[]): void {
    if (!trajectories?.length) return;
    const traces: any[] = [];
    const colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'];
    trajectories.forEach((t, i) => {
      const validY: number[] = []; const validX: number[] = [];
      t.years.forEach((yr: number, j: number) => {
        if (t.depths[j] != null) { validX.push(yr); validY.push(t.depths[j]); }
      });
      traces.push({
        type: 'scatter', mode: 'lines+markers', x: validX, y: validY,
        name: t.joint != null ? `J${t.joint}` : `#${i + 1}`,
        line: { color: colors[i % colors.length] },
        hovertemplate: '%{x}: %{y:.1f}%<extra></extra>',
      });
      // Extrapolation to 2030
      if (t.predicted_2030 != null && validY.length) {
        traces.push({
          type: 'scatter', mode: 'lines', x: [2022, 2030], y: [validY[validY.length - 1], t.predicted_2030],
          line: { dash: 'dot', color: colors[i % colors.length] }, showlegend: false, opacity: 0.5,
        });
      }
    });

    this.trajectoryChart = {
      data: traces,
      layout: {
        height: 420, margin: { t: 10, b: 50, l: 60, r: 20 },
        xaxis: { title: { text: 'Year' } }, yaxis: { title: { text: 'Depth (%)' } },
        shapes: [{ type: 'line', x0: 2005, x1: 2035, y0: 80, y1: 80, line: { dash: 'dash', color: 'red' } }],
        annotations: [{ x: 2033, y: 80, text: '80% Repair Threshold', showarrow: false, font: { color: 'red', size: 11 } }],
      }
    };
  }
}
