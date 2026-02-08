import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { PlotlyModule } from 'angular-plotly.js';
import { MetricCardComponent } from '../../components/metric-card/metric-card.component';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-matching',
  standalone: true,
  imports: [CommonModule, FormsModule, PlotlyModule, MetricCardComponent],
  template: `
    <div class="page-container">
      <h1 class="page-title">Anomaly Matching Results</h1>

      <!-- Pair selector -->
      <div class="pair-selector">
        <label>Run Pair: </label>
        <select [(ngModel)]="selectedPair" (ngModelChange)="loadPair()">
          <option value="2007-2015">2007 vs 2015</option>
          <option value="2015-2022">2015 vs 2022</option>
          <option value="2007-2022">2007 vs 2022 (direct)</option>
        </select>
      </div>

      <!-- Metric cards -->
      <div class="cards-row" *ngIf="stats">
        <app-metric-card label="Total Matches" [value]="stats.total_matches" color="#1a237e"></app-metric-card>
        <app-metric-card label="HIGH Confidence" [value]="stats.high_confidence" color="#2ecc71"></app-metric-card>
        <app-metric-card label="MEDIUM Confidence" [value]="stats.medium_confidence" color="#f39c12"></app-metric-card>
        <app-metric-card label="LOW Confidence" [value]="stats.low_confidence" color="#e74c3c"></app-metric-card>
        <app-metric-card label="New Anomalies" [value]="newCount" color="#9c27b0"></app-metric-card>
        <app-metric-card label="Missing Anomalies" [value]="missingCount" color="#795548"></app-metric-card>
      </div>

      <!-- Charts row -->
      <div class="chart-row">
        <div class="chart-card" *ngIf="confHistChart.data.length">
          <h3>Confidence Score Distribution</h3>
          <plotly-plot [data]="confHistChart.data" [layout]="confHistChart.layout" [config]="{responsive:true}"></plotly-plot>
        </div>
        <div class="chart-card" *ngIf="pieChart.data.length">
          <h3>Match Quality Breakdown</h3>
          <plotly-plot [data]="pieChart.data" [layout]="pieChart.layout" [config]="{responsive:true}"></plotly-plot>
        </div>
      </div>

      <!-- Match table with filters -->
      <div class="table-container" *ngIf="filteredMatches.length">
        <div class="table-filters">
          <label>Confidence:
            <select [(ngModel)]="confFilter" (ngModelChange)="applyFilters()">
              <option value="ALL">All</option>
              <option value="HIGH">HIGH</option>
              <option value="MEDIUM">MEDIUM</option>
              <option value="LOW">LOW</option>
            </select>
          </label>
          <label>Min Depth:
            <input type="number" [(ngModel)]="minDepth" (ngModelChange)="applyFilters()" style="width:60px">
          </label>
          <label>Sort:
            <select [(ngModel)]="sortBy" (ngModelChange)="applyFilters()">
              <option value="confidence">Confidence</option>
              <option value="depth_growth_rate">Growth Rate</option>
              <option value="risk_score">Risk Score</option>
            </select>
          </label>
          <span class="filter-count">Showing {{filteredMatches.length}} of {{allMatches.length}}</span>
        </div>
        <table class="full-width-table">
          <thead>
            <tr>
              <th>Conf</th><th>Score</th>
              <th>Earlier Joint</th><th>Earlier Dist</th><th>Earlier Depth</th>
              <th>Later Joint</th><th>Later Dist</th><th>Later Depth</th>
              <th>Growth %/yr</th><th>Risk</th>
            </tr>
          </thead>
          <tbody>
            <tr *ngFor="let m of filteredMatches.slice(0, 200)" [class]="'conf-' + m.confidence_label">
              <td><span class="conf-badge" [class]="m.confidence_label">{{m.confidence_label}}</span></td>
              <td>{{m.confidence?.toFixed(2)}}</td>
              <td>{{m.earlier_joint}}</td><td>{{m.earlier_distance?.toFixed(0)}}</td><td>{{m.earlier_depth_pct}}%</td>
              <td>{{m.later_joint}}</td><td>{{m.later_distance?.toFixed(0)}}</td><td>{{m.later_depth_pct}}%</td>
              <td>{{m.depth_growth_rate?.toFixed(2)}}</td>
              <td><span class="risk-badge" [class]="m.risk_category">{{m.risk_category}}</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  `,
  styles: [`
    .pair-selector { margin-bottom: 20px; }
    .pair-selector select { padding: 8px 12px; border-radius: 4px; border: 1px solid #ccc; font-size: 14px; margin-left: 8px; }
    .table-filters { display: flex; gap: 16px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
    .table-filters label { font-size: 13px; color: #546e7a; }
    .table-filters select, .table-filters input { padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; margin-left: 4px; }
    .filter-count { font-size: 12px; color: #90a4ae; margin-left: auto; }
    table { border-collapse: collapse; font-size: 13px; }
    th { background: #eceff1; padding: 8px 10px; text-align: right; font-weight: 500; white-space: nowrap; }
    td { padding: 8px 10px; border-bottom: 1px solid #eceff1; text-align: right; }
    .conf-badge { padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
    .conf-badge.HIGH { background: #e8f5e9; color: #2e7d32; }
    .conf-badge.MEDIUM { background: #fff3e0; color: #e65100; }
    .conf-badge.LOW { background: #ffebee; color: #c62828; }
    .risk-badge { padding: 2px 8px; border-radius: 10px; font-size: 11px; }
    .risk-badge.Critical { background: #ffcdd2; color: #b71c1c; }
    .risk-badge.High { background: #ffe0b2; color: #e65100; }
    .risk-badge.Medium { background: #bbdefb; color: #1565c0; }
    .risk-badge.Low { background: #c8e6c9; color: #2e7d32; }
  `]
})
/**
 * Matching results page with confidence distributions, quality pie chart, and filterable match table.
 */
export class MatchingComponent implements OnInit {
  selectedPair = '2015-2022';
  stats: any = null;
  newCount = 0;
  missingCount = 0;
  allMatches: any[] = [];
  filteredMatches: any[] = [];
  confFilter = 'ALL';
  minDepth = 0;
  sortBy = 'confidence';

  confHistChart: any = { data: [], layout: {} };
  pieChart: any = { data: [], layout: {} };

  constructor(private api: ApiService) {}

  /** Loads match data for the default pair on init. */
  ngOnInit(): void {
    this.loadPair();
  }

  /** Fetches match data for the selected run pair from the API. */
  loadPair(): void {
    this.api.getMatches(this.selectedPair).subscribe(d => {
      this.stats = d.stats;
      this.newCount = d.new_anomalies_count;
      this.missingCount = d.missing_anomalies_count;
      this.allMatches = d.matches || [];
      this.applyFilters();
      this.buildCharts();
    });
  }

  /** Client-side filtering by confidence level, minimum depth, and sorting by selected column. */
  applyFilters(): void {
    let result = [...this.allMatches];
    if (this.confFilter !== 'ALL') {
      result = result.filter(m => m.confidence_label === this.confFilter);
    }
    if (this.minDepth > 0) {
      result = result.filter(m => (m.later_depth_pct || 0) >= this.minDepth);
    }
    result.sort((a, b) => (b[this.sortBy] || 0) - (a[this.sortBy] || 0));
    this.filteredMatches = result;
  }

  /** Builds a confidence histogram with HIGH/MEDIUM threshold lines and a quality breakdown pie chart. */
  buildCharts(): void {
    const confidences = this.allMatches.map(m => m.confidence).filter(Boolean);
    this.confHistChart = {
      data: [{ type: 'histogram', x: confidences, nbinsx: 30, marker: { color: '#636EFA' } }],
      layout: {
        height: 300, margin: { t: 10, b: 40, l: 50, r: 20 },
        xaxis: { title: { text: 'Confidence Score' } }, yaxis: { title: { text: 'Count' } },
        shapes: [
          { type: 'line', x0: 0.85, x1: 0.85, y0: 0, y1: 1, yref: 'paper', line: { dash: 'dash', color: 'green' } },
          { type: 'line', x0: 0.60, x1: 0.60, y0: 0, y1: 1, yref: 'paper', line: { dash: 'dash', color: 'orange' } },
        ],
      }
    };

    const labels = this.allMatches.map(m => m.confidence_label);
    const counts: Record<string, number> = {};
    labels.forEach(l => counts[l] = (counts[l] || 0) + 1);
    this.pieChart = {
      data: [{
        type: 'pie', labels: Object.keys(counts), values: Object.values(counts),
        marker: { colors: Object.keys(counts).map(k => k === 'HIGH' ? '#2ecc71' : k === 'MEDIUM' ? '#f39c12' : '#e74c3c') },
      }],
      layout: { height: 300, margin: { t: 10, b: 10, l: 10, r: 10 } }
    };
  }
}
