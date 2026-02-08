import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PlotlyModule } from 'angular-plotly.js';
import { MetricCardComponent } from '../../components/metric-card/metric-card.component';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-alignment',
  standalone: true,
  imports: [CommonModule, PlotlyModule, MetricCardComponent],
  template: `
    <div class="page-container">
      <h1 class="page-title">Reference Point Alignment</h1>

      <div class="cards-row" *ngIf="stats">
        <app-metric-card label="Common Girth Welds" [value]="stats.common_joints" color="#1a237e"></app-metric-card>
        <app-metric-card label="Joint Range" [value]="'J' + stats.joint_range?.[0] + ' - J' + stats.joint_range?.[1]" color="#0d47a1"></app-metric-card>
        <app-metric-card *ngFor="let d of driftStats"
          [label]="'Avg Drift ' + d.pair" [value]="d.abs_mean + ' ft'" [subtitle]="'Max: ' + d.max + ' ft'" [color]="d.color">
        </app-metric-card>
      </div>

      <!-- Distance comparison scatters -->
      <div class="chart-row">
        <div class="chart-card" *ngIf="scatter07_15.data.length">
          <h3>2007 vs 2015 Girth Weld Distances</h3>
          <plotly-plot [data]="scatter07_15.data" [layout]="scatter07_15.layout" [config]="{responsive:true}"></plotly-plot>
        </div>
        <div class="chart-card" *ngIf="scatter15_22.data.length">
          <h3>2015 vs 2022 Girth Weld Distances</h3>
          <plotly-plot [data]="scatter15_22.data" [layout]="scatter15_22.layout" [config]="{responsive:true}"></plotly-plot>
        </div>
      </div>

      <!-- Drift along pipeline -->
      <div class="chart-card" *ngIf="driftChart.data.length">
        <h3>Odometer Drift Along Pipeline</h3>
        <plotly-plot [data]="driftChart.data" [layout]="driftChart.layout" [config]="{responsive:true}"></plotly-plot>
      </div>

      <!-- Alignment table -->
      <div class="table-container" *ngIf="tableRows.length">
        <h3 class="section-title">Joint-by-Joint Alignment (first 100)</h3>
        <table class="full-width-table">
          <thead>
            <tr>
              <th>Joint</th><th>Dist 2007</th><th>Dist 2015</th><th>Dist 2022</th>
              <th>Delta 07-15</th><th>Delta 15-22</th>
            </tr>
          </thead>
          <tbody>
            <tr *ngFor="let r of tableRows.slice(0, 100)">
              <td>{{r.joint_number}}</td>
              <td>{{r.dist_2007?.toFixed(1)}}</td><td>{{r.dist_2015?.toFixed(1)}}</td><td>{{r.dist_2022?.toFixed(1)}}</td>
              <td>{{r.delta_2007_2015?.toFixed(2)}}</td><td>{{r.delta_2015_2022?.toFixed(2)}}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  `,
  styles: [`
    table { border-collapse: collapse; font-size: 13px; }
    th { background: #eceff1; padding: 8px 12px; text-align: right; font-weight: 500; }
    th:first-child { text-align: left; }
    td { padding: 8px 12px; border-bottom: 1px solid #eceff1; text-align: right; }
    td:first-child { text-align: left; }
  `]
})
/**
 * Alignment page showing girth weld distance scatter plots, odometer drift, and joint alignment table.
 */
export class AlignmentComponent implements OnInit {
  stats: any = null;
  driftStats: any[] = [];
  tableRows: any[] = [];
  scatter07_15: any = { data: [], layout: {} };
  scatter15_22: any = { data: [], layout: {} };
  driftChart: any = { data: [], layout: {} };

  constructor(private api: ApiService) {}

  /** Fetches alignment data and builds all charts on init. */
  ngOnInit(): void {
    this.api.getAlignment().subscribe(d => {
      this.stats = d.stats;
      this.tableRows = d.alignment_table;
      this.buildDriftStats();
      this.buildScatters();
      this.buildDriftChart();
    });
  }

  /** Computes summary drift statistics for each run pair for the metric cards. */
  buildDriftStats(): void {
    const pairs = [
      { key: 'drift_2007_2015', pair: '2007-2015', color: '#636EFA' },
      { key: 'drift_2015_2022', pair: '2015-2022', color: '#EF553B' },
    ];
    this.driftStats = pairs
      .filter(p => this.stats[p.key])
      .map(p => ({ ...this.stats[p.key], pair: p.pair, color: p.color }));
  }

  /** Creates distance comparison scatter plots with a perfect alignment (y=x) reference line. */
  buildScatters(): void {
    const buildScatter = (xCol: string, yCol: string, xLabel: string, yLabel: string) => {
      const x = this.tableRows.map(r => r[xCol]);
      const y = this.tableRows.map(r => r[yCol]);
      const joints = this.tableRows.map(r => r.joint_number);
      const minV = Math.min(...x.filter(Boolean), ...y.filter(Boolean));
      const maxV = Math.max(...x.filter(Boolean), ...y.filter(Boolean));
      return {
        data: [
          { type: 'scatter', mode: 'markers', x, y, name: 'Girth Weld Positions',
            marker: { size: 4, color: '#636EFA', opacity: 0.7 },
            customdata: joints, hovertemplate: 'J%{customdata}<br>%{x:.1f} → %{y:.1f} ft<extra></extra>' },
          { type: 'scatter', mode: 'lines', x: [minV, maxV], y: [minV, maxV],
            line: { dash: 'dash', color: 'red', width: 2 }, name: 'Perfect Alignment (no drift)', showlegend: true },
        ],
        layout: { height: 380, margin: { t: 10, b: 50, l: 60, r: 20 },
                  xaxis: { title: { text: xLabel } }, yaxis: { title: { text: yLabel } },
                  legend: { orientation: 'h', y: 1.08, x: 0.5, xanchor: 'center' } }
      };
    };
    this.scatter07_15 = buildScatter('dist_2007', 'dist_2015', '2007 Distance (ft)', '2015 Distance (ft)');
    this.scatter15_22 = buildScatter('dist_2015', 'dist_2022', '2015 Distance (ft)', '2022 Distance (ft)');
  }

  /** Shows odometer drift along the pipeline distance with a zero-drift reference line. */
  buildDriftChart(): void {
    const traces: any[] = [
      { type: 'scatter', mode: 'markers', x: this.tableRows.map(r => r.dist_2022),
        y: this.tableRows.map(r => r.delta_2007_2015), marker: { size: 3, color: '#636EFA' }, name: 'Drift 2007 vs 2015' },
      { type: 'scatter', mode: 'markers', x: this.tableRows.map(r => r.dist_2022),
        y: this.tableRows.map(r => r.delta_2015_2022), marker: { size: 3, color: '#EF553B' }, name: 'Drift 2015 vs 2022' },
    ];
    this.driftChart = {
      data: traces,
      layout: {
        height: 350, margin: { t: 10, b: 50, l: 60, r: 20 },
        xaxis: { title: { text: 'Pipeline Distance (ft)' } }, yaxis: { title: { text: 'Odometer Drift (ft)' } },
        shapes: [{ type: 'line', x0: 0, x1: 58000, y0: 0, y1: 0, line: { dash: 'dash', color: 'gray' } }],
        legend: { orientation: 'h', y: 1.08, x: 0.5, xanchor: 'center' },
      }
    };
  }
}
