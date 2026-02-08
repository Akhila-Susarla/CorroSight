import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PlotlyModule } from 'angular-plotly.js';
import { MetricCardComponent } from '../../components/metric-card/metric-card.component';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-integrity-dashboard',
  standalone: true,
  imports: [CommonModule, PlotlyModule, MetricCardComponent],
  template: `
    <div class="page-container">
      <h1 class="page-title">Integrity Dashboard</h1>
      <p class="page-subtitle">ASME B31G Interaction Assessment, Segment Risk Analysis, Repair Prioritization & Population Analytics</p>

      <div *ngIf="loading" class="loading">Loading integrity analytics...</div>

      <div *ngIf="!loading && data">
        <!-- Summary Cards -->
        <div class="cards-row">
          <app-metric-card label="Immediate Repairs" [value]="data.summary.immediate_count" color="#c62828"></app-metric-card>
          <app-metric-card label="Scheduled Repairs" [value]="data.summary.scheduled_count" color="#e65100"></app-metric-card>
          <app-metric-card label="Monitor" [value]="data.summary.monitor_count" color="#1565c0"></app-metric-card>
          <app-metric-card label="Interaction Clusters" [value]="data.summary.interaction_clusters" subtitle="ASME B31G" color="#6a1b9a"></app-metric-card>
          <app-metric-card label="High-Risk Segments" [value]="data.summary.high_risk_segments + ' / ' + data.summary.total_segments" color="#bf360c"></app-metric-card>
        </div>

        <!-- Segment Risk Heatmap -->
        <div class="section">
          <h2 class="section-title">Pipeline Segment Risk Heatmap</h2>
          <p class="section-desc">Pipeline divided into 1,000 ft segments. Color = composite risk score (anomaly density + max depth + growth rate + critical count).</p>
          <div class="chart-card" *ngIf="heatmapChart.data.length">
            <plotly-plot [data]="heatmapChart.data" [layout]="heatmapChart.layout" [config]="{responsive:true}"></plotly-plot>
          </div>
        </div>

        <!-- Anomaly Interaction Assessment -->
        <div class="section">
          <h2 class="section-title">
            ASME B31G Anomaly Interaction Assessment
            <span class="eng-tag">ASME B31G</span>
          </h2>
          <p class="section-desc">Anomalies within 6x wall thickness axially may interact, forming a larger effective defect that reduces burst pressure. Per ASME B31G / RSTRENG standards.</p>

          <div *ngIf="data.interactions.length === 0" class="no-data">No interacting anomaly clusters detected.</div>

          <table class="full-width-table" *ngIf="data.interactions.length">
            <thead>
              <tr>
                <th>Cluster</th><th>Anomalies</th><th>Location (ft)</th><th>Span (ft)</th>
                <th>Eff. Length (in)</th><th>Max Depth %</th><th>Avg Depth %</th>
                <th>Max Growth %/yr</th><th>Threshold (in)</th><th>Severity</th>
              </tr>
            </thead>
            <tbody>
              <tr *ngFor="let c of data.interactions">
                <td>{{c.cluster_id}}</td>
                <td>{{c.anomaly_count}}</td>
                <td>{{c.start_distance_ft?.toFixed(0)}} - {{c.end_distance_ft?.toFixed(0)}}</td>
                <td>{{c.span_ft?.toFixed(1)}}</td>
                <td>{{c.effective_length_in?.toFixed(1)}}</td>
                <td>{{c.max_depth_pct}}</td>
                <td>{{c.avg_depth_pct}}</td>
                <td>{{c.max_growth_rate?.toFixed(2) ?? 'N/A'}}</td>
                <td>{{c.interaction_threshold_in}} (6 x {{c.wall_thickness_in}})</td>
                <td><span class="severity-badge" [class]="c.severity">{{c.severity}}</span></td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Automated Dig List -->
        <div class="section">
          <h2 class="section-title">Automated Dig List / Repair Prioritization</h2>
          <p class="section-desc">Prioritized repair schedule based on urgency score = current depth (40%) + growth rate (30%) + remaining life (30%).</p>

          <!-- Category summary bars -->
          <div class="dig-summary">
            <div class="dig-cat immediate">
              <div class="cat-label">IMMEDIATE</div>
              <div class="cat-count">{{data.summary.immediate_count}}</div>
              <div class="cat-desc">Depth >= 70% or life &lt; 3 years</div>
            </div>
            <div class="dig-cat scheduled">
              <div class="cat-label">SCHEDULED</div>
              <div class="cat-count">{{data.summary.scheduled_count}}</div>
              <div class="cat-desc">Depth >= 50% or life &lt; 7 years</div>
            </div>
            <div class="dig-cat monitor">
              <div class="cat-label">MONITOR</div>
              <div class="cat-count">{{data.summary.monitor_count}}</div>
              <div class="cat-desc">Growing anomalies to track</div>
            </div>
          </div>

          <div class="table-scroll">
            <table class="full-width-table dig-table" *ngIf="digListDisplay.length">
              <thead>
                <tr>
                  <th>Priority</th><th>Category</th><th>Joint</th><th>Distance (ft)</th>
                  <th>Clock</th><th>Depth %</th><th>Growth %/yr</th>
                  <th>Remaining Life</th><th>Type</th><th>ID/OD</th>
                  <th>Urgency</th>
                </tr>
              </thead>
              <tbody>
                <tr *ngFor="let d of digListDisplay; let i = index" [class]="'dig-row-' + d.category">
                  <td>{{i + 1}}</td>
                  <td><span class="cat-badge" [class]="d.category">{{d.category}}</span></td>
                  <td>{{d.joint}}</td>
                  <td>{{d.distance_ft?.toFixed(0)}}</td>
                  <td>{{d.clock?.toFixed(1)}}</td>
                  <td>{{d.depth_pct}}</td>
                  <td>{{d.growth_rate?.toFixed(2)}}</td>
                  <td>{{d.remaining_life_years?.toFixed(1) ?? 'N/A'}} yr</td>
                  <td>{{d.event_type}}</td>
                  <td>{{d.id_od}}</td>
                  <td>{{d.urgency_score}}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div class="show-more" *ngIf="data.dig_list.length > digListLimit">
            <button class="show-more-btn" (click)="showMoreDig()">
              Show {{data.dig_list.length > digListLimit + 25 ? 25 : data.dig_list.length - digListLimit}} more ({{data.dig_list.length}} total)
            </button>
          </div>
        </div>

        <!-- Population Growth Analytics -->
        <div class="section">
          <h2 class="section-title">Population Growth Analytics</h2>
          <p class="section-desc">Growth patterns by pipe quadrant and corrosion type reveal systemic mechanisms (e.g., bottom-of-pipe = water settling, top = gas phase corrosion).</p>

          <div class="chart-row">
            <div class="chart-card" *ngIf="quadrantChart.data.length">
              <h3>Growth Rate by Clock Quadrant</h3>
              <plotly-plot [data]="quadrantChart.data" [layout]="quadrantChart.layout" [config]="{responsive:true}"></plotly-plot>
            </div>
            <div class="chart-card" *ngIf="idOdChart.data.length">
              <h3>Growth Rate by Corrosion Type (ID/OD)</h3>
              <plotly-plot [data]="idOdChart.data" [layout]="idOdChart.layout" [config]="{responsive:true}"></plotly-plot>
            </div>
          </div>

          <div class="chart-row">
            <div class="chart-card" *ngIf="depthBandChart.data.length">
              <h3>Growth Rate by Depth Band</h3>
              <plotly-plot [data]="depthBandChart.data" [layout]="depthBandChart.layout" [config]="{responsive:true}"></plotly-plot>
            </div>
            <div class="chart-card" *ngIf="crossTabChart.data.length">
              <h3>Quadrant x Corrosion Type Heatmap</h3>
              <plotly-plot [data]="crossTabChart.data" [layout]="crossTabChart.layout" [config]="{responsive:true}"></plotly-plot>
            </div>
          </div>

          <!-- Interpretation cards -->
          <div class="insight-grid" *ngIf="data.population.by_quadrant.length">
            <div class="insight-card" *ngFor="let q of data.population.by_quadrant">
              <div class="insight-header">
                <span class="insight-icon">{{getQuadrantIcon(q.quadrant)}}</span>
                <strong>{{q.quadrant}}</strong>
              </div>
              <div class="insight-stats">
                <div><span class="stat-val">{{q.count}}</span> anomalies</div>
                <div>Avg rate: <span class="stat-val">{{q.mean_growth_rate.toFixed(2)}} %/yr</span></div>
                <div>{{q.pct_high_growth}}% high-growth</div>
              </div>
              <div class="insight-interp">{{getQuadrantInterpretation(q)}}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .page-subtitle { color: #78909c; font-size: 14px; margin-top: -8px; margin-bottom: 24px; }
    .loading { text-align: center; padding: 60px; color: #78909c; font-size: 16px; }
    .no-data { text-align: center; padding: 24px; color: #90a4ae; font-style: italic; }

    .section { margin-top: 36px; }
    .section-desc { color: #78909c; font-size: 13px; margin-bottom: 16px; }

    .eng-tag {
      background: linear-gradient(135deg, #e65100, #ff6d00);
      color: white; font-size: 10px; font-weight: 700;
      padding: 2px 8px; border-radius: 8px; margin-left: 8px; vertical-align: middle;
    }

    table { border-collapse: collapse; font-size: 13px; width: 100%; }
    th { background: #eceff1; padding: 8px 10px; text-align: right; font-weight: 500; white-space: nowrap; }
    td { padding: 8px 10px; border-bottom: 1px solid #eceff1; text-align: right; }

    .severity-badge, .cat-badge {
      padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600;
    }
    .severity-badge.HIGH, .cat-badge.IMMEDIATE { background: #ffcdd2; color: #b71c1c; }
    .severity-badge.MEDIUM, .cat-badge.SCHEDULED { background: #ffe0b2; color: #e65100; }
    .severity-badge.LOW, .cat-badge.MONITOR { background: #bbdefb; color: #1565c0; }

    .dig-summary { display: flex; gap: 16px; margin-bottom: 20px; }
    .dig-cat {
      flex: 1; padding: 16px; border-radius: 12px; text-align: center;
    }
    .dig-cat.immediate { background: linear-gradient(135deg, #ffebee, #ffcdd2); border: 2px solid #ef9a9a; }
    .dig-cat.scheduled { background: linear-gradient(135deg, #fff3e0, #ffe0b2); border: 2px solid #ffcc80; }
    .dig-cat.monitor { background: linear-gradient(135deg, #e3f2fd, #bbdefb); border: 2px solid #90caf9; }
    .cat-label { font-size: 12px; font-weight: 700; letter-spacing: 1px; }
    .cat-count { font-size: 32px; font-weight: 800; margin: 4px 0; }
    .cat-desc { font-size: 11px; color: #78909c; }
    .immediate .cat-label, .immediate .cat-count { color: #b71c1c; }
    .scheduled .cat-label, .scheduled .cat-count { color: #e65100; }
    .monitor .cat-label, .monitor .cat-count { color: #1565c0; }

    .dig-row-IMMEDIATE { background: #fff8f8; }
    .dig-row-SCHEDULED { background: #fffaf5; }

    .table-scroll { max-height: 500px; overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 8px; }
    .dig-table th { position: sticky; top: 0; z-index: 1; }

    .show-more { text-align: center; margin-top: 12px; }
    .show-more-btn {
      padding: 8px 20px; background: #e8eaf6; color: #1a237e;
      border: 1px solid #c5cae9; border-radius: 6px; cursor: pointer; font-size: 13px;
    }
    .show-more-btn:hover { background: #c5cae9; }

    .insight-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 16px; margin-top: 20px;
    }
    .insight-card {
      background: white; border-radius: 10px; padding: 16px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.1); border-top: 3px solid #1a237e;
    }
    .insight-header { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; font-size: 15px; }
    .insight-icon { font-size: 20px; }
    .insight-stats { font-size: 13px; color: #555; line-height: 1.8; }
    .stat-val { font-weight: 700; color: #1a237e; }
    .insight-interp { font-size: 12px; color: #78909c; margin-top: 8px; font-style: italic; border-top: 1px solid #eee; padding-top: 8px; }
  `]
})
/**
 * Integrity engineering dashboard with ASME B31G interaction assessment,
 * segment risk heatmap, automated dig list, and population growth analytics.
 */
export class IntegrityDashboardComponent implements OnInit {
  data: any = null;
  loading = true;
  digListLimit = 25;

  heatmapChart: any = { data: [], layout: {} };
  quadrantChart: any = { data: [], layout: {} };
  idOdChart: any = { data: [], layout: {} };
  depthBandChart: any = { data: [], layout: {} };
  crossTabChart: any = { data: [], layout: {} };

  constructor(private api: ApiService) {}

  /** Fetches integrity dashboard data and builds heatmap and population charts on init. */
  ngOnInit(): void {
    this.api.getIntegrityDashboard().subscribe({
      next: (d: any) => {
        this.data = d;
        this.loading = false;
        this.buildHeatmap(d.segments);
        this.buildPopulationCharts(d.population);
      },
      error: () => {
        this.loading = false;
      }
    });
  }

  /** Returns the current page of the dig list (paginated by digListLimit). */
  get digListDisplay(): any[] {
    if (!this.data?.dig_list) return [];
    return this.data.dig_list.slice(0, this.digListLimit);
  }

  /** Increases the dig list page size to show 25 more rows. */
  showMoreDig(): void {
    this.digListLimit += 25;
  }

  /** Builds a bar chart of per-segment risk scores with green-to-red color coding based on score thresholds. */
  buildHeatmap(segments: any[]): void {
    if (!segments?.length) return;

    const scores = segments.map(s => s.risk_score);
    const mids = segments.map(s => s.midpoint_ft);
    const labels = segments.map(s =>
      `Segment ${s.segment}<br>${s.start_ft.toFixed(0)}-${s.end_ft.toFixed(0)} ft<br>` +
      `Anomalies: ${s.anomaly_count}<br>Max Depth: ${s.max_depth_pct}%<br>` +
      `Avg Growth: ${s.avg_growth_rate.toFixed(2)} %/yr<br>Risk Score: ${s.risk_score}`
    );

    // Color scale: green → yellow → red
    const colors = scores.map(s => {
      if (s < 20) return '#4caf50';
      if (s < 40) return '#8bc34a';
      if (s < 55) return '#ffeb3b';
      if (s < 70) return '#ff9800';
      if (s < 85) return '#f44336';
      return '#b71c1c';
    });

    this.heatmapChart = {
      data: [{
        type: 'bar',
        x: mids,
        y: scores,
        marker: { color: colors },
        text: labels,
        hoverinfo: 'text',
        width: segments.length > 0 ? (mids[1] - mids[0]) * 0.9 : 900,
      }],
      layout: {
        height: 280,
        margin: { t: 10, b: 50, l: 60, r: 20 },
        xaxis: { title: { text: 'Pipeline Distance (ft)' } },
        yaxis: { title: { text: 'Risk Score' }, range: [0, 100] },
        shapes: [
          { type: 'line', x0: mids[0], x1: mids[mids.length - 1], y0: 60, y1: 60, line: { dash: 'dash', color: '#f44336', width: 1 } },
        ],
        annotations: [
          { x: mids[mids.length - 1], y: 62, text: 'High Risk', showarrow: false, font: { color: '#f44336', size: 10 } },
        ],
      }
    };
  }

  /** Builds 4 population charts: quadrant bar, ID/OD bar, depth band bar, and quadrant x ID/OD heatmap. */
  buildPopulationCharts(pop: any): void {
    if (!pop) return;

    // Quadrant bar chart
    if (pop.by_quadrant?.length) {
      const quads = pop.by_quadrant;
      this.quadrantChart = {
        data: [{
          type: 'bar',
          x: quads.map((q: any) => q.quadrant),
          y: quads.map((q: any) => q.mean_growth_rate),
          marker: {
            color: quads.map((q: any) => {
              if (q.quadrant.includes('Top')) return '#42a5f5';
              if (q.quadrant.includes('Bottom')) return '#ef5350';
              if (q.quadrant.includes('Right')) return '#66bb6a';
              return '#ffa726';
            })
          },
          text: quads.map((q: any) => `n=${q.count}`),
          textposition: 'outside',
        }],
        layout: {
          height: 320, margin: { t: 20, b: 80, l: 60, r: 20 },
          xaxis: { title: { text: 'Pipe Quadrant' } },
          yaxis: { title: { text: 'Mean Growth Rate (%/yr)' } },
        }
      };
    }

    // ID/OD bar chart
    if (pop.by_id_od?.length) {
      const items = pop.by_id_od;
      this.idOdChart = {
        data: [{
          type: 'bar',
          x: items.map((d: any) => d.type),
          y: items.map((d: any) => d.mean_growth_rate),
          marker: { color: items.map((d: any) => d.type === 'Internal' ? '#7e57c2' : d.type === 'External' ? '#26a69a' : '#bdbdbd') },
          text: items.map((d: any) => `n=${d.count}`),
          textposition: 'outside',
        }],
        layout: {
          height: 320, margin: { t: 20, b: 60, l: 60, r: 20 },
          xaxis: { title: { text: 'Corrosion Type' } },
          yaxis: { title: { text: 'Mean Growth Rate (%/yr)' } },
        }
      };
    }

    // Depth band bar chart
    if (pop.by_depth_band?.length) {
      const bands = pop.by_depth_band.sort((a: any, b: any) => {
        const order: Record<string, number> = { '0-20%': 0, '20-40%': 1, '40-60%': 2, '60%+': 3 };
        return (order[a.band] ?? 99) - (order[b.band] ?? 99);
      });
      this.depthBandChart = {
        data: [{
          type: 'bar',
          x: bands.map((d: any) => d.band),
          y: bands.map((d: any) => d.mean_growth_rate),
          marker: { color: ['#66bb6a', '#ffa726', '#ef5350', '#b71c1c'] },
          text: bands.map((d: any) => `n=${d.count}`),
          textposition: 'outside',
        }],
        layout: {
          height: 320, margin: { t: 20, b: 60, l: 60, r: 20 },
          xaxis: { title: { text: 'Current Depth Band' } },
          yaxis: { title: { text: 'Mean Growth Rate (%/yr)' } },
        }
      };
    }

    // Cross-tab heatmap: Quadrant x ID/OD
    if (pop.quadrant_id_od?.length) {
      const items = pop.quadrant_id_od;
      const quads = [...new Set(items.map((d: any) => d.quadrant))];
      const types = [...new Set(items.map((d: any) => d.id_od))];

      const z: number[][] = [];
      const text: string[][] = [];
      for (const t of types) {
        const row: number[] = [];
        const textRow: string[] = [];
        for (const q of quads) {
          const item = items.find((d: any) => d.quadrant === q && d.id_od === t);
          row.push(item ? item.mean_growth_rate : 0);
          textRow.push(item ? `${item.mean_growth_rate.toFixed(2)} %/yr (n=${item.count})` : 'N/A');
        }
        z.push(row);
        text.push(textRow);
      }

      this.crossTabChart = {
        data: [{
          type: 'heatmap',
          x: quads,
          y: types,
          z: z,
          text: text,
          hoverinfo: 'text',
          colorscale: [[0, '#e8f5e9'], [0.5, '#fff9c4'], [1, '#ffcdd2']],
          showscale: true,
          colorbar: { title: '%/yr' },
        }],
        layout: {
          height: 300, margin: { t: 20, b: 80, l: 100, r: 20 },
          xaxis: { title: { text: 'Quadrant' } },
          yaxis: { title: { text: 'Corrosion Type' } },
        }
      };
    }
  }

  /** Returns a directional arrow emoji for the given pipe quadrant label. */
  getQuadrantIcon(quadrant: string): string {
    if (quadrant.includes('Top')) return '\u2B06';
    if (quadrant.includes('Bottom')) return '\u2B07';
    if (quadrant.includes('Right')) return '\u27A1';
    if (quadrant.includes('Left')) return '\u2B05';
    return '\u2753';
  }

  /** Returns a domain interpretation of corrosion patterns by pipe position and growth rate. */
  getQuadrantInterpretation(q: any): string {
    const quad = q.quadrant;
    const rate = q.mean_growth_rate;
    if (quad.includes('Bottom') && rate > 0.5)
      return 'Elevated bottom-of-pipe corrosion may indicate water settling (internal) or soil-side corrosion (external).';
    if (quad.includes('Top') && rate > 0.5)
      return 'Top-of-pipe corrosion may suggest gas-phase corrosion (internal) or coating degradation (external).';
    if (quad.includes('Right') || quad.includes('Left'))
      return 'Lateral corrosion is less common. If elevated, check for pipe support contact or localized coating issues.';
    return 'Growth rates within normal range for this quadrant.';
  }
}
