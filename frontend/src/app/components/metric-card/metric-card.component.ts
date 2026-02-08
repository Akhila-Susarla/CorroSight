import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-metric-card',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="metric-card" [style.borderLeft]="'4px solid ' + color">
      <div class="metric-label">{{ label }}</div>
      <div class="metric-value">{{ value }}</div>
      <div class="metric-sub" *ngIf="subtitle">{{ subtitle }}</div>
    </div>
  `,
  styles: [`
    .metric-card {
      background: white;
      border-radius: 8px;
      padding: 16px 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12);
      min-width: 160px;
      flex: 1;
    }
    .metric-label { font-size: 12px; color: #78909c; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 28px; font-weight: 600; color: #263238; margin: 4px 0; }
    .metric-sub { font-size: 12px; color: #90a4ae; }
  `]
})
export class MetricCardComponent {
  @Input() label = '';
  @Input() value: string | number = '';
  @Input() subtitle = '';
  @Input() color = '#1a237e';
}
