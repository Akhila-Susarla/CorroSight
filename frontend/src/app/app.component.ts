import { Component, ViewChild, ElementRef } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { CommonModule } from '@angular/common';
import { ApiService } from './services/api.service';
import { ChatCopilotComponent } from './components/chat-copilot/chat-copilot.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet, RouterLink, RouterLinkActive, ChatCopilotComponent],
  template: `
    <div class="app-layout">
      <nav class="sidebar">
        <div class="sidebar-header">
          <h2>CorroSight</h2>
          <p>Pipeline Integrity Intelligence</p>
        </div>
        <ul class="nav-links">
          <li><a routerLink="/overview" routerLinkActive="active">Data Overview</a></li>
          <li><a routerLink="/alignment" routerLinkActive="active">Alignment</a></li>
          <li><a routerLink="/matching" routerLinkActive="active">Matching</a></li>
          <li><a routerLink="/growth" routerLinkActive="active">Growth Analysis</a></li>
          <li><a routerLink="/pipeline" routerLinkActive="active">Pipeline View</a></li>
          <li class="nav-divider"></li>
          <li><a routerLink="/integrity" routerLinkActive="active" class="eng-link">Integrity Dashboard</a></li>
          <li><a routerLink="/virtual-ili" routerLinkActive="active" class="ai-link">Virtual ILI Predictor</a></li>
        </ul>
        <div class="sidebar-footer">
          <button class="run-btn" (click)="fileInput.click()" [disabled]="pipelineRunning">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="white">
              <path d="M8 5v14l11-7z"/>
            </svg>
            {{ pipelineRunning ? 'Running...' : 'Run Pipeline' }}
          </button>
          <input type="file" #fileInput accept=".xlsx,.xls" style="display:none"
                 (change)="onFileSelected($event)">
          <button class="export-btn" (click)="onExport()">Export to Excel</button>
        </div>
      </nav>
      <main class="content">
        <router-outlet></router-outlet>
      </main>
    </div>
    <app-chat-copilot></app-chat-copilot>

    <!-- Pipeline progress overlay -->
    <div class="progress-overlay" *ngIf="showProgress">
      <div class="progress-modal">
        <div class="progress-header">
          <div>
            <h3>CorroSight Analysis</h3>
            <div class="progress-filename" *ngIf="uploadedFileName">{{ uploadedFileName }}</div>
          </div>
          <button class="close-btn" *ngIf="pipelineStatus === 'completed' || pipelineStatus === 'error'" (click)="showProgress = false">&times;</button>
        </div>

        <div class="progress-step">{{ progressStep }}</div>
        <div class="progress-counter">Step {{ progressStepNumber }} of {{ progressTotalSteps }}</div>

        <div class="progress-bar-track">
          <div class="progress-bar-fill" [style.width.%]="progressPercent"></div>
        </div>

        <div class="progress-stats" *ngIf="progressStats">
          <div class="stat" *ngIf="progressStats.runs_loaded">
            <span class="stat-val">{{ progressStats.runs_loaded }}</span>
            <span class="stat-label">Runs Loaded</span>
          </div>
          <div class="stat" *ngIf="progressStats.total_rows">
            <span class="stat-val">{{ progressStats.total_rows | number }}</span>
            <span class="stat-label">Total Rows</span>
          </div>
          <div class="stat" *ngIf="progressStats.girth_welds_matched">
            <span class="stat-val">{{ progressStats.girth_welds_matched | number }}</span>
            <span class="stat-label">Girth Welds</span>
          </div>
          <div class="stat" *ngIf="progressStats.total_matches">
            <span class="stat-val">{{ progressStats.total_matches | number }}</span>
            <span class="stat-label">Matches</span>
          </div>
          <div class="stat" *ngIf="progressStats.triple_matches !== undefined">
            <span class="stat-val">{{ progressStats.triple_matches | number }}</span>
            <span class="stat-label">3-Run Chains</span>
          </div>
        </div>

        <div class="progress-done" *ngIf="pipelineStatus === 'completed'">Analysis complete! Refresh any page to see updated results.</div>
        <div class="progress-error" *ngIf="pipelineStatus === 'error'">{{ progressError }}</div>
      </div>
    </div>
  `,
  styles: [`
    .app-layout { display: flex; min-height: 100vh; }
    .sidebar {
      width: 240px; background: #1a237e; color: white;
      display: flex; flex-direction: column; position: fixed;
      top: 0; left: 0; bottom: 0; z-index: 100;
    }
    .sidebar-header { padding: 24px 20px 16px; border-bottom: 1px solid rgba(255,255,255,0.1); }
    .sidebar-header h2 { font-size: 20px; font-weight: 600; }
    .sidebar-header p { font-size: 12px; opacity: 0.7; margin-top: 4px; }
    .nav-links { list-style: none; padding: 12px 0; flex: 1; }
    .nav-links li a {
      display: block; padding: 12px 20px; color: rgba(255,255,255,0.8);
      text-decoration: none; font-size: 14px; transition: all 0.2s;
    }
    .nav-links li a:hover { background: rgba(255,255,255,0.1); color: white; }
    .nav-links li a.active { background: rgba(255,255,255,0.15); color: white; font-weight: 500; border-left: 3px solid #7c4dff; }
    .nav-divider { height: 1px; background: rgba(255,255,255,0.1); margin: 8px 20px; }
    .ai-link, .eng-link { position: relative; }
    .ai-link::before {
      content: 'AI'; position: absolute; right: 16px; top: 50%; transform: translateY(-50%);
      background: linear-gradient(135deg, #7c4dff, #e040fb); color: white;
      font-size: 9px; font-weight: 700; padding: 2px 6px; border-radius: 8px;
    }
    .eng-link::before {
      content: 'B31G'; position: absolute; right: 12px; top: 50%; transform: translateY(-50%);
      background: linear-gradient(135deg, #e65100, #ff6d00); color: white;
      font-size: 8px; font-weight: 700; padding: 2px 5px; border-radius: 8px;
    }
    .sidebar-footer { padding: 16px 20px; border-top: 1px solid rgba(255,255,255,0.1); display: flex; flex-direction: column; gap: 8px; }
    .export-btn {
      width: 100%; padding: 10px; background: #7c4dff; color: white;
      border: none; border-radius: 6px; cursor: pointer; font-size: 14px;
    }
    .export-btn:hover { background: #651fff; }
    .run-btn {
      width: 100%; padding: 10px; background: #00897b; color: white;
      border: none; border-radius: 6px; cursor: pointer; font-size: 14px;
      display: flex; align-items: center; justify-content: center; gap: 8px;
      font-weight: 500; transition: all 0.2s;
    }
    .run-btn:hover { background: #00796b; }
    .run-btn:disabled { background: #78909c; cursor: default; }
    .content { margin-left: 240px; flex: 1; min-height: 100vh; }

    /* Progress overlay */
    .progress-overlay {
      position: fixed; top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0,0,0,0.6); z-index: 2000;
      display: flex; align-items: center; justify-content: center;
    }
    .progress-modal {
      background: white; border-radius: 16px; padding: 32px;
      width: 440px; box-shadow: 0 16px 48px rgba(0,0,0,0.3);
    }
    .progress-header {
      display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;
    }
    .progress-header h3 { font-size: 20px; color: #1a237e; margin: 0; }
    .progress-filename { font-size: 12px; color: #90a4ae; margin-top: 4px; }
    .close-btn {
      background: none; border: none; font-size: 24px; color: #999;
      cursor: pointer; padding: 0 4px; line-height: 1;
    }
    .progress-step { font-size: 14px; color: #333; margin-bottom: 4px; }
    .progress-counter { font-size: 12px; color: #999; margin-bottom: 12px; }
    .progress-bar-track {
      height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden; margin-bottom: 20px;
    }
    .progress-bar-fill {
      height: 100%; background: linear-gradient(90deg, #00897b, #26a69a);
      border-radius: 4px; transition: width 0.4s ease;
    }
    .progress-stats {
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 16px;
    }
    .stat {
      text-align: center; background: #f5f5f5; padding: 10px 8px; border-radius: 8px;
    }
    .stat-val { display: block; font-size: 18px; font-weight: 700; color: #1a237e; }
    .stat-label { font-size: 10px; color: #999; text-transform: uppercase; letter-spacing: 0.5px; }
    .progress-done {
      text-align: center; color: #2e7d32; font-weight: 500; font-size: 14px;
      padding: 8px; background: #e8f5e9; border-radius: 8px;
    }
    .progress-error {
      text-align: center; color: #c62828; font-size: 13px;
      padding: 8px; background: #ffebee; border-radius: 8px;
    }
  `]
})
/**
 * Main application shell component.
 * Provides sidebar navigation, Excel file upload with pipeline execution, and a progress overlay.
 */
export class AppComponent {
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;

  showProgress = false;
  pipelineRunning = false;
  pipelineStatus = 'idle';
  progressStep = '';
  progressStepNumber = 0;
  progressTotalSteps = 6;
  progressPercent = 0;
  progressStats: any = null;
  progressError = '';
  uploadedFileName = '';
  private pollTimer: any = null;

  constructor(private api: ApiService) {}

  /** Triggers an Excel export of all pipeline results via the API. */
  onExport(): void {
    this.api.exportExcel();
  }

  /** Handles file selection: uploads the Excel file to the backend and starts progress polling. */
  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files?.length) return;
    const file = input.files[0];
    this.uploadedFileName = file.name;
    input.value = '';  // reset so same file can be re-selected

    this.pipelineRunning = true;
    this.showProgress = true;
    this.pipelineStatus = 'running';
    this.progressStep = 'Uploading ' + file.name + '...';
    this.progressStepNumber = 0;
    this.progressPercent = 0;
    this.progressStats = null;
    this.progressError = '';

    this.api.runPipeline(file).subscribe({
      next: () => {
        this.startPolling();
      },
      error: () => {
        this.pipelineStatus = 'error';
        this.progressError = 'Failed to upload file.';
        this.pipelineRunning = false;
      }
    });
  }

  /** Polls /api/run-pipeline/status every 500ms until the pipeline completes or errors out. */
  private startPolling(): void {
    this.pollTimer = setInterval(() => {
      this.api.getPipelineStatus().subscribe({
        next: (res: any) => {
          this.pipelineStatus = res.status;
          this.progressStep = res.step || '';
          this.progressStepNumber = res.step_number || 0;
          this.progressTotalSteps = res.total_steps || 6;
          this.progressPercent = (this.progressStepNumber / this.progressTotalSteps) * 100;
          this.progressStats = res.stats || null;

          if (res.status === 'completed' || res.status === 'error') {
            this.pipelineRunning = false;
            if (res.status === 'error') {
              this.progressError = res.error || 'Unknown error';
            }
            if (res.status === 'completed') {
              this.progressPercent = 100;
            }
            clearInterval(this.pollTimer);
            this.pollTimer = null;
          }
        },
        error: () => {
          // Keep polling on transient errors
        }
      });
    }, 500);
  }
}
