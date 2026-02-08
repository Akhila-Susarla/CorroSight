import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', redirectTo: 'overview', pathMatch: 'full' },
  { path: 'overview', loadComponent: () => import('./pages/data-overview/data-overview.component').then(m => m.DataOverviewComponent) },
  { path: 'alignment', loadComponent: () => import('./pages/alignment/alignment.component').then(m => m.AlignmentComponent) },
  { path: 'matching', loadComponent: () => import('./pages/matching/matching.component').then(m => m.MatchingComponent) },
  { path: 'growth', loadComponent: () => import('./pages/growth/growth.component').then(m => m.GrowthComponent) },
  { path: 'pipeline', loadComponent: () => import('./pages/pipeline-view/pipeline-view.component').then(m => m.PipelineViewComponent) },
  { path: 'virtual-ili', loadComponent: () => import('./pages/virtual-ili/virtual-ili.component').then(m => m.VirtualIliComponent) },
  { path: 'integrity', loadComponent: () => import('./pages/integrity-dashboard/integrity-dashboard.component').then(m => m.IntegrityDashboardComponent) },
];
