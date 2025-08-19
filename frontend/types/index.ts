export interface RunRecord {
  id: string;
  status: "pending" | "running" | "completed" | "failed";
  started_at: string;
  finished_at?: string;
  metric?: number;
}

export interface MetricPoint {
  t: number; // epoch ms
  value: number;
  series?: string;
}
