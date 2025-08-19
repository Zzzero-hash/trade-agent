"use client";
import MetricChart from "@/components/MetricChart";
import LiveLog from "@/components/LiveLog";
import RunTable from "@/components/RunTable";
import { KpiCard } from "@/components/KpiCard";
import useSWR from "swr";
import { apiFetch, endpoints } from "@/lib/api";
import { RunRecord } from "@/types";

interface Health {
  status: string;
  version?: string;
  uptime_seconds?: number;
}

export default function DashboardPage() {
  const runsFetcher = (url: string) => apiFetch<RunRecord[]>(url);
  const healthFetcher = (url: string) => apiFetch<Health>(url);
  const { data: runs } = useSWR<RunRecord[]>(endpoints.runs(), runsFetcher);
  const { data: health } = useSWR<Health>(endpoints.health(), healthFetcher);
  const completed = (runs || []).filter((r) => r.status === "completed");
  return (
    <div className="space-y-6">
      <div className="grid md:grid-cols-4 gap-4">
        <KpiCard title="Total Runs" value={runs?.length ?? 0} />
        <KpiCard title="Completed" value={completed.length} />
        <KpiCard
          title="Success %"
          value={
            runs && runs.length
              ? ((completed.length / runs.length) * 100).toFixed(1) + "%"
              : "0%"
          }
        />
        <KpiCard title="Uptime" value={formatUptime(health?.uptime_seconds)} />
      </div>
      <div className="grid md:grid-cols-3 gap-4">
        <div className="md:col-span-2 space-y-4">
          <MetricChart />
          <RunTable />
        </div>
        <LiveLog />
      </div>
    </div>
  );
}

function formatUptime(s?: number) {
  if (!s) return "-";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}h ${m}m`;
}
