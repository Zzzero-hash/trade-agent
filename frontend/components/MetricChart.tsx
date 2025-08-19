"use client";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { useEventSource } from "@/hooks/useEventSource";
import { endpoints } from "@/lib/api";
import { useEffect, useState } from "react";
import { MetricPoint } from "@/types";

export default function MetricChart() {
  const { data } = useEventSource<MetricPoint>(endpoints.metricsStream());
  const [points, setPoints] = useState<MetricPoint[]>([]);

  useEffect(() => {
    if (data) setPoints((p) => [...p.slice(-499), data]);
  }, [data]);

  return (
    <div className="card h-80">
      <h3 className="text-sm mb-2 text-neutral-400">Live Metric</h3>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points}>
          <XAxis
            dataKey="t"
            tickFormatter={(t) => new Date(t).toLocaleTimeString()}
            hide
          />
          <YAxis
            domain={["auto", "auto"]}
            width={60}
            tick={{ fill: "#888" }}
            tickFormatter={(v) => v.toFixed(2)}
          />
          <Tooltip
            labelFormatter={(l) => new Date(Number(l)).toLocaleTimeString()}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#36abff"
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
