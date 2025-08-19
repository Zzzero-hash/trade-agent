import { ReactNode } from "react";

export function KpiCard({
  title,
  value,
  diff,
  icon,
}: {
  title: string;
  value: ReactNode;
  diff?: number;
  icon?: ReactNode;
}) {
  return (
    <div className="card space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-wide text-neutral-400">
          {title}
        </span>
        {icon}
      </div>
      <div className="text-2xl font-semibold">{value}</div>
      {typeof diff === "number" && (
        <div
          className={`text-xs ${diff >= 0 ? "text-emerald-400" : "text-rose-400"}`}
        >
          {diff >= 0 ? "+" : ""}
          {diff.toFixed(2)}%
        </div>
      )}
    </div>
  );
}
