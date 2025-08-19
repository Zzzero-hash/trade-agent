import useSWR from "swr";
import { apiFetch, endpoints } from "@/lib/api";
import { RunRecord } from "@/types";

export default function RunTable() {
  const fetcher = (url: string) => apiFetch<RunRecord[]>(url);
  const { data, isLoading } = useSWR<RunRecord[]>(endpoints.runs(), fetcher);
  return (
    <div className="card overflow-auto">
      <table className="w-full text-sm">
        <thead className="text-neutral-400">
          <tr>
            <th className="text-left p-2">ID</th>
            <th className="text-left p-2">Status</th>
            <th className="text-left p-2">Started</th>
            <th className="text-left p-2">Finished</th>
            <th className="text-left p-2">Metric</th>
          </tr>
        </thead>
        <tbody>
          {isLoading && (
            <tr>
              <td colSpan={5} className="p-2 text-center text-neutral-500">
                Loading...
              </td>
            </tr>
          )}
          {data?.map((r) => (
            <tr
              key={r.id}
              className="border-t border-neutral-800 hover:bg-neutral-800/50"
            >
              <td className="p-2 font-mono text-xs">{r.id.slice(0, 8)}</td>
              <td className="p-2">
                <span
                  className={`px-2 py-0.5 rounded text-xs bg-neutral-800 ${statusColor(r.status)}`}
                >
                  {r.status}
                </span>
              </td>
              <td className="p-2 text-xs">{formatDt(r.started_at)}</td>
              <td className="p-2 text-xs">
                {r.finished_at ? formatDt(r.finished_at) : "-"}
              </td>
              <td className="p-2 text-xs">{r.metric?.toFixed(4) ?? "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatDt(s: string) {
  return new Date(s).toLocaleString();
}
function statusColor(status: string) {
  switch (status) {
    case "running":
      return "text-amber-300";
    case "completed":
      return "text-emerald-300";
    case "failed":
      return "text-rose-300";
    default:
      return "text-neutral-300";
  }
}
