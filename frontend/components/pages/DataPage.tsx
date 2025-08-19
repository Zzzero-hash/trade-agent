"use client";
import useSWR from "swr";
import { apiFetch, endpoints } from "@/lib/api";

export default function DataPage() {
  const fetcher = (url: string) => apiFetch<any>(url);
  const { data, isLoading } = useSWR(endpoints.dataSample(), fetcher);
  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">Data Sample</h2>
      <div className="card text-xs overflow-auto max-h-[600px]">
        {isLoading && "Loading..."}
        <pre>{data ? JSON.stringify(data, null, 2) : null}</pre>
      </div>
    </div>
  );
}
