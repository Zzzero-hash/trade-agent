"use client";
import { useState } from "react";
import { apiFetch, endpoints } from "@/lib/api";
import toast from "react-hot-toast";

export default function BacktestPage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  async function runBacktest() {
    setLoading(true);
    try {
      const data = await apiFetch(endpoints.backtest(), {
        method: "POST",
        body: JSON.stringify({}),
      });
      setResult(data);
      toast.success("Backtest started");
    } catch (e) {
      /* handled */
    }
    setLoading(false);
  }
  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">Backtest</h2>
      <button
        disabled={loading}
        onClick={runBacktest}
        className="px-4 py-2 bg-brand-600 rounded disabled:opacity-50 text-sm"
      >
        {loading ? "Running..." : "Run Backtest"}
      </button>
      {result && (
        <div className="card text-xs overflow-auto max-h-[600px]">
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
