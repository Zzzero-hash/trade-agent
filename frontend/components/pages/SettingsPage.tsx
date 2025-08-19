"use client";
import { useState } from "react";
import { API_BASE_URL } from "@/lib/config";

export default function SettingsPage() {
  const [api] = useState(API_BASE_URL);
  return (
    <div className="space-y-4 max-w-xl">
      <h2 className="text-lg font-semibold">Settings</h2>
      <div className="card space-y-2">
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-neutral-400">
            API Base URL (readonly - set via env)
          </span>
          <input
            value={api}
            readOnly
            className="bg-neutral-800 rounded px-2 py-1 text-xs"
          />
        </label>
        <p className="text-xs text-neutral-500">
          Change by editing NEXT_PUBLIC_API_BASE_URL in .env.local
        </p>
      </div>
    </div>
  );
}
