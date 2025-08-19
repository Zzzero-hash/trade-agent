"use client";
import { useEventSource } from "@/hooks/useEventSource";
import { endpoints } from "@/lib/api";
import { useEffect, useRef, useState } from "react";

interface LogLine {
  ts: number;
  level: string;
  message: string;
}

export default function LiveLog() {
  const { data } = useEventSource<LogLine>(endpoints.logsStream());
  const [lines, setLines] = useState<LogLine[]>([]);
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (data) setLines((l) => [...l.slice(-499), data]);
  }, [data]);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [lines]);
  return (
    <div className="card h-80 flex flex-col">
      <h3 className="text-sm mb-2 text-neutral-400">Live Logs</h3>
      <div
        ref={ref}
        className="flex-1 overflow-auto font-mono text-xs space-y-0.5 pr-2"
      >
        {lines.map((l, i) => (
          <div key={i} className="whitespace-pre">
            <span className={levelColor(l.level)}>
              {new Date(l.ts).toLocaleTimeString()} {l.level.padEnd(5)}
            </span>{" "}
            {l.message}
          </div>
        ))}
      </div>
    </div>
  );
}

function levelColor(level: string) {
  switch (level) {
    case "ERROR":
      return "text-rose-400";
    case "WARN":
      return "text-amber-300";
    case "INFO":
      return "text-neutral-300";
    case "DEBUG":
      return "text-neutral-500";
    default:
      return "text-neutral-400";
  }
}
