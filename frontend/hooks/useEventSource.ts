import { useEffect, useRef, useState } from "react";
import { API_BASE_URL } from "@/lib/config";

export function useEventSource<T = any>(path: string | null) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!path) return;
    const url = path.startsWith("http") ? path : `${API_BASE_URL}${path}`;
    const es = new EventSource(url);
    (esRef as any).current = es;
    es.onopen = () => setOpen(true);
    es.onerror = () => {
      setError("SSE connection error");
    };
    es.onmessage = (evt) => {
      try {
        setData(JSON.parse(evt.data));
      } catch {
        /* ignore */
      }
    };
    return () => {
      es.close();
      setOpen(false);
    };
  }, [path]);

  return { data, error, open };
}
