# Trade Agent Frontend

Next.js (App Router) dashboard for the trade-agent FastAPI backend.

## Features

- Dashboard with live metrics (SSE) & logs (SSE)
- Runs table
- Data sample viewer
- Backtest trigger
- Settings page
- KPI cards & responsive layout
- TypeScript, TailwindCSS, SWR, Recharts
- Central API wrapper with toast-based error handling

## Getting Started

1. Copy env file:
   ```bash
   cp .env.local.example .env.local
   ```
2. Adjust `NEXT_PUBLIC_API_BASE_URL` if needed (defaults to FastAPI at 8000).
3. Install deps & run:
   ```bash
   npm install
   npm run dev
   ```
4. Open http://localhost:3000

## Production Build

```bash
npm run build
npm start
```

Deploy via any Next.js compatible platform (Vercel, Docker). For Docker you can create a multistage build using `next build` then `next start` (or output static if later adapted for static export scenario).

## Expected Backend Endpoints

| Path            | Method    | Notes                            |
| --------------- | --------- | -------------------------------- |
| /health         | GET       | uptime + status                  |
| /runs           | GET       | list run objects                 |
| /metrics/stream | GET (SSE) | events: `{ t, value }`           |
| /logs/stream    | GET (SSE) | events: `{ ts, level, message }` |
| /data/sample    | GET       | sample data preview              |
| /backtest       | POST      | trigger backtest job             |

Adjust `endpoints` map in `lib/api.ts` to match real backend paths.

## Notes

SSE uses the browser native EventSource. If auth (e.g. JWT) is needed, switch to a polyfill supporting headers or use a proxy route.
