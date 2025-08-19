import "./globals.css";
import { ReactNode } from "react";
import { Toaster } from "react-hot-toast";
import Link from "next/link";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Trade Agent Dashboard",
  description: "Monitoring & control UI for trade-agent backend",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen flex">
        <aside className="w-56 bg-neutral-900 border-r border-neutral-800 p-4 space-y-4">
          <h1 className="text-lg font-semibold">Trade Agent</h1>
          <nav className="flex flex-col gap-2 text-sm">
            <Link href="/">Dashboard</Link>
            <Link href="/runs">Runs</Link>
            <Link href="/data">Data</Link>
            <Link href="/backtest">Backtest</Link>
            <Link href="/settings">Settings</Link>
          </nav>
        </aside>
        <main className="flex-1 p-6 space-y-6">{children}</main>
        <Toaster position="bottom-right" />
      </body>
    </html>
  );
}
