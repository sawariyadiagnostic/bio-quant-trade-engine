---
name: institutional-cockpit-ergonomics
description: UX ergonomics, data visualization performance, and cockpit design standards modeled on Bloomberg Professional Terminal and TradingView. Use when designing, auditing, or upgrading financial dashboards.
---

# Institutional Cockpit Ergonomics Protocol (Bloomberg & TradingView Standard)

This skill guides the design, performance benchmarking, and sensory optimization of mission-critical trading cockpits.

## 1. Visual Ergonomics & Cognitive Load

Financial operators monitor dense telemetry under stress. The UI must optimize for pattern recognition:
1. **Dark Obsidian Palette:** Base background `#060913` to `#0a0f1d`. Avoid pure black `#000000` (causes high eye fatigue) or bright greys.
2. **Tabular Numerals:** Always apply `font-variant-numeric: tabular-nums` or monospace font family (`ui-monospace`, `Menlo`, `Consolas`) to all currency values, P&L, timestamps, and percentages so digits do not jump horizontally during live streaming.
3. **Semantic Status Colors:**
   - Emerald (`#10b981`): Profit, active order filled, system armed.
   - Cyan (`#06b6d4`): Cash buffer, liquid balance, telemetry sync.
   - Amber (`#f59e0b`): Mean-reversion active, trailing stop warning, rate limit caution.
   - Rose (`#f43f5e`): Drawdown breach, panic sentry armed, order rejection.

## 2. Rendering Performance (60 FPS Standard)

1. **Canvas vs. DOM for Matrix Views:**
   - If rendering $>100$ items with dynamic hover states (e.g. 20×20 MAP-Elites = 400 cells), prefer single HTML5 Canvas or optimize DOM to prevent layout thrashing.
   - Avoid creating new Chart.js instances inside polling intervals. Use `chart.data = ...; chart.update('none');` to prevent memory leaks and GC stutters.
2. **Offline-First & Air-Gapped Dependencies:**
   - All core scripts (`chart.umd.min.js`, `lucide.min.js`, `alpine.min.js`, `tailwind.min.js`) must be served locally from `/vendor/`.
   - Never rely on public CDNs for real-time institutional interfaces.

## 3. Telemetry Freshness & Audit Transparency

1. **Heartbeat Indicator:** Display a pulsing live timestamp (`SYNC: HH:MM:SS`) updated on every WebSocket or poll tick.
2. **Zero Ambiguity Disclosures:** Every position card must display Entry Price, Current MTM Price, Unrealized P&L, Target Price, Trailing Stop, and Allocated Capital.
3. **One-Click Emergency Controls:** Hardware/socket kill-switch (`FREEZE`, `PANIC TO CASH`) must be visible in the sticky master avionics header.
