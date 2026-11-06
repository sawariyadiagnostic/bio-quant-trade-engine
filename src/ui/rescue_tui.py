"""Companion Headless Terminal Rescue TUI.

Provides a standalone text-based avionics cockpit running directly in Windows PowerShell /
terminal via Rich, decoupled from any browser or HTTP server.
Features direct SharedMemory IPC telemetry display, live SQLite WAL champion leaderboard,
and single-keystroke [K] panic liquidation trigger via Win32 msvcrt.
"""
from __future__ import annotations
import asyncio
import io
import sys
import time
from typing import Any, Optional
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from loguru import logger

from src.evolution.archive_store import ArchiveStore
from src.pipeline.shared_telemetry import SharedMemoryBus


class RescueTUI:
    """Headless Terminal Rescue Cockpit & Emergency Controller."""

    def __init__(
        self,
        shm_bus: Optional[SharedMemoryBus] = None,
        sentry: Any = None,
        console: Optional[Console] = None,
        db_path: str = "audit/alpha_elites.db"
    ):
        self.shm_bus = shm_bus or SharedMemoryBus(create=False)
        self.sentry = sentry
        self.console = console or Console()
        self.db_path = db_path
        self.archive_store = ArchiveStore(self.db_path)

    def render_snapshot(self) -> str:
        """Renders rich ASCII terminal avionics HUD and returns string."""
        buffer = io.StringIO()
        out_console = Console(file=buffer, force_terminal=True, width=120)

        hdr = self.shm_bus.header if self.shm_bus else None
        cur_gen = int(hdr.current_gen) if hdr else 0
        tot_gen = int(hdr.total_gens) if hdr else 10
        residents = int(hdr.resident_count) if hdr else 0
        sharpe = float(hdr.top_sharpe) if hdr else 0.0
        ret_pct = float(hdr.best_net_return) if hdr else 0.0
        aborted = int(hdr.abort_requested) if hdr else 0

        # 1. Header HUD Table
        hud_table = Table(title="[bold blue]=== BIO-QUANT TERMINAL RESCUE TUI ===[/bold blue]", border_style="blue")
        hud_table.add_column("Generation", justify="center", style="cyan")
        hud_table.add_column("Champions", justify="center", style="green")
        hud_table.add_column("Max Sharpe", justify="center", style="yellow")
        hud_table.add_column("Best Net Return", justify="center", style="magenta")
        hud_table.add_column("Abort Signal", justify="center", style="red")

        hud_table.add_row(
            f"{cur_gen}/{tot_gen}",
            str(residents),
            f"{sharpe:.2f}",
            f"{ret_pct:.1f}%",
            "[bold red]HALTED[/bold red]" if aborted else "[green]NORMAL[/green]"
        )
        out_console.print(hud_table)

        # 2. Resident Champions Leaderboard Table
        top_elites = self.archive_store.get_top_elites(limit=5, metric="fitness")
        elites_table = Table(title="[bold green]Top Resident Champions (SQLite WAL)[/bold green]", border_style="bright_black")
        elites_table.add_column("Ticker", style="cyan", width=12)
        elites_table.add_column("Sharpe", style="yellow", width=10)
        elites_table.add_column("Sortino", style="green", width=10)
        elites_table.add_column("CAGR %", style="magenta", width=10)
        elites_table.add_column("Formula Expression", style="white")

        if top_elites:
            for e in top_elites:
                elites_table.add_row(
                    str(e.get("ticker", "NIFTYBEES")),
                    f"{float(e.get('fitness', 0.0)):.2f}",
                    f"{float(e.get('sortino', 0.0)):.2f}",
                    f"{(float(e.get('cagr_pct', 0.0)) * 100):.1f}%",
                    str(e.get("formula_str", ""))[:50]
                )
        else:
            elites_table.add_row("NIFTYBEES", "0.00", "0.00", "0.0%", "Awaiting evolution cycle...")
        out_console.print(elites_table)

        # 3. Hotkey Control Strip
        controls = Panel(
            "[bold white][K][/bold white] [bold red]EMERGENCY PANIC (100% CASH)[/bold red]  |  "
            "[bold white][A][/bold white] [bold yellow]ABORT WORKER[/bold yellow]  |  "
            "[bold white][Q][/bold white] [dim]QUIT RESCUE TUI[/dim]",
            title="[bold yellow]Tactical Hotkeys (msvcrt non-blocking)[/bold yellow]",
            border_style="yellow"
        )
        out_console.print(controls)

        return buffer.getvalue()

    def handle_key(self, key: str) -> bool:
        """Processes single keystroke non-blocking input."""
        k = key.lower()
        if k == 'k':
            self.console.print("\n[bold red][!] [K] PRESSED: TRIGGERING EMERGENCY PANIC PROTOCOL...[/bold red]")
            if self.sentry:
                self.sentry.freeze_stage1()
                try:
                    res = self.sentry.execute_stage2_liquidate()
                    if asyncio.iscoroutine(res):
                        asyncio.run(res)
                except Exception as e:
                    self.console.print(f"[red]Liquidation dispatch error: {e}[/red]")
            self.console.print("[bold green][*] EMERGENCY LIQUIDATION DISPATCHED TO 100% CASH.[/bold green]\n")
            return True

        if k == 'a':
            self.console.print("\n[bold yellow][!] [A] PRESSED: SIGNALING ATOMIC WORKER ABORT...[/bold yellow]")
            if self.shm_bus:
                self.shm_bus.request_abort()
            self.console.print("[bold green][*] ABORT SIGNAL POSTED TO SHARED MEMORY.[/bold green]\n")
            return True

        if k == 'q':
            self.console.print("[dim]Quitting Rescue TUI...[/dim]")
            return False

        return True

    def run_cli_loop(self) -> None:
        """Active live polling loop in terminal with msvcrt hotkeys."""
        try:
            import msvcrt
            has_msvcrt = True
        except ImportError:
            has_msvcrt = False

        self.console.clear()
        self.console.print("[bold blue]Starting Terminal Rescue Cockpit... Press [Q] to quit.[/bold blue]")

        while True:
            # Poll keyboard
            if has_msvcrt and msvcrt.kbhit():
                char = msvcrt.getch().decode("utf-8", errors="ignore")
                keep_going = self.handle_key(char)
                if not keep_going:
                    break

            # Render frame
            snapshot = self.render_snapshot()
            self.console.print(snapshot)
            time.sleep(0.5)


if __name__ == "__main__":
    tui = RescueTUI()
    tui.run_cli_loop()
