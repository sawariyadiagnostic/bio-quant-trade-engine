"""
Interactive Live Terminal Evolution Studio (Phase B).
Provides real-time animated visual tracking of Epigenetic MAP-Elites Evolution.
Features:
- Configurable Generations, Population Size, and Target Asset from Data Lake.
- Injection of Custom Seed Formulas & Classical Quantitative Alpha Libraries.
- Live updating Rich dashboard with generation progress, Thymic Chamber casualties,
  MAP-Elites 20x20 archive occupancy, and real-time elite leaderboard.
"""
import os
import sys
import argparse
import time
from typing import Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from src.domain.models import InstrumentType
from src.domain.formula_parser import FormulaParser, AlphaSeedLibrary
from src.manifold.lake_dataset_loader import LakeDatasetLoader
from src.evolution.genetic_engine import GeneticEvolutionEngine

console = Console(force_terminal=True, legacy_windows=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Bio-Quant Live Epigenetic Evolution Studio")
    parser.add_argument("--ticker", type=str, default="NIFTYBEES", help="Asset ticker from data lake (e.g. NIFTYBEES, INFY, TRENT)")
    parser.add_argument("--generations", type=int, default=10, help="Number of evolutionary generations (default: 10)")
    parser.add_argument("--pop-size", type=int, default=50, help="Population size / formulas in space (default: 50)")
    parser.add_argument("--warm-start", action="store_true", default=True, help="Warm-start population with classical quantitative seeds")
    parser.add_argument("--add-formula", type=str, nargs="*", default=[], help="Custom formula strings to inject into the gene pool")
    parser.add_argument("--target-mode", type=str, default="overnight", choices=["overnight", "intraday"], help="Target return horizon")
    return parser.parse_args()

def build_dashboard_table(gen: int, total_gens: int, stats: dict, top_elites: list, ticker: str, pop_size: int) -> Panel:
    """Renders real-time multi-panel telemetry view."""
    # Summary Table
    meta_table = Table(title=f"[GENOME] Epigenetic Evolution Studio: {ticker} (Pop: {pop_size})", expand=True, border_style="cyan")
    meta_table.add_column("Generation", justify="center", style="bold yellow")
    meta_table.add_column("Thymic Survivors", justify="center", style="bold green")
    meta_table.add_column("New Elites", justify="center", style="magenta")
    meta_table.add_column("Total Elites", justify="center", style="bold cyan")
    meta_table.add_column("Archive Occupancy", justify="center", style="bold white")

    surv = f"{stats.get('thymic_survivors', 0)} / {stats.get('offspring_count', pop_size)}"
    occ = f"{stats.get('archive_occupancy_pct', 0.0):.2f}% (20x20 Grid)"
    meta_table.add_row(
        f"{gen} / {total_gens}",
        surv,
        str(stats.get('archive_inserted', 0)),
        str(stats.get('total_elites', 0)),
        occ
    )

    # Leaderboard Table
    lead_table = Table(title="[LEADERBOARD] Top Surviving Alpha Genomes (Net Calmar Ranked)", expand=True, border_style="green")
    lead_table.add_column("Rank", justify="center", style="bold yellow", width=6)
    lead_table.add_column("Net Calmar", justify="right", style="bold green", width=12)
    lead_table.add_column("Win Rate", justify="right", style="cyan", width=10)
    lead_table.add_column("Net Return", justify="right", style="bold magenta", width=14)
    lead_table.add_column("Turnover", justify="right", style="white", width=10)
    lead_table.add_column("Formula AST Expression", style="italic yellow")

    if top_elites:
        for idx, elite in enumerate(top_elites[:5], 1):
            lead_table.add_row(
                f"#{idx}",
                f"{elite.fitness:,.2f}",
                f"{elite.win_rate:.1f}%",
                f"+{elite.net_return_pct:,.1f}%",
                f"{elite.turnover:.1f}x",
                elite.formula.to_formula_str()
            )
    else:
        lead_table.add_row("-", "Pending...", "-", "-", "-", "Evaluating initial generation...")

    layout = Layout()
    layout.split_column(
        Layout(meta_table, size=6),
        Layout(lead_table)
    )
    return Panel(layout, border_style="blue", title="[bold white]Bio-Quant Autonomous Engine - Live Evolution Chamber[/bold white]")

def main():
    args = parse_args()

    console.print(Panel(
        f"[bold cyan]Initializing Live Evolution Session[/bold cyan]\n"
        f"* Target Instrument: [bold yellow]{args.ticker}[/bold yellow]\n"
        f"* Search Space: [bold green]{args.pop_size} formulas[/bold green] across [bold green]{args.generations} generations[/bold green]\n"
        f"* Execution Horizon: [bold magenta]{args.target_mode.upper()}[/bold magenta] (Statutory Tax Aware)\n"
        f"* Custom Formulas Injected: [bold white]{len(args.add_formula)}[/bold white]",
        border_style="cyan"
    ))

    # 1. Load Dataset
    loader = LakeDatasetLoader()
    try:
        df = loader.load_dataset(args.ticker)
        console.print(f"[green][OK] Ingested {len(df)} historical trading sessions with 21 manifold & lead features.[/green]")
    except Exception as e:
        console.print(f"[red][ERROR] Loading {args.ticker}: {e}[/red]")
        return

    # 2. Compile Warm-Start & Custom Seed Genomes
    seed_nodes = []
    if args.warm_start:
        seed_nodes.extend(AlphaSeedLibrary.load_seed_nodes())
        console.print(f"[cyan][OK] Loaded {len(seed_nodes)} classical quantitative literature seeds (WorldQuant / Trend / Volume Flow).[/cyan]")

    if args.add_formula:
        custom_compiled = []
        for f_str in args.add_formula:
            try:
                node = FormulaParser.parse(f_str)
                custom_compiled.append(node)
                console.print(f"[green][OK] Compiled custom seed: [bold yellow]{f_str}[/bold yellow][/green]")
            except Exception as pe:
                console.print(f"[red][ERROR] Failed to parse custom formula '{f_str}': {pe}[/red]")
        seed_nodes.extend(custom_compiled)

    # 3. Instantiate Engine
    instrument = InstrumentType.ETF_CASH if "BEES" in args.ticker else InstrumentType.FUTURES
    engine = GeneticEvolutionEngine(
        population_size=args.pop_size,
        archive_grid=(20, 20),
        p_crossover=0.65,
        p_mutation=0.35,
        p_methylation=0.25,
        max_depth=4,
        instrument=instrument
    )
    engine.initialize_population(seed_nodes=seed_nodes)

    # 4. Live Evolution Rendering Loop
    console.print("\n[bold yellow]Starting Live Genetic Evolution... Press Ctrl+C at any time to freeze best formulas.[/bold yellow]\n")

    latest_stats = {
        "thymic_survivors": 0,
        "offspring_count": args.pop_size,
        "archive_inserted": 0,
        "total_elites": 0,
        "archive_occupancy_pct": 0.0
    }
    top_elites = []

    with Live(build_dashboard_table(0, args.generations, latest_stats, top_elites, args.ticker, args.pop_size), refresh_per_second=4) as live:
        for gen in range(1, args.generations + 1):
            stats = engine.evolve_generation(df, current_regime=0, target_mode=args.target_mode)
            top_elites = engine.archive.get_top_elites(5)
            latest_stats = stats
            live.update(build_dashboard_table(gen, args.generations, stats, top_elites, args.ticker, args.pop_size))
            time.sleep(0.080)

    # 5. Final Summary Output
    console.print("\n[bold green][OK] Evolutionary Search Complete![/bold green]")
    resident_elites = engine.archive.get_resident_elites()
    console.print(f"Total Unique Behavioral Niches Discovered: [bold cyan]{len(resident_elites)} / 400[/bold cyan] ({engine.archive.occupancy_rate() * 100.0:.2f}% occupancy)\n")

    top_10 = engine.archive.get_top_elites(10)
    final_table = Table(title=f"[HALL OF FAME] Top 10 Evolved Alphas for {args.ticker}", border_style="gold1")
    final_table.add_column("Rank", style="bold yellow")
    final_table.add_column("Net Calmar", style="bold green", justify="right")
    final_table.add_column("Win Rate", style="cyan", justify="right")
    final_table.add_column("Net Return", style="bold magenta", justify="right")
    final_table.add_column("Turnover", style="white", justify="right")
    final_table.add_column("Formula AST", style="italic white")

    for i, e in enumerate(top_10, 1):
        final_table.add_row(
            f"#{i}",
            f"{e.fitness:,.2f}",
            f"{e.win_rate:.1f}%",
            f"+{e.net_return_pct:,.1f}%",
            f"{e.turnover:.1f}x",
            e.formula.to_formula_str()
        )
    console.print(final_table)

if __name__ == "__main__":
    main()
