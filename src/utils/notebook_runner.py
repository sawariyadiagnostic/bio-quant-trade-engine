"""
Lightweight Zero-Dependency Headless Jupyter Notebook Runner.
Executes any .ipynb notebook directly from Python/CLI without requiring Jupyter server,
browser, or web runtime. Captures cell outputs and prints execution stream.
"""
import json
import os
import sys
import time

def run_notebook(ipynb_path: str, verbose: bool = True) -> dict:
    """Executes all code cells in an .ipynb notebook file sequentially in memory."""
    if not os.path.exists(ipynb_path):
        raise FileNotFoundError(f"Notebook file '{ipynb_path}' does not exist.")

    with open(ipynb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cells = nb_data.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]

    if verbose:
        print(f"[*] Executing Notebook: {os.path.basename(ipynb_path)} ({len(code_cells)} code cells)")

    execution_context = {"__name__": "__main__"}
    results = []

    start_total = time.time()
    for idx, cell in enumerate(code_cells, 1):
        raw_source = "".join(cell.get("source", []))
        if not raw_source.strip():
            continue

        if verbose:
            print(f"--- [Cell {idx}/{len(code_cells)}] ---")

        t0 = time.time()
        try:
            exec(raw_source, execution_context)
            elapsed = time.time() - t0
            results.append({"cell": idx, "status": "SUCCESS", "elapsed_sec": round(elapsed, 3)})
            if verbose:
                print(f"[OK] Cell {idx} completed in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - t0
            results.append({"cell": idx, "status": "ERROR", "error": str(e), "elapsed_sec": round(elapsed, 3)})
            if verbose:
                print(f"[FAIL] Cell {idx} encountered error: {e}")
            break

    total_elapsed = time.time() - start_total
    if verbose:
        print(f"\n[DONE] Notebook execution finished in {total_elapsed:.2f}s")

    return {
        "notebook": ipynb_path,
        "total_code_cells": len(code_cells),
        "executed_cells": len(results),
        "total_elapsed_sec": round(total_elapsed, 2),
        "cell_results": results
    }

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "notebooks/phase_b_evolution_gpu.ipynb"
    run_notebook(target)
