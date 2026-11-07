"""
Strategy Snapshot & Reproducibility Vault.
Permanently snapshots elite quantitative formulas, metrics, and parameters
bound to exact Git commit hashes so profitable changes are never lost during experimentation.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Ensure UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def get_current_git_commit() -> str:
    """Retrieves current Git commit hash and dirty status."""
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        commit = res.stdout.strip()
        status_res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        is_dirty = bool(status_res.stdout.strip())
        return f"{commit}-dirty" if is_dirty else commit
    except Exception:
        return "unversioned"

def load_archive_files() -> List[Dict[str, Any]]:
    """Loads all elite strategies from persistent audit archives."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "audit"))
    if not os.path.exists(base_dir):
        return []
    
    entries = []
    for fname in os.listdir(base_dir):
        if fname.startswith("archive_") and fname.endswith(".json"):
            fpath = os.path.join(base_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        entries.extend(data)
                    elif isinstance(data, dict):
                        entries.append(data)
            except Exception:
                continue
    return entries

def save_strategy_snapshot(tag: str, notes: str = "") -> Optional[str]:
    """Saves a permanent snapshot bundle of all current elite strategies."""
    elites = load_archive_files()
    git_hash = get_current_git_commit()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_tag = tag.strip().replace(" ", "_") or "unlabeled"

    vault_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "strategies", "vault"))
    os.makedirs(vault_dir, exist_ok=True)

    snapshot_data = {
        "snapshot_id": f"STRAT_{timestamp}_{clean_tag}",
        "tag": clean_tag,
        "notes": notes,
        "git_commit": git_hash,
        "created_at": datetime.now().isoformat(),
        "total_elites": len(elites),
        "strategies": elites
    }

    out_file = os.path.join(vault_dir, f"snapshot_{timestamp}_{clean_tag}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(snapshot_data, f, indent=2)

    return out_file

def list_snapshots() -> None:
    """Lists all historical strategy snapshots in the vault."""
    vault_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "strategies", "vault"))
    if not os.path.exists(vault_dir):
        print("Vault is empty. No snapshots recorded yet.")
        return

    print("=" * 70)
    print("🏛️ STRATEGY REPRODUCIBILITY VAULT")
    print("=" * 70)
    for fname in sorted(os.listdir(vault_dir), reverse=True):
        if fname.endswith(".json"):
            fpath = os.path.join(vault_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    sid = meta.get("snapshot_id", fname)
                    commit = meta.get("git_commit", "unknown")
                    count = meta.get("total_elites", 0)
                    tag = meta.get("tag", "")
                    created = meta.get("created_at", "")[:19]
                    print(f"📦 {sid} | Git: [{commit}] | Elites: {count:2d} | Created: {created}")
            except Exception:
                continue
    print("=" * 70 + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy Snapshot & Versioning Vault")
    parser.add_argument("--save", action="store_true", help="Save current MAP-Elites formulas into snapshot vault")
    parser.add_argument("--tag", type=str, default="baseline", help="Descriptive tag for this strategy snapshot")
    parser.add_argument("--notes", type=str, default="", help="Optional notes on market regime, parameters, or test results")
    parser.add_argument("--list", action="store_true", help="List all preserved historical strategy snapshots")
    args = parser.parse_args()

    if args.list:
        list_snapshots()
        return

    saved_file = save_strategy_snapshot(args.tag, args.notes)
    print("\n" + "=" * 70)
    print("✅ STRATEGY SNAPSHOT PERMANENTLY PRESERVED")
    print("=" * 70)
    print(f"Vault File:  {saved_file}")
    print(f"Git Commit:  {get_current_git_commit()}")
    print("You can freely experiment, mutate, or reset the search grid now.")
    print("This strategy configuration can be restored at any time.\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    main()
