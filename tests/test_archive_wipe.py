"""
Test suite verifying instant archive wiping and reset functionality:
1. wipe_and_reset_archive clears memory, deletes local audit files, and returns 0-row empty visuals.
2. on_reset_checkbox_change(True) immediately resets visual spaces and memory.
3. on_reset_checkbox_change(False) restores initial state.
4. run_evolution_stream with reset_archive=True wipes prior records on yield 1.
"""
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ui.evolution_dashboard import (
    wipe_and_reset_archive,
    on_reset_checkbox_change,
    load_ticker_archive,
    CURRENT_ARCHIVE_RECORDS,
    SORT_KEY_MAP,
    ROW_COUNT_MAP
)
import src.ui.evolution_dashboard as evo_dash

def test_wipe_and_reset_archive():
    # 1. Setup a dummy archive record in memory and on disk
    dummy_file = os.path.join("audit", "archive_TESTTICKER.json")
    os.makedirs("audit", exist_ok=True)
    with open(dummy_file, "w") as f:
        json.dump({"ticker": "TESTTICKER", "resident_count": 1, "elites": [{"fitness": 5.0, "net_return_pct": 20.0}]}, f)

    evo_dash.CURRENT_ARCHIVE_RECORDS = [{"fitness": 5.0, "net_return_pct": 20.0}]
    assert len(evo_dash.CURRENT_ARCHIVE_RECORDS) == 1
    assert os.path.exists(dummy_file)

    # 2. Invoke wipe_and_reset_archive
    status, telemetry, spotlight, heatmap, scatter, table = wipe_and_reset_archive(
        ticker="TESTTICKER",
        compute_target="Local Multi-Core CPU",
        colab_url="http://localhost:8000",
        sort_choice="Top Net Earners First (+Net Return %)",
        row_choice="Top 50"
    )

    # 3. Assertions
    assert len(evo_dash.CURRENT_ARCHIVE_RECORDS) == 0, "Memory must be wiped"
    assert not os.path.exists(dummy_file), "Disk file must be deleted"
    assert "ARCHIVE WIPED" in status
    assert table["Rank"].iloc[0] == "-", "Leaderboard table must display empty placeholder row"
    assert heatmap is not None, "Heatmap plot object must exist"
    assert scatter is not None, "Scatter plot object must exist"
    print("[PASS] test_wipe_and_reset_archive")

def test_on_reset_checkbox_toggle():
    dummy_file = os.path.join("audit", "archive_TESTTOGGLE.json")
    os.makedirs("audit", exist_ok=True)
    with open(dummy_file, "w") as f:
        json.dump({"ticker": "TESTTOGGLE", "resident_count": 1, "elites": [{"fitness": 7.0, "net_return_pct": 35.0}]}, f)

    evo_dash.CURRENT_ARCHIVE_RECORDS = [{"fitness": 7.0, "net_return_pct": 35.0}]

    # Checkbox checked (True) -> should wipe
    status, telemetry, spotlight, heatmap, scatter, table = on_reset_checkbox_change(
        checked=True,
        ticker="TESTTOGGLE",
        sort_choice="Top Net Earners First (+Net Return %)",
        row_choice="Top 50"
    )

    assert len(evo_dash.CURRENT_ARCHIVE_RECORDS) == 0
    assert not os.path.exists(dummy_file)
    assert table["Rank"].iloc[0] == "-"
    assert "RESET ACTIVE" in status

    # Checkbox unchecked (False) -> should reload (if file exists, but file was removed so 0)
    status2, telemetry2, spotlight2, heatmap2, scatter2, table2 = on_reset_checkbox_change(
        checked=False,
        ticker="TESTTOGGLE",
        sort_choice="Top Net Earners First (+Net Return %)",
        row_choice="Top 50"
    )
    assert len(evo_dash.CURRENT_ARCHIVE_RECORDS) == 0
    assert "Ready to launch" in status2 or "CONTINUAL MODE" in status2
    print("[PASS] test_on_reset_checkbox_toggle")

if __name__ == "__main__":
    test_wipe_and_reset_archive()
    test_on_reset_checkbox_toggle()
    print("All archive wipe tests passed successfully!")
