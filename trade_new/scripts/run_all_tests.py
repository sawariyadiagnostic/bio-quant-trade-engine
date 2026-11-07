"""
Institutional Test Runner for Sovereign Bio-Quant System (trade_new).
Executes automated test discovery across trade_new/tests, validates invariants,
and asserts zero-regression baseline with exit code 0 on success, 1 on failure.
"""
import os
import sys
import time
import unittest


def main() -> int:
    start_time = time.perf_counter()

    # Normalize base directory to trade_new
    script_dir = os.path.dirname(os.path.abspath(__file__))
    trade_new_dir = os.path.abspath(os.path.join(script_dir, ".."))
    test_dir = os.path.join(trade_new_dir, "tests")

    # Ensure trade_new is first on sys.path
    if trade_new_dir not in sys.path:
        sys.path.insert(0, trade_new_dir)

    print("=" * 80)
    print("       BIO-QUANT SOVEREIGN TEST RUNNER (trade_new)")
    print(f"       Root: {trade_new_dir}")
    print(f"       Test Directory: {test_dir}")
    print("=" * 80)

    loader = unittest.defaultTestLoader
    suite = loader.discover(start_dir=test_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    elapsed = time.perf_counter() - start_time
    total = result.testsRun
    errors = len(result.errors)
    failures = len(result.failures)
    skipped = len(result.skipped)
    passed = total - (errors + failures + skipped)

    print("\n" + "=" * 80)
    print("TEST EXECUTION SUMMARY:")
    print(f"  Total Tests Executed : {total}")
    print(f"  Passed               : {passed}")
    print(f"  Failures             : {failures}")
    print(f"  Errors               : {errors}")
    print(f"  Skipped              : {skipped}")
    print(f"  Elapsed Time         : {elapsed:.3f}s")
    print("=" * 80)

    if result.wasSuccessful():
        print("VERDICT: ALL TESTS PASSED [100% GREEN - ZERO BUFFER CRASHES]")
        return 0
    else:
        print("VERDICT: TEST SUITE FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
