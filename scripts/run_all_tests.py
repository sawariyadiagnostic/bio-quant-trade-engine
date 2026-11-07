"""Autonomous test runner for bio-quant test suite."""
import importlib
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run():
    test_files = sorted([os.path.splitext(os.path.basename(f))[0] for f in glob.glob("tests/test_*.py")])
    total_passed = 0
    failed_tests = []

    print(f"Discovered {len(test_files)} test suite files.")
    for mod_name in test_files:
        try:
            mod = importlib.import_module(f"tests.{mod_name}")
            test_funcs = [getattr(mod, f) for f in dir(mod) if f.startswith("test_") and callable(getattr(mod, f))]
            for fn in test_funcs:
                fn()
                total_passed += 1
            print(f"  [PASS] tests/{mod_name}.py ({len(test_funcs)} assertions)")
        except Exception as e:
            print(f"  [FAIL] tests/{mod_name}.py: {e}")
            failed_tests.append((mod_name, str(e)))

    print("=" * 60)
    print(f"Total assertions passed: {total_passed}")
    if failed_tests:
        print(f"Failed modules: {len(failed_tests)}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED SUCCESSFULLY.")
        sys.exit(0)

if __name__ == "__main__":
    run()
