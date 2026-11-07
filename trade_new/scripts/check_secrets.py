"""
Pre-Commit Secret Scanner & Security Hygiene Validator.
Prevents accidental commits of private credentials, API secrets, and sensitive financial logs.
"""
import os
import re
import subprocess
import sys
from typing import List, Tuple

# Reconfigure stdout/stderr for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BLOCKED_PATTERNS = [
    (re.compile(r"SHOONYA_PASSWORD\s*=\s*['\"][^'\"]+['\"]"), "Shoonya Password assignment in text"),
    (re.compile(r"SHOONYA_TOTP_KEY\s*=\s*['\"][^'\"]+['\"]"), "Shoonya TOTP Secret Key assignment in text"),
    (re.compile(r"SHOONYA_SECRET_CODE\s*=\s*['\"][^'\"]+['\"]"), "Shoonya API Secret Code assignment in text"),
    (re.compile(r"(?:api[_-]?key|secret[_-]?code|access[_-]?token)\s*=\s*['\"][a-zA-Z0-9_\-]{25,}['\"]", re.IGNORECASE), "High-entropy secret token literal"),
    (re.compile(r"BEGIN\s+(?:RSA|OPENSSH|DSA|EC)?\s*PRIVATE\s+KEY", re.IGNORECASE), "Private cryptographic key block"),
]

def get_dynamic_env_secrets() -> List[Tuple[re.Pattern, str]]:
    """Loads active secrets from .env so any live credential is automatically blocked."""
    env_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    dynamic_patterns = []
    if os.path.exists(env_file):
        from dotenv import dotenv_values
        vals = dotenv_values(env_file)
        for key in ["SHOONYA_PASSWORD", "SHOONYA_TOTP_KEY", "SHOONYA_SECRET_CODE", "SHOONYA_ACCESS_TOKEN", "SHOONYA_API_KEY"]:
            secret = (vals.get(key) or "").strip()
            if secret and len(secret) >= 6 and secret not in ("YourShoonyaPasswordHere", "FA12345"):
                dynamic_patterns.append((re.compile(re.escape(secret)), f"Live credential matching {key} in .env"))
    return dynamic_patterns

BLOCKED_FILES = {
    ".env", ".env.local", ".env.production", "shoonya_paper_orders.json"
}

def get_staged_files() -> List[str]:
    """Retrieves list of files staged for commit in git."""
    try:
        res = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            check=True
        )
        return [f.strip() for f in res.stdout.splitlines() if f.strip()]
    except Exception:
        return []

def scan_file_for_secrets(filepath: str) -> List[Tuple[int, str]]:
    """Scans an individual file line by line against blocked secret patterns."""
    basename = os.path.basename(filepath)
    if basename in BLOCKED_FILES:
        return [(0, f"Critical: Direct tracking of forbidden secret file '{basename}'")]

    if not os.path.exists(filepath):
        return []

    # Skip scanning binary or example template files
    if filepath.endswith(".example") or filepath.endswith(".png") or filepath.endswith(".ico"):
        return []

    violations = []
    dynamic_patterns = get_dynamic_env_secrets()
    all_patterns = BLOCKED_PATTERNS + dynamic_patterns
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f, 1):
                for pattern, desc in all_patterns:
                    if pattern.search(line):
                        violations.append((idx, f"Line {idx}: {desc}"))
    except Exception as err:
        violations.append((0, f"Failed to read file for secret scan: {err}"))

    return violations

def run_security_audit() -> bool:
    """Audits staged git changes and fails if any secret pattern is found."""
    staged = get_staged_files()
    if not staged:
        # If nothing is staged, audit all tracked files
        try:
            res = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
            staged = [f.strip() for f in res.stdout.splitlines() if f.strip()]
        except Exception:
            staged = []

    print("=" * 65)
    print("🛡️ RUNNING PRE-COMMIT SECRET SCAN & HYGIENE AUDIT")
    print("=" * 65)
    print(f"Scanning {len(staged)} candidate file(s)...")

    has_error = False
    for path in staged:
        issues = scan_file_for_secrets(path)
        if issues:
            has_error = True
            print(f"\n❌ SECURITY VIOLATION in {path}:")
            for line_no, desc in issues:
                print(f"   • {desc}")

    if has_error:
        print("\n🚫 COMMIT REJECTED: Sensitive credentials or secret keys detected!")
        print("Please move all secrets into .env (which is .gitignored).")
        print("=" * 65 + "\n")
        return False

    print("\n✅ PASSED: No credentials, tokens, or private secrets detected.")
    print("All security gates satisfied.\n" + "=" * 65 + "\n")
    return True

if __name__ == "__main__":
    success = run_security_audit()
    sys.exit(0 if success else 1)
