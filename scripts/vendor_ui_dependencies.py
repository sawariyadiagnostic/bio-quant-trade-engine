"""Vendors frontend JavaScript and CSS dependencies for air-gapped offline cockpit operation.

Downloads:
1. Alpine.js (reactive state engine)
2. AG Grid Community JS & CSS (virtualized table engine)
3. Plotly.js Basic (WebGL fast chart engine)
4. Tailwind CDN script
"""
import os
import urllib.request
from loguru import logger

VENDOR_DIR = os.path.abspath("src/ui/static/vendor")

ASSETS = {
    "alpine.min.js": "https://cdn.jsdelivr.net/npm/alpinejs@3.13.5/dist/cdn.min.js",
    "ag-grid-community.min.js": "https://cdn.jsdelivr.net/npm/ag-grid-community@31.0.1/dist/ag-grid-community.min.js",
    "ag-grid.min.css": "https://cdn.jsdelivr.net/npm/ag-grid-community@31.0.1/styles/ag-grid.min.css",
    "ag-theme-alpine.min.css": "https://cdn.jsdelivr.net/npm/ag-grid-community@31.0.1/styles/ag-theme-alpine.min.css",
    "plotly.min.js": "https://cdn.plot.ly/plotly-basic-2.29.1.min.js",
    "tailwind.min.js": "https://cdn.tailwindcss.com"
}


def vendor_all() -> bool:
    os.makedirs(VENDOR_DIR, exist_ok=True)
    success = True
    for fname, url in ASSETS.items():
        out_path = os.path.join(VENDOR_DIR, fname)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            logger.info(f"[Vendor] Already exists: {fname} ({os.path.getsize(out_path)} bytes)")
            continue
        try:
            logger.info(f"[Vendor] Downloading {fname} from {url}...")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read()
                with open(out_path, "wb") as f:
                    f.write(content)
            logger.success(f"[Vendor] Saved {fname} ({len(content):,} bytes).")
        except Exception as e:
            logger.warning(f"[Vendor] Failed to download {fname}: {e}. Local fallback will be used.")
            success = False
    return success


if __name__ == "__main__":
    vendor_all()
