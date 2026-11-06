"""src/ui/evolution_dashboard.py
Institutional Decoupled Cockpit Launcher (Migration Stub).

DEPRECATION NOTICE:
The legacy single-threaded Gradio UI has been completely replaced by the
decoupled Tri-Process Institutional Cockpit:
- Process A: FastAPI Web Cockpit (Port 7860) with differential SSE
- Process B: Spawned Compute Worker (Cores 2-3) with SharedMemory IPC
- Process C: Out-of-band Risk Sentry (Cores 0-1) with sub-0.5ms freeze

Running `python -m src.ui.evolution_dashboard` seamlessly launches the high-performance
FastAPI ASGI Cockpit on http://127.0.0.1:7860.
"""
from __future__ import annotations
import sys
import uvicorn
from loguru import logger


def launch_dashboard(host: str = "127.0.0.1", port: int = 7860) -> None:
    """Launches the modern decoupled FastAPI cockpit on port 7860."""
    logger.info("=" * 80)
    logger.info("⚡ BIO-QUANT INSTITUTIONAL DECOUPLED ALPHA COCKPIT ⚡")
    logger.info("MIGRATION NOTICE: Legacy Gradio upgraded to Tri-Process FastAPI Cockpit.")
    logger.info(f"Serving Cockpit UI on: http://{host}:{port}")
    logger.info("=" * 80)
    uvicorn.run("src.ui.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    launch_dashboard()
