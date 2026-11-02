"""
Colab Bridge Client (Remote GPU Compute Gateway).
Connects the local WebUI and CLI to an active Google Colab GPU Worker via secure HTTP/Tunnel.
Streams generation-by-generation evolution metrics in real-time.
"""
import time
import json
from typing import Generator, Dict, Any, Optional
import requests
from loguru import logger

class ColabBridgeClient:
    """Client for dispatching evolutionary workloads to Google Colab Cloud GPU."""

    @staticmethod
    def clean_url(base_url: str) -> str:
        """Strips whitespace, quotes, and trailing slashes from URL."""
        url = base_url.strip().strip("'").strip('"')
        if url.endswith("/"):
            url = url[:-1]
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url
        return url

    @classmethod
    def check_health(cls, base_url: str) -> Dict[str, Any]:
        """
        Tests connection to Google Colab GPU Worker.
        Returns health status, GPU hardware model, VRAM, and round-trip latency.
        """
        target = cls.clean_url(base_url)
        endpoint = f"{target}/health"
        start_t = time.time()
        try:
            resp = requests.get(endpoint, timeout=8.0)
            latency_ms = round((time.time() - start_t) * 1000.0, 1)
            if resp.status_code == 200:
                data = resp.json()
                data["online"] = True
                data["latency_ms"] = latency_ms
                data["base_url"] = target
                return data
            else:
                return {
                    "online": False,
                    "error": f"Colab returned HTTP status {resp.status_code}: {resp.text}",
                    "latency_ms": latency_ms
                }
        except requests.exceptions.Timeout:
            return {"online": False, "error": f"Connection timed out after 8s connecting to {endpoint}."}
        except requests.exceptions.ConnectionError as ce:
            return {"online": False, "error": f"Could not reach Colab Worker at {endpoint}. Ensure tunnel is running: {ce}"}
        except Exception as e:
            return {"online": False, "error": str(e)}

    @classmethod
    def stream_evolution(cls, base_url: str, job_params: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """
        Dispatches evolution job to Google Colab GPU Worker.
        Streams NDJSON progress events (generation telemetry, top resident elites) in real-time.
        """
        target = cls.clean_url(base_url)
        endpoint = f"{target}/evolve"
        logger.info(f"Dispatching remote evolution to Google Colab: {endpoint} on {job_params.get('ticker')}")

        try:
            with requests.post(endpoint, json=job_params, stream=True, timeout=600.0) as response:
                if response.status_code != 200:
                    yield {
                        "event": "error",
                        "message": f"Colab Worker returned HTTP {response.status_code}: {response.text}"
                    }
                    return

                for line in response.iter_lines():
                    if line:
                        decoded = line.decode("utf-8").strip()
                        if decoded:
                            try:
                                payload = json.loads(decoded)
                                yield payload
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to decode Colab streaming line: {decoded}")
        except Exception as e:
            logger.error(f"Remote Colab evolution connection dropped: {e}")
            yield {
                "event": "error",
                "message": f"Connection to Colab dropped during evolution: {e}"
            }

    @classmethod
    def fetch_archive(cls, base_url: str, ticker: str, lookback: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Downloads full 20x20 MAP-Elites archive JSON from Colab worker."""
        target = cls.clean_url(base_url)
        endpoint = f"{target}/archive/{ticker}"
        params = {"lookback": lookback} if lookback else {}
        try:
            resp = requests.get(endpoint, params=params, timeout=15.0)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"Could not fetch remote archive for {ticker}: {e}")
        return None

    @classmethod
    def reset_archive(cls, base_url: str, ticker: str, lookback: Optional[str] = None) -> bool:
        """Sends reset signal to remote Colab worker to wipe archive for ticker."""
        target = cls.clean_url(base_url)
        endpoint = f"{target}/archive/{ticker}/reset"
        params = {"lookback": lookback} if lookback else {}
        try:
            resp = requests.post(endpoint, params=params, timeout=8.0)
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"Could not reset remote archive for {ticker}: {e}")
            return False
