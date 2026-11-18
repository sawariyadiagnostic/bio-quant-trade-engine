"""Launch the RAVANA Alpha Cockpit Server on Port 7860 and open browser."""
import webbrowser
import uvicorn
from ravana.ui.server import app

if __name__ == "__main__":
    print("Launching RAVANA Sovereign Alpha Cockpit on http://localhost:7860...")
    webbrowser.open("http://localhost:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860)
