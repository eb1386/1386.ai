# launch chat ui

import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def main():
    try:
        import uvicorn
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]"])
        import uvicorn

    from web.app import app  # noqa: F401

    port = 8000
    print(f"\n  1386.ai starting on http://localhost:{port}")
    print(f"  Press Ctrl+C to stop.\n")

    import threading
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    uvicorn.run("web.app:app", host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
