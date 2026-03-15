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

    from web.model_manager import MODEL_REGISTRY

    print("\n  1386.ai")
    print()

    found = False
    for model_id, info in MODEL_REGISTRY.items():
        if info["checkpoint"].exists():
            size = info["checkpoint"].stat().st_size / 1e6
            print(f"    {info['name']} ({info['params']}) loaded")
            found = True
        else:
            print(f"    {info['name']} ({info['params']}) not found")

    if not found:
        print("\n  No models found. Train one first:")
        print("    python scripts/run_1.1.py")
        sys.exit(1)

    from web.app import app  # noqa: F401

    port = 8000
    print(f"\n  http://localhost:{port}")
    print(f"  Ctrl+C to stop.\n")

    import threading
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    uvicorn.run("web.app:app", host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
