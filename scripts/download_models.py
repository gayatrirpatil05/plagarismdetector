"""Pre-download every model so the first request (and Docker start-up) is fast.

    python scripts/download_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core.models import hub  # noqa: E402

hub.preload()
for name, st in hub.status().items():
    print(f"{name:14s} {'OK ' if st['loaded'] else 'FAIL'} {st['model']}  {st['error'] or ''}")
