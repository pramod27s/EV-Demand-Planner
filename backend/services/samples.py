"""Built-in sample dataset offered in the Workspace (one click instead of an upload).

Only Palo Alto — the network the model was trained on — is built in. The other cities'
files in data/sample/ (Boulder, Cary, Leeds, Hyderabad) can be uploaded via "My data".
"""
from __future__ import annotations

from pathlib import Path

# id -> (title, kind, file). "results:" files come from the exported artifacts,
# the others from data/sample (created by scripts/fetch_city_samples.py).
SAMPLES = {
    "palo_alto": ("Palo Alto, USA", "sessions", "results:sample_upload.csv"),
}


def sample_path(sample_id: str, api_dir: Path, sample_dir: Path) -> Path | None:
    if sample_id not in SAMPLES:
        return None
    file = SAMPLES[sample_id][2]
    path = api_dir / file.split(":", 1)[1] if file.startswith("results:") else sample_dir / file
    return path if path.exists() else None


def available(api_dir: Path, sample_dir: Path) -> list[dict]:
    return [{"id": sid, "title": title, "kind": kind}
            for sid, (title, kind, _) in SAMPLES.items() if sample_path(sid, api_dir, sample_dir)]
