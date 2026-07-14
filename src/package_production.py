"""
Build the frozen production bundle from the already-trained artifacts.

This script does not retrain any model. It scans the saved model files and
metrics, chooses the best supported candidate, copies it into a stable
production filename, and writes the registry manifest.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.artifacts import build_artifact_registry, load_artifact_registry


def main():
    registry = build_artifact_registry(force=True)
    print(json.dumps({
        "production_model": registry.get("production", {}).get("name"),
        "production_bundle": registry.get("production", {}).get("production_bundle_path"),
        "artifact_registry": registry.get("production", {}).get("sha256"),
        "candidates": len(registry.get("candidates", [])),
    }, indent=2))


if __name__ == "__main__":
    main()
