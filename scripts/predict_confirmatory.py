#!/usr/bin/env python3
"""Standalone entry point; CUDA must use the project module launcher."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from arggnn.confirmatory_prediction_runner import main
if __name__ == "__main__":
    main()
