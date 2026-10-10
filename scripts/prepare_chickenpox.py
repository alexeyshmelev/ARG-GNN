"""Prepare the public CC-BY-4.0 Chickenpox archive. No model assessment."""
from __future__ import annotations
import argparse
import json
from arggnn.chickenpox import ForecastProtocol, prepare_chickenpox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/chickenpox")
    parser.add_argument("--output-dir", default="data/processed/chickenpox")
    parser.add_argument("--history", type=int, default=52)
    args = parser.parse_args()
    result = prepare_chickenpox(args.raw_dir, args.output_dir,
                               ForecastProtocol(history=args.history))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
