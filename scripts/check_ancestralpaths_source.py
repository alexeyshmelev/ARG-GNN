#!/usr/bin/env python
"""Read-only source/dependency inspection, without importing upstream scripts."""
import argparse
import json
from pathlib import Path

from arggnn.ancestralpaths_native_bridge import inspect_author_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".runtime/author_sources/ancestralpaths_77db3cfae100"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--published-source", type=Path)
    parser.add_argument("--published-archive", type=Path)
    args = parser.parse_args()
    try:
        result = inspect_author_source(args.source, archive=args.archive,
                                       published_root=args.published_source,
                                       published_archive=args.published_archive)
    except (OSError, ValueError, SyntaxError) as exc:
        parser.exit(1, f"AncestralPaths source rejected: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
