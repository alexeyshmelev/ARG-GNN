"""Launcher-compatible entry point for the synthetic admission script."""
from pathlib import Path
import runpy


def main():
    script = Path(__file__).resolve().parents[2]/'scripts/check_development_cuda_admission.py'
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
