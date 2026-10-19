"""Tiny standalone CPU-only child for version 2 supervisor lifecycle checks."""
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sleep-seconds',type=float,required=True)
    args = parser.parse_args()
    assert 0. <= args.sleep_seconds <= 10.
    print('Synthetic child ready; standard library only.',flush=True)
    time.sleep(args.sleep_seconds)


if __name__ == '__main__':
    main()
