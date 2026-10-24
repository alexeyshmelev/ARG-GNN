"""CPU qualification entry point for the four-worker operational runtime."""
import argparse
from .smoke import run

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["smoke"])
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    result=run(args.output)
    print(result["status"],result["workers"],result["parameters"])

if __name__=="__main__":
    main()
