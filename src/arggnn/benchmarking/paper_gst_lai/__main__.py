"""Describe or synthetically qualify GST-LAI-8; full stages require ROOT admission."""
import argparse,json,traceback
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["describe","smoke"])
    parser.add_argument("--output")
    args=parser.parse_args()
    if args.command == "describe":
        from .model import specification
        print(json.dumps(specification(),indent=2,sort_keys=True))
    else:
        from .smoke import run
        result=run(args.output)
        print(json.dumps(dict(status=result["status"],output=args.output)))
if __name__ == "__main__":
    try: main()
    except BaseException:
        print(traceback.format_exc(),flush=True)
        raise
