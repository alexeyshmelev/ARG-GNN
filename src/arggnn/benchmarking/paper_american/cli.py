"""Create reviewable task files or run an already admitted native task."""
import argparse
import json
from pathlib import Path
from . import protocol as p

def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--stage", choices=["mlm", "contrastive", "predict"], required=True)
    plan.add_argument("--preparation", required=True)
    plan.add_argument("--output-directory", required=True)
    plan.add_argument("--task-file", required=True)
    plan.add_argument("--variant", choices=p.VARIANTS, default="paper_aligned")
    plan.add_argument("--checkpoint-completion")
    plan.add_argument("--workers", type=int, default=8)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "plan":
        spec = p.build_spec(stage=args.stage, preparation_ref=p.ref(args.preparation),
                            output_directory=args.output_directory, variant=args.variant,
                            checkpoint_completion_ref=p.ref(args.checkpoint_completion) if args.checkpoint_completion else None,
                            num_workers=args.workers)
        print(json.dumps(p.save(spec, args.task_file)))
    else:
        from .smoke import run_smoke
        print(json.dumps(p.save(run_smoke(), args.output)))

if __name__ == "__main__":
    main()
