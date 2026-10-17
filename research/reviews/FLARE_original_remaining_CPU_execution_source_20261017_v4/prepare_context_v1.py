"""Future ROOT metadata-only context preparation, using unchanged original predicates."""
import argparse,json,sys
from pathlib import Path
from . import root_driver_v1 as d
def original():
 d.activate();m=d.load(d.bindings()['original_prepare_context_ref'],'_FLARE_phase_original_context');m.c=d.c;return m
def bind_selected_context_v1(record):return original().bind_selected_context_v1(record)
def remaining_cell_evidence_v1(record):return original().remaining_cell_evidence_v1(record)
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bind-ref');a=p.parse_args(argv)
 if not a.bind_ref:print(json.dumps(dict(status='SOURCE_ONLY_INACTIVE',writes=False,admission=False)));return 0
 d.need(Path.cwd()==d.ROOT and sys.dont_write_bytecode and sys.flags.no_user_site,'Exact ROOT -B -s interpreter')
 print(json.dumps(bind_selected_context_v1(json.loads(a.bind_ref)),sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
