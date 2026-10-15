#!/usr/bin/env python3
"""Validate only source/config and tiny deterministic metadata examples."""
import argparse
import importlib
import io
import json
import sys
import unittest
from pathlib import Path
from panel_adapter import HERE,Panel,make_roster,sha256,verify_sources,write_json

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipt")
    args=parser.parse_args()
    source_files=sorted(HERE.glob("*.py"))
    for path in source_files:
        compile(path.read_text(),str(path),"exec")
    module=importlib.import_module("test_metadata_adapters")
    suite=unittest.defaultTestLoader.loadTestsFromModule(module)
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    receipt={"schema":"prospective1611_native_metadata_validation_v1",
             "passed":result.wasSuccessful(),"tests_run":result.testsRun,
             "errors":len(result.errors),"failures":len(result.failures),
             "source_syntax_compiles":len(source_files),
             "frozen_author_source_pins_verified":verify_sources(),
             "source_file_sha256":{p.name:sha256(p) for p in source_files},
             "full_synthetic_observed_metadata_summary":Panel(make_roster(),require_bound=False).summary(),
             "scope":"Source/config hashes, syntax and tiny deterministic mapping/feature metadata only. No native extraction/fit/predict command, numerical library, real observed/truth payload, checkpoint, guard, job, install, GPU/process probe or Git operation.",
             "native_numeric_paths_executed":False,"paper_reproduction":False,
             "test_log":stream.getvalue()}
    if args.receipt:
        write_json(args.receipt,receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!="test_log"},indent=2,sort_keys=True))
    if not result.wasSuccessful():
        print(stream.getvalue())
        raise SystemExit(1)

if __name__=="__main__":
    main()
