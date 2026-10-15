#!/usr/bin/env python3
"""Only source/config hashes, syntax and deterministic injected metadata cases."""
import argparse
import importlib
import io
import json
import unittest
import generation as g

def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt');args=p.parse_args()
    files=sorted(g.HERE.glob('*.py'))
    for f in files:compile(f.read_text(),str(f),'exec')
    suite=unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module('test_metadata_generation'))
    log=io.StringIO();r=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    receipt={'schema':'prospective1611_generation_source_validation_v1','passed':r.wasSuccessful(),'tests_run':r.testsRun,
             'errors':len(r.errors),'failures':len(r.failures),'source_syntax_compiles':len(files),'source_pins_verified':g.verify_sources(),
             'source_file_sha256':{f.name:g.sha(f) for f in files},'native_numeric_paths_executed':False,
             'actual_simulations_payloads_models_guards_jobs_Git_installs_downloads_or_probes':False,'test_log':log.getvalue()}
    if args.receipt:g.write_json(args.receipt,receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='test_log'},indent=2,sort_keys=True))
    if not r.wasSuccessful():print(log.getvalue());raise SystemExit(1)
if __name__=='__main__':main()
