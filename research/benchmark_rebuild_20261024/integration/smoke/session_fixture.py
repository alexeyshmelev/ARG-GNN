"""Synthetic process ownership fixture. No benchmark or GPU access."""
import argparse
import subprocess
import sys
import time
parser=argparse.ArgumentParser()
parser.add_argument('mode',choices=['parent','child'])
args=parser.parse_args()
if args.mode=='parent':
    child=subprocess.Popen([sys.executable,'-B',__file__,'child'])
    print(child.pid,flush=True)
else:
    time.sleep(45)
