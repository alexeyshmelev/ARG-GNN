"""Separate native author stage entry point, requested process title and GPU scope."""
from pathlib import Path
import hashlib
import importlib.util
import sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
MODULE=Path(__file__).resolve().parent/'native_inference_v1.py'
if __name__=='__main__':
    spec=importlib.util.spec_from_file_location('_ARGMix_native_inference_stage',MODULE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.launch_native_stage(sys.argv[1])
