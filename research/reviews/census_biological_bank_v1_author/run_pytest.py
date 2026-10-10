"""Run only H021 handmade CPU tests and preserve attempt output/hash receipts."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, sys, time
root = Path(__file__).resolve().parents[3]
dest = Path(__file__).resolve().parent
assert Path.cwd() == root
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
attempt = int(sys.argv[1])
assert attempt > 0
output_path = dest / f"pytest_attempt_{attempt:02d}.txt"
receipt_path = output_path.with_suffix(".json")
assert not output_path.exists() and not receipt_path.exists()
paths = ["src/arggnn/census_biological_bank_v1.py", "tests/test_census_biological_bank_v1.py", "configs/census_biological_bank_v1.json"]
command = [str(root / ".runtime/env/bin/python"), "-m", "pytest", paths[1], "-q", "-p", "no:cacheprovider", "--basetemp=" + str(dest / f"pytest_tmp_{attempt:02d}")]
pre = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
start = time.monotonic()
result = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy())
duration = time.monotonic() - start
output = result.stdout + result.stderr
output_path.write_text(output)
receipt = {"attempt": attempt, "started_at_utc": started, "command": command, "cwd": str(root), "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}, "seconds": duration, "exit_code": result.returncode, "source_sha256": pre, "output_sha256": hashlib.sha256(output.encode()).hexdigest(), "test_scope": "Handmade CPU TS/unit forwards/backward and one synthetic optimizer step only; no actual arrays/native inference/model fitting/GPU jobs."}
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
print(output)
print(json.dumps(receipt, indent=2))
sys.exit(result.returncode)
