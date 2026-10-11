import json, sys, traceback
from pathlib import Path
from unittest.mock import patch
from arggnn import biological_bank_cache_v1 as c
from arggnn import biological_candidate as f
from arggnn import biological_mechanism_controls_v1 as m
OUT=Path("research/reviews/biological_bank_cache_v1_independent")
class EmptyCache:
    def __init__(self):self.closed=False
    def close(self):self.closed=True
    def report(self):return {"closed":self.closed,"CPU_resident_bytes":0,"GPU_LRU_resident_bytes":0}
config=json.loads(Path("configs/biological_bank_cache_v1.json").read_text())
original_argv_object=sys.argv
runs=[]
for failure in (False,True):
    output=OUT/"handmade"/("mock_original_runner_error" if failure else "mock_original_runner_success")
    output.mkdir()
    refs={
       "original_config":{"path":"research/reviews/biological_bank_cache_v1_independent/handmade/scientific_config.json","sha256":"handmade_bound_config_SHA"},
       "observed_manifest":{"path":"research/reviews/biological_bank_cache_v1_independent/handmade/observed_manifest.json","sha256":"handmade_bound_observed_SHA"},
       "prepared_manifest":{"path":"research/reviews/biological_bank_cache_v1_independent/handmade/prepared_manifest.json","sha256":"handmade_bound_prepared_SHA"}}
    admission={"refs":refs,"seed":101,"device":"cpu","output_directory":str(output)}
    argv=["--train","--config",refs["original_config"]["path"],"--expected-config-sha256",refs["original_config"]["sha256"],
       "--expected-base-config-sha256",m.BASE_CONFIG_SHA256,"--observed-manifest",refs["observed_manifest"]["path"],
       "--expected-observed-manifest-sha256",refs["observed_manifest"]["sha256"],"--prepared-manifest",refs["prepared_manifest"]["path"],
       "--expected-prepared-manifest-sha256",refs["prepared_manifest"]["sha256"],"--seed","101","--device","cpu","--output",str(output),
       "--storage-snapshot","handmade_snapshot_path","--expected-storage-snapshot-sha256","handmade_snapshot_SHA"]
    cache=EmptyCache()
    def spy():
        assert sys.argv[1:]==argv
        assert sys.argv[0]==str(f.ROOT/"scripts/train_biological_mechanism_controls_v1.py")
        assert f.load_bank is cache and m.load_bank is cache
        if failure:raise RuntimeError("deliberate handmade runner spy failure")
    try:
        with patch.object(c,"admit_cache_v1",return_value=(cache,admission,config)),patch.object(m,"main",spy):
            c.run_original_mechanisms_with_cache_v1({"path":"handmade_mock_only","sha256":"mock"},argv)
    except RuntimeError as error:
        assert failure and "deliberate" in str(error)
    assert sys.argv is original_argv_object and cache.closed
    assert f.load_bank is c.BASE_LOAD_BANK and m.load_bank is c.BASE_LOAD_BANK
    report=json.loads((output/"bank_cache_execution_v1.json").read_text())
    assert report["original_science_argv"]==argv and report["closed"]
    runs.append({"failure_in_spy":failure,"original_argv_forwarded_verbatim":True,"aliases_and_sysargv_restored":True,"report_written_after_cleanup":True})
result={"schema":"biological_bank_cache_v1_independent_cli_test","status":"pass","runs":runs,"admission_mocked":True,"original_runner_mocked":True,"actual_profile_fit_or_scientific_arrays":False}
(OUT/"cli_test_results.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps(result))

