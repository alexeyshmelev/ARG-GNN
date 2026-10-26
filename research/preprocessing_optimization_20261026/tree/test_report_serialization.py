"""Qualification report serialization, atomic publication and strict failure tests."""
from pathlib import Path
import json
import tempfile
import numpy as np
from .full_process_parent import OUT,json_scalar,write_record_atomic


def main():
    event={'event':'parallel_preparation_chunk','first_hold':np.int64(0),
        'last_hold':np.int64(128),'charged_bytes':np.int64(64000000),
        'pending_charged_bytes':np.uint64(128000000),'torch_threads':1,
        'native_threads':[1,1],'cuda_initialized':np.bool_(False),
        'rss':np.float64(630000000.),'sample':np.float32(0.5)}
    result=json.loads(json.dumps(event,default=json_scalar,allow_nan=False))
    assert result=={'event':'parallel_preparation_chunk','first_hold':0,
        'last_hold':128,'charged_bytes':64000000,'pending_charged_bytes':128000000,
        'torch_threads':1,'native_threads':[1,1],'cuda_initialized':False,
        'rss':630000000.,'sample':0.5}
    with tempfile.TemporaryDirectory(dir=OUT,prefix='report_test_') as temp:
        target=Path(temp)/'report.json'
        write_record_atomic(target,{'events':[event]})
        assert json.loads(target.read_text())=={'events':[result]}
        assert not target.with_suffix('.json.tmp').exists()
        try:write_record_atomic(target,{'changed':True})
        except FileExistsError:pass
        else:raise AssertionError('Existing evidence was overwritten')
        assert json.loads(target.read_text())=={'events':[result]}
        for index,value in enumerate([object(),np.array([1]),np.float64(float('nan'))]):
            invalid=Path(temp)/f'invalid_{index}.json'
            try:write_record_atomic(invalid,{'invalid':value})
            except (TypeError,ValueError):pass
            else:raise AssertionError('Unsupported or nonfinite report accepted')
            assert not invalid.exists()
            assert not invalid.with_suffix('.json.tmp').exists()
    print(json.dumps({'passed':True,'numpy_scalar_events':True,'atomic_publication':True,
        'preserve_existing_evidence':True,'unsupported_and_nonfinite_rejection':True}))

if __name__=='__main__':main()
