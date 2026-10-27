"""Synthetic bounded-evidence/AST checks only; no observation or child execution."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from .preparation_evidence_v2 import bounded_signature, canonical_digest, validate_signature
from .compare_two_parent_preparation_v2 import BASE, write
from .launch_two_parent_comparison_v2 import argv_for


def rejects(call):
    try:
        call()
    except (ValueError, TypeError, KeyError):
        return
    raise AssertionError('Invalid evidence was accepted')


def function(source, name):
    node=next(n for n in ast.parse(source).body if getattr(n,'name',None)==name)
    return ast.dump(node)


def main():
    stages=[]
    signature=dict(arrays={'bank.example':'a'*64, 'outer.focal':'b'*64},
        structures={'source_fingerprints':['0123456789abcdef'*8]*70000},
        outer={'parent_id':'synthetic_only','count':7},metadata={'unicode':'µ','ordered':[1,2,3]})
    layout={'bank.example':dict(shape=[2,3],dtype='<i8',count=6),
            'outer.focal':dict(shape=[1],dtype='<i4',count=1)}
    raw=json.dumps(signature,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')
    assert len(raw)>8*2**20
    result=bounded_signature(signature,array_metadata=layout,resource_check=stages.append)
    assert result['complete_signature_digest']==dict(sha256=hashlib.sha256(raw).hexdigest(),canonical_bytes=len(raw))
    assert len(json.dumps(result).encode())<8192
    assert stages.count('canonical_signature_stream')>=16
    validate_signature(result)
    reordered={key:signature[key] for key in reversed(list(signature))}
    assert bounded_signature(reordered,array_metadata=layout,resource_check=stages.append)==result
    for section in ('arrays','structures','outer','metadata'):
        changed=copy.deepcopy(signature)
        if section=='arrays':changed[section]['bank.example']='c'*64
        elif section=='structures':changed[section]['source_fingerprints'][-1]='changed-last-fingerprint'
        elif section=='outer':changed[section]['count']=8
        else:changed[section]['ordered']=[3,2,1]
        other=bounded_signature(changed,array_metadata=layout,resource_check=lambda stage:None)
        assert other!=result and other['section_digests'][section]!=result['section_digests'][section]
    rejects(lambda:canonical_digest(float('nan'),resource_check=lambda stage:None))
    rejects(lambda:bounded_signature({},array_metadata=layout,resource_check=lambda stage:None))
    altered=copy.deepcopy(result);altered['array_metadata']['bank.example']['count']=7
    rejects(lambda:validate_signature(altered))
    altered=copy.deepcopy(result);altered['arrays']['bank.example']='c'*64
    rejects(lambda:validate_signature(altered))
    rejects(lambda:write(BASE/'never_written_oversized_signature_v2.json',signature))
    assert not (BASE/'never_written_oversized_signature_v2.json').exists()
    old=(BASE/'compare_two_parent_preparation_v1.py').read_text()
    new=(BASE/'compare_two_parent_preparation_v2.py').read_text()
    assert old.split('def run_arm')[1].split('# No signature pass')[0]==new.split('def run_arm')[1].split('# No signature pass')[0]
    for name in ('write','identity','restore_CPU_random_state','CPU_random_signature','FamilyMonitor'):
        assert function(old,name)==function(new,name),name
    argv=argv_for(SimpleNamespace(directory='research/efficiency_20261027/root/runs/never_launched_comparison_v2',
        gpu=5,peer_directory=['research/efficiency_20261027/root/runs/european_cohort8_full_gpu5_v1'],source_pins_sha256='synthetic'))
    assert argv[argv.index('--device-gib')+1]=='0' and argv[argv.index('--host-gib')+1]=='288'
    assert 'research.efficiency_20261027.execution.compare_two_parent_preparation_v2' in argv
    assert not Path('research/efficiency_20261027/root/runs/never_launched_comparison_v2').exists()
    receipt=dict(status='passed',synthetic_raw_canonical_bytes=len(raw),bounded_receipt_bytes=len(json.dumps(result).encode()),
        all_sections_full_SHA256_and_bytecounts=True,array_shape_dtype_count_and_original_hashes=True,
        mutation_at_end_of_large_structure_detected=True,key_order_canonical=True,nonfinite_rejected=True,
        old_8MiB_cap_unchanged_and_oversized_payload_rejected=True,timed_prepare_pack_source_identical=True,
        family_monitor_RNG_restore_and_writer_AST_identical=True,science_launches=0,child_launches=0,
        fixture_kind='synthetic_metadata_only',visible_launch_plan=argv)
    write(BASE/'TWO_PARENT_RUNNER_CPU_v2.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
