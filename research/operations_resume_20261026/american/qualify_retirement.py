"""Exercise retirement deletion selection on tiny isolated synthetic files only."""
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from arggnn.benchmarking.paper_american import protocol as p
from research.operations_resume_20261026.american import argformer_retirement as m
HERE=ROOT/'research/operations_resume_20261026/american'


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only synthetic test required')
    with tempfile.TemporaryDirectory(prefix='retirement_fixture_',dir=HERE) as temporary:
        fixture=Path(temporary)
        campaign=fixture/'experiments/results/paper_american_20261024'
        base=campaign/m.RESUMED_PARENT
        with patch.object(p,'ROOT',fixture),patch.object(m,'CAMPAIGN',campaign),patch.object(m,'PRESERVED_CHECKPOINT_BYTES',3):
            def blob(relative,data=b'metadata'):
                destination=fixture/relative
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(data)
                return p.ref(destination)
            def metadata(relative,value):
                destination=fixture/relative
                destination.parent.mkdir(parents=True,exist_ok=True)
                return p.save(value,destination)
            def rel(path):return str(path.relative_to(fixture))
            old=base/'mlm/native/checkpoints/ep2-ba119500-rank0.pt'
            preserved=blob(rel(old),b'old')
            old_config=base/'mlm/native/resolved_config.yaml'
            blob(rel(old_config),b'untouched old config')
            new=base/'mlm/native'/m.RESUME_LEAF/'checkpoints/final.pt'
            new_ref=blob(rel(new),b'new')
            contrastive=base/'contrastive/native/checkpoints/final.pt'
            contrastive_ref=blob(rel(contrastive),b'new contrastive')
            evidence=metadata('evidence.json',dict(schema='american_argformer_full_state_resume_20261026',checkpoint_ref=preserved))
            unary=metadata('unary.json',dict(collapsed_num_nodes=4))
            observed=metadata('observed.json',dict(unary_preprocessing_ref=unary))
            prep=dict(observation_ref=observed,vocab_ref=blob('vocab.bin'),focals_ref=blob('focals.bin'))
            prep_ref=metadata('preparation.json',prep)
            records={}
            for stage in ('mlm','contrastive','predict'):
                value={}
                if stage in ('mlm','contrastive'):
                    value['checkpoint_ref']=new_ref if stage=='mlm' else contrastive_ref
                    value['resolved_config_ref']=blob(rel(m.producer_output(base,stage)/'resolved_config.yaml'))
                else:
                    value['prediction_ref']=blob(rel(base/'predict/native/predictions.bin'))
                    value['reference_pca_ref']=blob(rel(base/'predict/native/pca.bin'))
                spec=metadata(rel(m.producer_output(base,stage)/'executed_spec.json'),dict(output_directory=rel(m.producer_output(base,stage))))
                value['self_spec_ref']=spec
                records[stage]=dict(value=value,reference=metadata(rel(base/stage/'native/completed.json'),value),
                                    executed_spec_ref=spec,wait_ref=metadata(rel(base/stage/'actual_wait.json'),dict(status='succeeded')))
            maximum=4*(100000000+768*8)+32*2**20
            request=dict(parent_id=m.RESUMED_PARENT,resume_evidence_ref=evidence,
                         output_directory=rel(base/'retire/native'),max_retained_model_bytes=maximum)
            def compact(torch,checkpoint_path,destination,maximum_bytes,check):
                destination.write_bytes(b'compact')
                return dict(model_ref=p.ref(destination),unique_tensor_storage_bytes=7,
                            all_model_tensors_bitwise_preserved=True,tensor_count=1)
            fixture_source=fixture/'retirement_source.py'
            fixture_source.write_text(Path(m.__file__).read_text())
            with patch.object(m,'_chain',lambda request:(base,prep_ref,prep,records)),patch.object(m,'_compact',compact),patch.object(m,'__file__',str(fixture_source)):
                result=m.run_task(request,resource_check=lambda stage:None)
            assert old.read_bytes()==b'old'
            assert old_config.read_bytes()==b'untouched old config'
            assert not new.exists() and not contrastive.exists()
            assert result['retired_file_count']==2 and result['preserved_interrupted_checkpoint_bytes']==3
            assert result['retained_nonmodel_cap_bytes']==128*2**20+3
            inventory=p.read(result['inventory_ref'])
            assert preserved in inventory['retained_refs']
            assert all(item['path']!=preserved['path'] for item in inventory['files'])
            assert m.producer_output(base,'mlm')==base/'mlm/native'/m.RESUME_LEAF
            assert m.producer_output(base,'contrastive')==base/'contrastive/native'
            try:m.preserved_checkpoint(dict(request,parent_id='argformer_american_validation_002'))
            except ValueError:pass
            else:raise AssertionError('Wrong parent accepted')
    report=dict(status='passed',synthetic_files_only=True,production_files_deleted=0,
                old_checkpoint_and_config_preserved=True,new_mlm_and_contrastive_checkpoint_cleanup=True,
                retained_storage_includes_old_checkpoint=True,wrong_parent_rejected=True,
                original_scientific_compaction_and_lineage_mocked=True)
    p.save(report,HERE/'argformer_retirement.qualification.json')
    print(json.dumps(report))


if __name__=='__main__':run()
