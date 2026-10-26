"""Small structural tests, no model, process, queue or archive operations."""
from pathlib import Path
import copy
import json
from . import migrate_resource592 as candidate
from arggnn.benchmarking.paper_campaign import runtime as r


def main():
    source={'schema':'paper_campaign_repair_request_v1','request_id':'original',
        'reason':'Preserve scientific setup.', 'reset_task_ids':['dependent'],
        'replacement_tasks':[dict(id=candidate.original.TASK,max_rss_bytes=768*2**30,
            cpu_threads=1,source_refs=[{'unchanged':'pin'}],
            call=dict(module='research.preprocessing_optimization_20261026.integration.treeloom_resume',
                function='run_task',spec=dict(stage='fit',arguments=dict(
                    max_rss_bytes=768*2**30,observation_cache_bytes=352*2**30,
                    preparation_headroom_bytes=128*2**30,training_headroom_bytes=128*2**30,
                    max_truth_uncompressed_bytes=8*2**30,science={'unchanged':[1,2,3]}))))]}
    before=copy.deepcopy(source)
    amended=candidate.amend_request(source)
    assert source==before
    assert amended['replacement_tasks'][0]['max_rss_bytes']==592*2**30
    assert amended['replacement_tasks'][0]['call']['spec']['arguments']['max_rss_bytes']==592*2**30
    restored=copy.deepcopy(amended)
    restored['request_id']=source['request_id']
    restored['reason']=source['reason']
    restored['replacement_tasks'][0]['max_rss_bytes']=768*2**30
    restored['replacement_tasks'][0]['call']['spec']['arguments']['max_rss_bytes']=768*2**30
    assert restored==source
    rejected=0
    for key in ['observation_cache_bytes','preparation_headroom_bytes','training_headroom_bytes','max_truth_uncompressed_bytes','max_rss_bytes']:
        invalid=copy.deepcopy(source)
        invalid['replacement_tasks'][0]['call']['spec']['arguments'][key]-=1
        try:candidate.amend_request(invalid)
        except ValueError:rejected+=1
        else:raise AssertionError(key)
    for field,value in [('id','another_task'),('max_rss_bytes',767*2**30)]:
        invalid=copy.deepcopy(source)
        invalid['replacement_tasks'][0][field]=value
        try:candidate.amend_request(invalid)
        except ValueError:rejected+=1
        else:raise AssertionError(field)
    invalid=copy.deepcopy(source)
    invalid['replacement_tasks'].append(copy.deepcopy(invalid['replacement_tasks'][0]))
    try:candidate.amend_request(invalid)
    except ValueError:rejected+=1
    else:raise AssertionError('Multiple replacements')
    GiB=2**30
    exact=candidate.admission_arithmetic(1393*GiB,115*GiB,[596*GiB],[256*GiB])
    assert exact['margin_bytes']==0 and exact['admissible_projection']
    low=candidate.admission_arithmetic(1393*GiB-1,115*GiB,[596*GiB],[256*GiB])
    assert low['margin_bytes']==-1 and not low['admissible_projection']
    competing_gpu=candidate.admission_arithmetic(1393*GiB,115*GiB,[596*GiB],[256*GiB,128*GiB])
    assert competing_gpu['margin_bytes']==-128*GiB and not competing_gpu['admissible_projection']
    after_exit=candidate.admission_arithmetic(1508*GiB,0,[596*GiB],[256*GiB])
    assert after_exit['margin_bytes']==0 and after_exit['admissible_projection']
    for ref in candidate.PINS:
        assert r.ref(ref['path'])==ref
    result=dict(status='pass',source_ref=r.ref(candidate.__file__),test_source_ref=r.ref(__file__),
        pinned_inputs=candidate.PINS,exact_task_difference_fields=['max_rss_bytes','call.spec.arguments.max_rss_bytes'],
        rejected_mutations=rejected,arithmetic_boundaries=4,production_files_changed=False,
        process_operations=False,model_executed=False,original_source_preserved=True)
    output=r.ROOT/'research/preprocessing_optimization_20261026/review/resource592_structural_tests.json'
    with output.open('x') as handle:
        json.dump(result,handle,indent=2)
        handle.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
