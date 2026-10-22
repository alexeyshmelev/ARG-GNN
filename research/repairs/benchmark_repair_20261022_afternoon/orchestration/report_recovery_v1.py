"""Record the operational retry and refresh concise human benchmark status."""
from collections import Counter
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import subprocess
from research.repairs.benchmark_repair_20261022_afternoon.runtime import readable_workflow_v1 as queue

c=queue.c
HERE=Path(__file__).parent
ACTUAL=HERE/'actual_v1'


def report():
    reference=c.file_ref('configs/benchmarks/executable_workflow_20261022_afternoon_v1.json')
    plan=queue.load_plan(reference,approved=True)
    state=c.read_json(c.contained_path(plan['records_directory'])/'state.json')
    now=datetime.now(timezone.utc).isoformat()
    projection=queue.storage_projection(plan,state)
    active=[]
    for binding in plan['workflow_external_predecessors']:
        _,entry=queue.support.predecessor._source(binding)
        if queue.support.predecessor._drained(binding,entry) is None:
            active.append(dict(task_id=binding['source_task_id'],supervisor_pid=binding['original_supervisor_pid'],
                source_status=entry['status'],resource=binding['resource'],max_rss_bytes=binding['max_rss_bytes']))
    receipt=dict(updated_utc=now,plan_ref=reference,controller_identity=queue.shared_gpu.identity(2726331),
        counts=dict(Counter(entry['status'] for entry in state['tasks'].values())),
        healthy_preserved_jobs=active,American_retry=state['tasks']['AM_treeloom_fit'],
        storage_projection_bytes=projection,operational_ceiling_bytes=980000000000,
        original_American_updates=600,original_American_epochs=30,new_matched_ancestry_scores=None,
        prior_failure=dict(completed_optimizer_updates=0,CUDA_allocator_cap_GiB=32,
            files_preserved=True,reason='CUDA allocator ceiling reached during backward despite physical free memory'),
        repair=dict(CUDA_allocator_cap_GiB=64,minimum_free_GPU_GiB=72,max_host_RSS_GiB=512,
            max_parallel_RSS_GiB=1792,model_or_data_changed=False),signals_sent=0)
    queue.save(ACTUAL/'status.json',receipt)
    lock_path=c.ROOT/'.runtime/benchmarking/workflow_reporting/reporting.lock'
    with lock_path.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        decisions=c.read_json(c.ROOT/'research/experiment_decisions.json')
        identifier='benchmark_American_GPU_memory_retry_20261022'
        if not any(row['id']==identifier for row in decisions['decisions']):
            decisions['decisions'].append(dict(id=identifier,hypothesis='Complete the unfinished frozen American TreeLoom benchmark fit under an adequate operational GPU memory cap',
                status='released_waiting_for_full_RAM_reservation',prior_hypothesis_ids=['H050','H062'],architecture_search=False,
                method='Original 20 TRAIN parents, 30 epochs, 600 full-parent updates. Increase only GPU 2 allocator cap from 32 to 64 GiB and admission floor from 40 to 72 GiB. Preserve 512 GiB RSS. Same data, model, RNG and checkpoints. Fresh output, no resume checkpoint exists.',
                conclusion='The 32 GiB attempt failed during backward with zero complete updates and about 46.77 GiB physical memory still free. New full 349-stage recovery queue adopted 236 genuine successes and preserved four healthy jobs. Cap sufficiency is unproven until complete backward succeeds. No new ancestry score.',
                repeat_policy='Operational retry of unfinished cell only. Do not change science, reopen answered pilots or compare training loss as accuracy. Preserve all failed artifacts and genuine waits.',
                registered_utc=now,evidence=[str((ACTUAL/'launch_receipt.json').relative_to(c.ROOT)),str((ACTUAL/'preparation_receipt.json').relative_to(c.ROOT)),str((ACTUAL/'status.json').relative_to(c.ROOT)),
                    'research/repairs/benchmark_repair_20261022_afternoon/training/failed_attempt_v3.receipt.json',
                    'research/repairs/benchmark_repair_20261022_afternoon/runtime/validation_v1.json']))
        decisions['last_root_update_utc']=now
        queue.save(c.ROOT/'research/experiment_decisions.json',decisions)
        for script in ['scripts/update_research_ledger.py','scripts/update_current_scoreboard.py']:
            subprocess.run([str(c.ROOT/'.runtime/env/bin/python'),'-B','-s',script],cwd=c.ROOT,check=True)
        readme=c.ROOT/'README.md'
        text=readme.read_text()
        begin=text.index('<!-- BEGIN CURRENT BENCHMARK DIAGNOSIS -->')
        end=text.index('<!-- END CURRENT BENCHMARK DIAGNOSIS -->',begin)+len('<!-- END CURRENT BENCHMARK DIAGNOSIS -->')
        block='''<!-- BEGIN CURRENT BENCHMARK DIAGNOSIS -->
**22 October: the repaired benchmark queue is active, with 236 completed stages preserved.** American TreeLoom's 32 GiB attempt ran out of its CUDA allowance during backward, with physical GPU memory still available. It completed 0/600 updates. The retry uses a 64 GiB cap, 72 GiB free-memory admission and the original 512 GiB host reservation. Its model, data and 600-update/30-epoch training budget are unchanged. It is queued for the next available full RAM reservation. **Fresh matched ancestry accuracy and F1 remain pending.**

| Workstream | Device assignment | Current execution |
| --- | --- | --- |
| American TreeLoom | GPU 2 | Repaired full fit queued, waiting for 512 GiB host reservation. A complete backward is still required to establish memory sufficiency |
| Native ARGformer, validation 003 | GPU 3 | Pretraining advancing. Last verified 47,759/226,425 updates at 11:22 UTC |
| Native ARGformer, validation 002 | GPU 5 | All 223,655 pretraining updates completed. Contrastive fit advancing, 360/2,094 updates at 11:22 UTC |
| European TreeLoom, native AP endpoint | GPU 3 after CPU preparation | Current implementation qualification passed. Full CPU preparation is advancing |
| European TreeLoom, four ancestry classes | GPU 5 after CPU preparation | Current implementation qualification passed. Full CPU preparation is advancing |
| Released ARGMix | GPU 2 for fitting | First repaired Rust archive completed: 65,597,378 rows and 95,623 intervals. Three further archives await RAM |
| FLARE | CPU | All 15 American prediction outputs retained for paired scoring |

The fresh recovery controller preserves the four healthy jobs as live predecessors. Its local counters can show zero running workers while those preserved jobs continue under their original supervisors. The older queue blocks describe those supervisors and historical failures. Use the afternoon recovery queue for future dispatch. No failure was relabelled as a success, and no unrelated process was changed.

All individual RAM limits remain intact. The aggregate reservation ceiling is 1,792 GiB on a host with about 2,015 GiB, with the original 64 GiB host margin. The integrated storage projection is 935.54 GB, below the 980 GB operational ceiling and 1 TB project limit. GPU 2's retry must wait until at least one 256 GiB native reservation drains. GitHub SSH connections have been timing out, so commits are retained on anogena pending network recovery.

[Repair and progress](research/repairs/benchmark_repair_20261022_afternoon/orchestration/actual_v1/status.json) · [Current queue](research/benchmark_recovery_afternoon_current.json) · [Matched results](research/resource_matched_benchmark_results.json)
<!-- END CURRENT BENCHMARK DIAGNOSIS -->'''
        text=text[:begin]+block+text[end:]
        old='**The fresh matched comparison still has no ancestry scores.**'
        a=text.index(old);b=text.index('\n\n',a)
        text=text[:a]+'**The fresh matched comparison still has no ancestry scores.** FLARE predictions are complete for 15/15 American simulations and native ARGformer predictions for 2/15. American TreeLoom has no completed fit in this fresh campaign. Its failed 32 GiB attempt is preserved, and the repaired 64 GiB retry is queued. See the current diagnosis above for active training and memory admission.'+text[b:]
        queue.save(c.ROOT/'research/benchmark_repair_status_20261022.json',receipt)
        temporary=readme.with_name('README.md.root_recovery')
        temporary.write_text(text)
        temporary.replace(readme)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':report()
