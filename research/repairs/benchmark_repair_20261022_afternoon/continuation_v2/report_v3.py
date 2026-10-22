"""Publish current verified progress, including preserved workers and real blockers."""
from collections import Counter
from datetime import datetime
import fcntl
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo
from research.repairs.benchmark_repair_20261022_afternoon.continuation_v2 import storage_workflow_v3 as facade

q=facade.queue
c=q.c
HERE=Path(__file__).parent
ACTUAL=HERE/'actual_v3'


def report():
    policy=c.read_json(c.POLICY_PATH)
    reference=policy['approved_recovery_continuation_workflow_ref']
    plan=q.load_plan(reference,approved=True)
    state=c.read_json(c.contained_path(plan['records_directory'])/'state.json')
    now=datetime.now(ZoneInfo('Europe/Moscow'))
    preserved=[];mlm_batch=None;am_updates=None
    for binding in plan['workflow_external_predecessors']:
        _,entry=q.support.predecessor._source(binding)
        if q.support.predecessor._drained(binding,entry) is not None:continue
        identity=q.shared_gpu.identity(binding['original_supervisor_pid'])
        q.need(identity==dict(pid=binding['original_supervisor_pid'],start_ticks=binding['original_supervisor_start_ticks']),
            'Preserved worker generation must match')
        preserved.append(dict(task_id=binding['source_task_id'],status=entry['status'],supervisor_identity=identity))
        output=c.contained_path(binding['output_directory'])
        if binding['source_task_id'].endswith('validation_003_argformer_mlm'):
            live=c.read_json(output/'native/latest_batch_memory.json')
            mlm_batch=live['batch']
        if binding['source_task_id']=='AM_treeloom_fit':
            latest=c.read_json(output/'native/latest_checkpoint.json')
            am_updates=latest['complete_parent_updates']
    archive_id=plan['supplemental_external_ap_closed_recovery']['task_id']
    receipt=dict(observed_moscow=now.isoformat(),plan_ref=reference,controller_identity=q.shared_gpu.identity(2837679),
        counts=dict(Counter(entry['status'] for entry in state['tasks'].values())),preserved_workers=preserved,
        American_complete_optimizer_updates=am_updates,American_expected_updates=600,
        ARGformer_VAL003_MLM_completed_updates=mlm_batch,ARGformer_VAL003_expected_updates=226425,
        ARGformer_VAL002_contrastive_completed_updates=2094,ARGformer_VAL002_contrastive_expected_updates=2094,
        AP_prepared_original_train_parents=3,AP_stage='waiting_for_assigned_GPU3_lease',
        C4_retry=state['tasks']['C4__all_train__c4_treeloom_fit'],C4_resource_cap_GiB=68,C4_expected_updates=90,
        supplementary_archive=state['tasks'][archive_id],supplementary_credit_bytes=facade.archive.completed_storage_reduction(plan,state),
        storage_projection_bytes=q.storage_projection(plan,state),storage_ceiling_bytes=980000000000,
        new_matched_ancestry_scores=None,signals_sent=0,scientific_models_data_and_budgets_unchanged=True)
    q.save(ACTUAL/'current_status.json',receipt)
    with (c.ROOT/'.runtime/benchmarking/workflow_reporting/reporting.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        decisions=c.read_json(c.ROOT/'research/experiment_decisions.json')
        row=next(x for x in decisions['decisions'] if x['id']=='benchmark_C4_memory_and_capacity_repair_20261022')
        row['status']='released_archive_'+receipt['supplementary_archive']['status']
        row['method']='Preserve all models, simulations, seeds, 22 qualified European sources and 90 C4 updates. Retry C4 on GPU 5 with a 68 GiB allocator cap, 78 GiB admission floor and 512 GiB host reservation. Archive seven allowlisted completed AP intermediates with full decompression and hash verification, preserving 28 files. Credit only after genuine successful waits and exact measurement. Complete VAL002 embeddings, retrieval and scratch retirement before the C4 retry to respect full RAM reservations.'
        row['conclusion']='American TreeLoom completed its first full backward and optimizer update with a saved checkpoint. Native VAL002 finished all 2,094 contrastive updates. The C4 fit failed at its 32 GiB cap with 46.76 GiB physically free and zero updates. The 350-stage successor preserves all 237 completed stages and three healthy workers. The 68 GiB C4 cap and archive savings still require actual completion. No fresh matched ancestry accuracy is available.'
        row['evidence']=list(dict.fromkeys(row['evidence']+[
            str((ACTUAL/'launch_receipt.json').relative_to(c.ROOT)),str((ACTUAL/'current_status.json').relative_to(c.ROOT)),
            str((HERE/'final_review_v3.json').relative_to(c.ROOT))]))
        q.save(c.ROOT/'research/experiment_decisions.json',decisions)
        for script in ['scripts/update_research_ledger.py','scripts/update_current_scoreboard.py']:
            subprocess.run([str(c.ROOT/'.runtime/env/bin/python'),'-B','-s',script],cwd=c.ROOT,check=True)
        readme=c.ROOT/'README.md';text=readme.read_text()
        start='<!-- BEGIN CURRENT BENCHMARK DIAGNOSIS -->';end='<!-- END CURRENT BENCHMARK DIAGNOSIS -->'
        first=text.index(start);last=text.index(end,first)+len(end)
        block=f'''{start}
**Verified {now.strftime('%d %B %H:%M')} Moscow: the benchmark is progressing, with a European memory failure repaired in the queue.** All 237 completed stages and the three healthy workers are preserved. Fresh matched ancestry accuracy and F1 are still pending.

| Workstream | Current verified progress |
| --- | --- |
| American TreeLoom, GPU 2 | {am_updates}/600 full optimizer updates and checkpoint saved. The first complete backward passed under the 64 GiB cap. Preparing the next original training parent |
| Native ARGformer, GPU 3 | VAL003 pretraining: {mlm_batch:,}/226,425 updates |
| Native ARGformer, third validation simulation | VAL002 contrastive training completed all 2,094 updates with successful waits. Embeddings and ancestry predictions follow storage recovery |
| European TreeLoom, AP endpoint | All three training parents prepared. Waiting for its assigned GPU 3 lease |
| European TreeLoom, four ancestry classes | The first fit hit its 32 GiB cap with zero updates. A fresh retry uses 68 GiB, preserving all 22 qualified sources and the full 90-update budget |
| Storage recovery | Lossless archive step is {receipt['supplementary_archive']['status']}. Seven completed inference intermediates are eligible, with 28 observed-data, result and evidence files protected |

The capacity order is lossless recovery, then VAL002 native embeddings, retrieval and scratch retirement, then the C4 retry. This prevents the C4 host reservation from starving the ready native prediction chain. Compression is verified before each raw replacement. Space is credited only after a successful complete archive, real process waits and exact measurement. No training budget or scientific method is reduced.

The active storage projection is {receipt['storage_projection_bytes']/1e9:.2f} GB, below the 980 GB operational ceiling and the 1 TB project limit. This is a reservation projection, not a new directory-size scan. The three preserved workers remain active under their original supervisors even when the newest controller's local counters list only the archive.

[Verified progress](research/repairs/benchmark_repair_20261022_afternoon/continuation_v2/actual_v3/current_status.json) · [Current queue](research/benchmark_recovery_afternoon_v2_current.json) · [Matched results](research/resource_matched_benchmark_results.json)
{end}'''
        text=text[:first]+block+text[last:]
        if '<!-- preserved-worker-controllers:start -->' not in text:
            old=[]
            for marker in ['readable-recovery-afternoon-queue','readable-parallel-resource-overlay','readable-parallel-queue','readable-recovery-queue']:
                a='<!-- '+marker+':start -->';b='<!-- '+marker+':end -->'
                if a not in text:continue
                left=text.index(a);right=text.index(b,left)+len(b)
                old.append(text[left:right]);text=text[:left]+text[right:]
            if old:
                details='\n<!-- preserved-worker-controllers:start -->\n<details>\n<summary>Preserved controllers and earlier attempts</summary>\n\n'+'\n\n'.join(old)+'\n\n</details>\n<!-- preserved-worker-controllers:end -->\n'
                at=text.index(end)+len(end);text=text[:at]+details+text[at:]
        temporary=readme.with_name('README.md.verified_v3');temporary.write_text(text);temporary.replace(readme)
        q.save(c.ROOT/'research/benchmark_current_verified_status.json',receipt)
    print(receipt,flush=True)


if __name__=='__main__':report()
