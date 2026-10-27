"""Strict originalCPU1/128 anchor for a separately reviewed CPU8 successor."""
RUNNER_REF=dict(path='research/efficiency_20261027/root/compare_selected_full.py',
    sha256='caaaa39c3388f3f4198b9ea1815712e313645c123e6f00e8586bb6ae4e509b0f')


def need(ok,message):
    if not ok:raise ValueError(message)


def validate_full_qualification(full,selected):
    need(full.get('schema')=='TreeLoom_selected_large_VJP_compare_v1'
         and full.get('status')=='completed' and full.get('FP32') is True,
         'Completed selected-full FP32 comparison against original production required')
    refs=full.get('source_refs')
    need(type(refs) is list and refs and RUNNER_REF in refs,
         'Exact reviewed runner source establishes original compare_state anchor and strict tolerances')
    threads=full.get('CPU_threads_by_kind',{})
    need(threads.get('original')==1 and threads.get('explicit_cpu_vjp')==8,
         'OriginalCPU1 and selectedCPU8 comparison required')
    partition=full.get('execution_partitions',{})
    need(partition.get('original_holds_per_block')==128
         and partition.get('selected_holds_per_block')==selected['holds_per_block']
         and partition.get('haplotypes_per_gru')==64 and partition.get('history_window')==1024,
         'Exact original128 and selected block with unchanged decoder/history partitions required')
    records=full.get('records')
    need(type(records) is list and records,'Nonempty completed full comparison records required')
    first=records[0]
    need(first.get('kind')=='original' and first.get('phase')=='full_first_call'
         and first.get('repeat')==0 and first.get('order_position')==0
         and first.get('CPU_threads')==1 and first.get('holds_per_block')==128,
         'First actual full record must establish originalCPU1/128 as the comparison anchor')
    for kind,CPU,block in (('original',1,128),('explicit_cpu_vjp',8,selected['holds_per_block'])):
        rows=[row for row in records if row.get('kind')==kind]
        need(rows,'Both original and selected full-parent rows are required')
        for row in rows:
            compare=row.get('comparison',{})
            categories=compare.get('categories',{})
            need(row.get('CPU_threads')==CPU and row.get('holds_per_block')==block
                 and compare.get('passed') is True and compare.get('argmax_disagreements')==0
                 and all(categories.get(name,{}).get('passed') is True for name in
                         ('logits','loss','gradients','parameters','moments'))
                 and categories['gradients'].get('tensors')==49
                 and categories['parameters'].get('tensors')==49
                 and categories['moments'].get('tensors')==147,
                 'Every required full row must pass exact-thread/partition/all49-state comparison')
    return True
