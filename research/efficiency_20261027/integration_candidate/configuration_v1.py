"""Explicit execution configuration; qualification is checked by the release gate."""

def configuration(value):
    required={'executor_kind','CPU_threads','holds_per_block','haplotypes_per_gru',
        'history_holds_per_window','context_residency','layout_cache_max_bytes','layout_cache_policy'}
    if not (type(value) is dict and set(value)==required
        and value['executor_kind']=='explicit_vjp' and value['CPU_threads']==8
        and type(value['holds_per_block']) is int and value['holds_per_block'] in (128,512,1024)
        and value['haplotypes_per_gru']==64 and value['history_holds_per_window']==1024
        and value['context_residency']=='CPU' and type(value['layout_cache_max_bytes']) is int
        and 8*2**30<=value['layout_cache_max_bytes']<=80*2**30
        and value['layout_cache_policy']=='actual_RSS_evict_unleased'):
        raise ValueError('Exact explicitly qualified CPU8/128,512,1024/64/1024 execution configuration required')
    return dict(value)
