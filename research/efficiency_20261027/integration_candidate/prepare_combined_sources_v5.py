"""Prepare isolated combinedv5;only declared pre-forward qualified route selection."""
from pathlib import Path
HERE=Path(__file__).parent
NAMES=('configuration','envelope','qualification','training','entrypoint','restore_entrypoint','prepare_task',
       'test_contracts','test_scheduler_restore')
for name in NAMES:
    source=HERE/(name+'_combined_v4.py');target=HERE/(name+'_combined_v5.py')
    text=source.read_text().replace('combined_v4','combined_v5').replace('combined_fixture_v4','combined_fixture_v5')
    text=text.replace('COMBINED_CPU_v4','COMBINED_CPU_v5').replace('CPU_readonly_probe_v4','CPU_readonly_probe_v5')
    if name=='configuration':
        text=text.replace('One explicit combined candidate; no automatic scientific or memory fallback.',
            'Explicit pre-forward choice between two qualified CPU-context routes;no OOM retry.')
        text=text.replace('architecture.exact_upward_residency.large_resident_executor_v1:build_executor',
            'architecture.qualified_residency_router.router_v1:build_executor')
        text=text.replace('executor_factory=FACTORY, boundary_fusion=True, upward_direct_residency=True,',
            "executor_factory=FACTORY, boundary_fusion='resident_route_only', upward_direct_residency='capacity_selected_before_forward',\n    resident_soft_bytes=74*GIB, capacity_selection='actual_prepared_nodes_and_live_capacity_before_forward',\n    qualified_fallback_factory='research.efficiency_20261027.execution.large_block_executors:build_executor',\n    fallback_context_residency='CPU', no_mid_update_fallback=True,")
        text=text.replace('CPU-context resident adaptive2 configuration required','CPU-context qualified capacity-route adaptive2 configuration required')
    if name=='envelope':
        text=text.replace('cold_compact_pool=True, default_fallback_allowed=False)',
            'cold_compact_pool=True, default_fallback_allowed=False,\n        declared_pre_forward_qualified_route_selection=True, mid_update_fallback_allowed=False)')
    if name=='training':
        text=text.replace('from .runtime_combined_v1 import','from .runtime_combined_v2 import')
        text=text.replace('explicit_VJP_CPU8_512_boundary_resident_adaptive2_768GiB_v1',
            'explicit_VJP_CPU8_512_qualified_capacity_route_adaptive2_768GiB_v2')
        text=text.replace('executor_factory=selected["executor_factory"],boundary_fusion=True,\n                upward_direct_residency=True,resident_scratch_bytes=selected["resident_scratch_bytes"],',
            'executor_factory=selected["executor_factory"],boundary_fusion=selected["boundary_fusion"],\n                upward_direct_residency=selected["upward_direct_residency"],resident_scratch_bytes=selected["resident_scratch_bytes"],\n                resident_soft_bytes=selected["resident_soft_bytes"],capacity_selection=selected["capacity_selection"],\n                qualified_fallback_factory=selected["qualified_fallback_factory"],no_mid_update_fallback=True,')
    if name=='test_contracts':
        text=text.replace('from . import step_combined_v1 as step','from . import step_combined_v2 as step')
        text=text.replace('one_resident_factory_includes_boundary_fusion=True,',
            'explicit_pre_forward_router_factory_bound=True,actual_route_receipt_per_complete_parent=True,')
    with target.open('x') as stream:stream.write(text)
text=(HERE/'step_combined_v1.py').read_text()
text=text.replace('from research.efficiency_20261027.architecture.exact_upward_residency.large_resident_executor_v1 import build_executor',
    'from research.efficiency_20261027.architecture.qualified_residency_router.router_v1 import build_executor')
text=text.replace('from .configuration_combined_v1 import FACTORY','from .configuration_combined_v5 import FACTORY,EXPECTED')
text=text.replace('executor = build_executor(model,bundle,metadata,',
    'executor = build_executor(model,bundle,metadata,\n                resident_soft_bytes=EXPECTED["resident_soft_bytes"],resident_scratch_bytes=EXPECTED["resident_scratch_bytes"],')
text=text.replace('executor_kind=executor_kind,executor_factory=FACTORY,decoder=decoder_receipt,',
    'executor_kind=executor_kind,executor_factory=FACTORY,decoder=decoder_receipt,\n            capacity_route_receipt=dict(executor.capacity_route_receipt),')
with (HERE/'step_combined_v2.py').open('x') as stream:stream.write(text)
text=(HERE/'runtime_combined_v1.py').read_text().replace('from .step_combined_v1 import','from .step_combined_v2 import')
with (HERE/'runtime_combined_v2.py').open('x') as stream:stream.write(text)
print('Staged combinedv5 modules;no launch and no sealedv4 changes.')
