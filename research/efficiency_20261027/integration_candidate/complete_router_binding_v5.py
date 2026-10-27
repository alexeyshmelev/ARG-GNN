"""Add exact source/CPU/independent router evidence to combinedv5 release."""
from pathlib import Path
HERE=Path(__file__).parent
path=HERE/'qualification_combined_v5.py'
text=path.read_text()
text=text.replace("ADAPTIVE_MANIFEST = BASE+'integration_candidate/source_pins_adaptive_v2.json'", "ADAPTIVE_MANIFEST = BASE+'integration_candidate/source_pins_adaptive_v2.json'\nROUTER_MANIFEST = BASE+'architecture/qualified_residency_router/source_pins_v1.json'\nROUTER_CPU = BASE+'architecture/qualified_residency_router/CPU_RESULTS_v1.json'")
addition='''def validate_router(value):
    need(type(value) is dict and set(value)=={'source_manifest_ref','independent_source_review_ref','CPU_contracts_ref'},
         'Exact declared qualified pre-forward router evidence')
    manifest_ref=c.file_ref(ROUTER_MANIFEST)
    need(value['source_manifest_ref']==manifest_ref and value['CPU_contracts_ref']==c.file_ref(ROUTER_CPU),
         'Exact frozen router source and CPU boundary/interface evidence')
    authenticate_manifest(manifest_ref)
    review=read(value['independent_source_review_ref']);tests=read(value['CPU_contracts_ref'])
    need(review.get('passed') is True and review.get('source_manifest_ref')==manifest_ref,
         'Independent review binds the exact two-route source manifest')
    need(tests.get('schema')=='TreeLoom_qualified_CPUcontext_capacity_router_CPU_v1' and tests.get('passed') is True
         and tests.get('CUDA_initialized') is False and tests.get('scientific_payloads_opened') is False
         and tests.get('scalar_boundaries',{}).get('passed') is True
         and tests.get('factory_interface',{}).get('passed') is True
         and tests.get('factory_interface',{}).get('constructor_failure_propagates_without_fallback') is True
         and tests.get('tiny_model',{}).get('passed') is True
         and tests.get('tiny_model',{}).get('all49_gradients') is True
         and tests.get('tiny_model',{}).get('all49_updated_parameters') is True
         and tests.get('tiny_model',{}).get('all147_Adam_tensors') is True,
         'CPU route boundaries,original-factory interface and tiny full-state equality required')
    return value


'''
text=text.replace('def validate_release(reference):',addition+'def validate_release(reference):')
text=text.replace("    selected = configuration(release.get('configuration'))", "    selected = configuration(release.get('configuration'))\n    validate_router(release.get('qualified_router'))")
text=text.replace("         deployment_uses_then_latest_checkpoint=True),", "         deployment_uses_then_latest_checkpoint=True,capacity_route_before_forward_required=True,\n         explicit_CPU512_fallback_qualified=True,no_mid_update_fallback=True),")
path.write_text(text)
print('Added exact router evidence and declared pre-forward route acceptance;no launch.')
