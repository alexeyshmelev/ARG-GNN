from pathlib import Path
import hashlib,json
r=Path('/workspace-SR003.nfs2/ARG-GNN')
pairs=[('anchor_compact_cuda_bootstrap_v2','475f0e30595687c14b4ea28ec6fb621cf4aaae55974eb866ec3dd6f0993f028c','b68c28fcfa8020ca82a1f3827edf171a93a785f2c646904fff207ff5399dd27f'),('gpu_pool_research_v3','4b41143ea5e9d52a34f5d79e6d25f577dc3b5451c50f8707ac2de2732e3df6e3','5a639412ee741e96a8da1b536d0b1cec885355a7e378cf30d340903479d99abf')]
for name,expected_receipt,expected_manifest in pairs:
    d=r/'research/reviews'/f'{name}_independent';a=r/'research/reviews'/f'{name}_author'
    snapshot=d/'author_final_attempt02';snapshot.mkdir(exist_ok=True)
    assert hashlib.sha256((a/'review_receipt.json').read_bytes()).hexdigest()==expected_receipt
    assert hashlib.sha256((a/'artifact_manifest_v1.json').read_bytes()).hexdigest()==expected_manifest
    manifest=json.loads((a/'artifact_manifest_v1.json').read_text());receipt=json.loads((a/'review_receipt.json').read_text());checks={}
    for p,h in manifest['artifact_sha256'].items():
        raw=(a/p).read_bytes();actual=hashlib.sha256(raw).hexdigest();assert actual==h,p
        checks[p]=actual;t=snapshot/p;t.parent.mkdir(parents=True,exist_ok=True)
        if t.exists():assert t.read_bytes()==raw
        else:t.write_bytes(raw)
    for p,h in manifest['source_sha256'].items():assert hashlib.sha256((r/p).read_bytes()).hexdigest()==h,p
    assert receipt['actual_execution_admitted'] is False
    for p in ['review_receipt.json','artifact_manifest_v1.json']:
        raw=(a/p).read_bytes();t=snapshot/p
        if t.exists():assert t.read_bytes()==raw
        else:t.write_bytes(raw)
    initial=json.loads((d/'author_final_attempt01/minimal_actual_permit_contract.json').read_text())
    final=json.loads((snapshot/'minimal_actual_permit_contract.json').read_text())
    changed=[k for k in set(initial)|set(final) if initial.get(k)!=final.get(k)]
    assert changed==['required_root_permit_kind'],changed
    assert final['required_root_permit_kind']=='root_compact_cuda_bootstrap_v1_permit'
    assert hashlib.sha256((snapshot/'minimal_actual_permit_contract.json').read_bytes()).hexdigest()=='1584bf83ea61d933e38e509e676f109b3a8610cd984bfe4e61cf6b6331439b94'
    test=(d/'test_author_contract_attempt01.py').read_text().replace('author_final_attempt01','author_final_attempt02')
    if name.startswith('anchor'):
        paths=list(json.loads((d/'source_snapshot_manifest.json').read_text()));finalmanifest={}
        for p in paths:
            raw=(r/p).read_bytes();t=d/'sealed_source_final'/p;t.parent.mkdir(parents=True,exist_ok=True)
            if t.exists():assert t.read_bytes()==raw
            else:t.write_bytes(raw)
            finalmanifest[p]=hashlib.sha256(raw).hexdigest()
        old_doc=(d/'sealed_source/docs/anchor_compact_cuda_bootstrap_v2.md').read_text();new_doc=(r/'docs/anchor_compact_cuda_bootstrap_v2.md').read_text()
        assert old_doc.replace('use kind root_compact_cuda_bootstrap_v2_permit','use the inherited schema kind root_compact_cuda_bootstrap_v1_permit')==new_doc,'Unexpected documentation change'
        assert finalmanifest['docs/anchor_compact_cuda_bootstrap_v2.md']=='fc3285feb7e1b411ea6582544ffdf4061177cd99c1c0456a79317d9e5ae7f668'
        (d/'source_snapshot_final_manifest.json').write_text(json.dumps(finalmanifest,indent=2,sort_keys=True)+'\n')
        test+='''\ndef test_final_documentation_kind_matches_inherited_schema():\n    import re\n    text=(Path(__file__).parent/'sealed_source_final/docs/anchor_compact_cuda_bootstrap_v2.md').read_text()\n    kinds=re.findall(r'root_compact_cuda_bootstrap_v\\d+_permit',text)\n    assert kinds and set(kinds)=={'root_compact_cuda_bootstrap_v1_permit'}\n'''
    test+='''\ndef test_fixed_contract_is_inactive_and_matches_all_core_pins():\n    contract=json.loads((Path(__file__).parent/'author_final_attempt02/minimal_actual_permit_contract.json').read_text())\n    assert contract['actual_execution_admitted'] is False\n    assert contract['bootstrap_source']['sha256']=='d6f1504901b926d45a59664fcc9d211892eb087586aca1a53d34256e05db1fe5'\n    assert contract['bootstrap_config']['sha256']=='bf94468e5b8d613e027c1d46fbb7991a672870d5a6f0159ec9511af1d8285dac'\n    assert contract['pool_extension']['source']['sha256']=='10176c2b2580aa0e27b46aabe3ceadf88981d5e279408277d70e830675045504'\n    assert contract['pool_extension']['config']['sha256']=='e67fce8bbd59042f3deba8bbf434c9d8759f1f9d26781df74eccf84b432e29c7'\n'''
    (d/'test_author_contract_v2.py').write_text(test)
    followup={'kind':'independent_author_artifact_correction_audit','verdict':'PASS','actual_execution_admitted':False,'all_initial_failure_evidence_preserved':True,'initial_contract_field_only_change':changed,'final_contract_sha256':'1584bf83ea61d933e38e509e676f109b3a8610cd984bfe4e61cf6b6331439b94','author_receipt_sha256':expected_receipt,'author_manifest_sha256':expected_manifest,'all_author_manifest_artifacts_verified':checks,'doc_correction_only_kind_and_inherited_schema_wording':name.startswith('anchor'),'all_four_core_source_config_hashes_unchanged':True}
    (d/'author_correction_audit_v2.json').write_text(json.dumps(followup,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'review':name,'final_author_artifacts_verified':len(checks),'only_contract_field_changed':changed,'final_author_receipt_sha256':expected_receipt}))
