"""Create immutable proof-role envelopes and authenticate the qualified release."""
from pathlib import Path
import argparse
import json
from arggnn.benchmarking.paper_campaign import runtime as r
from .hooks import source_refs
from .treeloom_resume import authenticate_release

BASE=r.ROOT/'research/preprocessing_optimization_20261026'
ROLES={
 'parser_parity':'observed/qualification.json',
 'parser_adversarial':'review/parser_adversarial.json',
 'tree_parity':'tree/parity.json',
 'tree_adversarial':'review/tree_adversarial.json',
 'native_kernel_safety':'tree/private_safety.json',
 'full_observed_parent':'observed/full_train_qualification.json',
 'full_american_parent':'integration/full_parent_qualification.json',
 'gpu_numerical_parity':'integration/gpu_parity_final.json',
 'resume_contract':'integration/resume_contract.json',
 'independent_review':'review/release_review.json',
 'process_parity':'tree/process_parity.json',
 'full_process_parent':'integration/full_parent_qualification.json',
}

def write_new(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')
    return r.ref(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,choices=[1,4],required=True)
    args=parser.parse_args()
    proofs=[]
    for role,relative in ROLES.items():
        if args.workers==1 and role in ('process_parity','full_process_parent'):
            continue
        path=BASE/relative
        evidence=json.loads(path.read_text())
        assert evidence['status']=='pass', 'Unpassed evidence: '+role
        reference=write_new(BASE/'integration/proofs'/(role+'.json'),dict(
            schema='exact_preprocessing_proof_v1',role=role,status='pass',
            evidence_refs=[r.ref(path)]))
        proofs.append(dict(role=role,reference=reference))
    value=dict(schema='exact_preprocessing_release_v1',status='passed_release',
        encoder_workers=args.workers,native_competitor_changed=False,
        candidate_source_refs=source_refs(),proof_refs=proofs,
        scope='Our exact observed-input preprocessing only. Original model, loss, optimizer, training order, simulations and scoring retained.',
        numerical_contract='Preprocessing arrays match byte for byte. GPU results agree within atol1e-6,rtol1e-5, with unchanged-baseline repeat as control. No bitwise GPU training claim.',
        independent_review_ref=r.ref(BASE/'review/release_review.json'))
    reference=write_new(BASE/'integration/release.json',value)
    authenticate_release(reference)
    print(json.dumps(dict(authenticated_release_ref=reference,encoder_workers=args.workers)))

if __name__=='__main__':main()
