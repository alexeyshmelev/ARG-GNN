"""CPU-only complete predictor equivalence for the focal-only no_cross view."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import copy,json,time
import torch
from joint_model_v1 import JointAncestryModel,load_observations,prepare_observations,collate_prepared,prepared_to_state,prepared_from_state
from no_cross_view import focal_only_no_cross

ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    batches=[prepare_observations(load_observations(ROOT.parent/f'data/profile_observations_v1/site_{i:06d}.npz')) for i in (0,27)]
    cases=batches+[collate_prepared(batches)]
    results=[]
    for index,batch in enumerate(cases):
        started=time.perf_counter();view=focal_only_no_cross(batch);view_time=time.perf_counter()-started
        torch.manual_seed(101);full=JointAncestryModel(width=64,layers=3);small=copy.deepcopy(full)
        y=full(batch,'no_cross');z=small(view,'no_cross');probe=torch.randn_like(y)
        (y*probe).sum().backward();(z*probe).sum().backward()
        torch.testing.assert_close(y,z,atol=2e-6,rtol=2e-5)
        active=0;worst_absolute=0.;grad_sq=reference_sq=0.
        for (name,p),(other,q) in zip(full.named_parameters(),small.named_parameters()):
            assert name==other and (p.grad is None)==(q.grad is None),name
            if p.grad is not None:
                active+=1;torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=3e-4)
                worst_absolute=max(worst_absolute,float((p.grad-q.grad).abs().max()))
                grad_sq+=float(((p.grad-q.grad).double()**2).sum());reference_sq+=float((p.grad.double()**2).sum())
        assert torch.equal(view.focal_sample,batch.focal_sample) and torch.equal(view.focal_site,batch.focal_site)
        restored=prepared_from_state(prepared_to_state(view))
        torch.testing.assert_close(small(restored,'no_cross'),z,atol=0,rtol=0)
        separately=collate_prepared([focal_only_no_cross(b) for b in batches]) if index==2 else view
        if index==2:torch.testing.assert_close(small(separately,'no_cross'),z,atol=2e-6,rtol=2e-5)
        timing={}
        for label,bb,mm in [('full_context',batch,full),('focal_only',view,small)]:
            with torch.no_grad():mm(bb,'no_cross')
            values=[]
            for repeat in range(3):
                mm.zero_grad(set_to_none=True);start=time.perf_counter();out=mm(bb,'no_cross');forward=time.perf_counter()-start
                start=time.perf_counter();(out*probe).sum().backward();backward=time.perf_counter()-start
                values.append({'forward_seconds':forward,'backward_seconds':backward})
            timing[label]=values
        result={'case':index,'targets':batch.target_count,'view_seconds':view_time,
                'full_nodes':len(batch.parent),'view_nodes':len(view.parent),'full_snp_rows':len(batch.site_tree),'view_snp_rows':len(view.site_tree),
                'logit_max_absolute':float((y-z).abs().max()),'parameter_vjp_max_absolute':worst_absolute,
                'parameter_vjp_relative_l2':(grad_sq/max(reference_sq,1e-100))**.5,'active_parameter_tensors':active,
                'view_statistics':view.statistics,'timing':timing}
        results.append(result);print('Passed case',index,flush=True)
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'CPU only, two real observation packets, no ancestry labels, no optimizer, no GPU',
       'model_sha256':sha256((ROOT/'joint_model_v1.py').read_bytes()).hexdigest(),
       'view_sha256':sha256((ROOT/'no_cross_view.py').read_bytes()).hexdigest(),
       'test_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'all_passed':True,'results':results,
       'qualification':'width64 layers3, logits and every active/inactive parameter VJP; target routing and collation equivalence',
       'timing_limitation':'Three warmed CPU2-thread samples per case; full epoch/device timing remains separate.'}
    (ROOT/'no_cross_view_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
