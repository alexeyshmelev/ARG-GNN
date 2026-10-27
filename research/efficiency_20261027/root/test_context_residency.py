"""Small differential CUDA tests of the exact decoder storage transition."""
from pathlib import Path
import argparse,copy,hashlib,json,os,time
import torch
from torch import nn
from research.benchmark_rebuild_20261024.american.treeloom_speed.decoder_v2 import decode_parent_backward
from research.benchmark_rebuild_20261024.american.treeloom_speed.gru_v2 import install_chunked_gru
from .context_residency import decode_resident_parent_backward

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_gru=nn.GRU(11,8)
        self.reverse_gru=nn.GRU(11,8)
        self.classifier=nn.Linear(24,3)
        install_chunked_gru(self,chunk_holds=1024)

def run(model,context,geometry,holds,columns,labels,tile,resident):
    model.zero_grad(set_to_none=True)
    outputs=[]
    hook=model.classifier.register_forward_hook(lambda module,inputs,out:outputs.append(out.detach().cpu()))
    try:
        function=decode_resident_parent_backward if resident else decode_parent_backward
        loss,gradient,receipt=function(model,context,geometry,holds,columns,labels,tile_haplotypes=tile)
        return loss.detach().cpu(),gradient.cpu(),{n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()},torch.cat(outputs),receipt
    finally:hook.remove()

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path('/workspace-SR003.nfs2/ARG-GNN')
    output=Path(args.output).resolve()
    assert output.is_relative_to(root/'research/efficiency_20261027') and not output.exists()
    assert os.environ.get('ARGGNN_PHYSICAL_GPU') in ('3','4','5')
    assert os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') and ',' not in os.environ['CUDA_VISIBLE_DEVICES']
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.manual_seed(5829)
    cases=[]
    for t,h,tile,strided in ((1,1,1,False),(5,7,3,False),(9,7,1,True),(1025,7,3,False),(1025,7,128,True)):
        x=torch.randn(t,h,16 if strided else 8)
        if strided:x=x[:,:,::2]
        geometry=torch.randn(t,3)
        holds=torch.tensor([0,t-1,t//2,t-1],dtype=torch.long)
        columns=torch.tensor([0,h-1,0,h-1],dtype=torch.long)
        labels=torch.tensor([0,2,1,0],dtype=torch.long)
        model=Toy().cuda()
        a=run(model,x,geometry,holds,columns,labels,tile,False)
        b=run(model,x.cuda(),geometry,holds,columns,labels,tile,True)
        errors={}
        for name,i in (('loss',0),('context_gradient',1),('logits',3)):
            torch.testing.assert_close(a[i],b[i],atol=1e-6,rtol=1e-5)
            errors[name]=float((a[i]-b[i]).abs().max())
        for name in a[2]:torch.testing.assert_close(a[2][name],b[2][name],atol=1e-6,rtol=1e-4)
        assert torch.equal(a[3].argmax(-1),b[3].argmax(-1))
        for key in ('completed_decoder_tiles','zero_focal_tiles','native_focals','full_sequence_per_tile'):
            assert a[4][key]==b[4][key]
        assert not bool(b[1][:,1:max(1,h-1)].any())
        cases.append(dict(holds=t,haplotypes=h,tile=tile,strided_input=strided,errors=errors,
            parameter_gradient_max_error=max(float((a[2][n]-b[2][n]).abs().max()) for n in a[2]),argmax_disagreements=0,
            full_sequence=True,duplicate_focals=True,zero_target_tiles=a[4]['zero_focal_tiles']))
    rejected=[]
    for name,cols,labs in [('outside_column',torch.tensor([7]),torch.tensor([0])),('empty_targets',torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long))]:
        try:
            decode_resident_parent_backward(model,torch.zeros(3,7,8,device='cuda'),torch.zeros(3,3),torch.zeros(len(labs),dtype=torch.long),cols,labs,tile_haplotypes=3)
        except ValueError:rejected.append(name)
        else:raise AssertionError('Invalid input was accepted: '+name)
    output.parent.mkdir(parents=True,exist_ok=True)
    receipt=dict(status='pass',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=cases,
        rejected=rejected,device=os.environ['ARGGNN_PHYSICAL_GPU'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        actual_model=False,scope='Differential decoder edge cases only. Actual49parameter fullmodel parity is recorded separately.')
    output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
