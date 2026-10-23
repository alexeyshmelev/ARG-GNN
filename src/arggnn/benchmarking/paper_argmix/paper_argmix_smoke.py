"""Small CPU scientific tests; no benchmark generation or GPU allocation."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import time
import numpy as np
import torch
import tskit
from . import paper_argmix_generator as g
from .paper_argmix_features import LCA,capped_distance,feature_batch,ScopeGuard
from .paper_argmix_model import SpatialBiasAttention,TmrcaAttn


def synthetic_tree():
    tables=tskit.TableCollection(sequence_length=20)
    current=[tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=i%3) for i in range(64)]
    generation=10
    while len(current)>1:
        parents=[]
        for i in range(0,len(current),2):
            parent=tables.nodes.add_row(time=generation)
            for child in current[i:i+2]:
                tables.edges.add_row(0,20,parent,child)
            parents.append(parent)
        current=parents
        generation*=2
    # Sparse inactive node IDs must not inflate per-tree lifting tables.
    for i in range(10000):
        tables.nodes.add_row(time=100000+i)
    tables.sort()
    return tables.tree_sequence()


def run(output):
    output=g.confined(output)
    g.need(not output.exists(),"Fresh smoke output required")
    output.mkdir()
    torch.set_num_threads(1)
    torch.manual_seed(71)
    started=time.monotonic()
    checks=[]
    for path in Path(__file__).parent.glob("*.py"):
        ast.parse(path.read_text())
    checks.append("All package modules parse as plaintext Python")
    ts=synthetic_tree()
    tree=ts.first()
    lca=LCA(tree)
    g.need(len(lca.nodes)==127 and lca.n==10127,"Sparse active-node bound failed")
    rng=np.random.Generator(np.random.PCG64(71))
    a,b=rng.integers(0,127,size=(2,5000))
    observed=lca.mrca(a,b)
    expected=np.asarray([tree.mrca(int(x),int(y)) for x,y in zip(a,b)])
    g.need(np.array_equal(observed,expected),"Vector LCA disagrees with tskit")
    checks.append("5000 vector LCA pairs match tskit; 10000 inactive nodes excluded")
    config=g.metadata(g.CONFIG)
    features=config["features"]
    cohort_order=("NF","Yam","Ana","CHG","EHG","WHG")
    roster=[]
    for i in range(64):
        cohort=cohort_order[(i//2)%len(cohort_order)] if i<40 else "Present"
        roster.append(dict(sample_index=i,sample_id=f"person_{i//2:03d}",haplotype=i%2,
                           stable_haplotype_id=f"person_{i//2:03d}:h{i%2}",observed_cohort=cohort,
                           reference_eligible=i<40,query_eligible=True,inferred_tree_node_id=i))
    focal=[0,1,2,40,41,63]
    x,d,nodes=feature_batch(tree,roster,focal,features)
    g.need(x.shape==(6,29) and d.shape==(6,29,29) and x.dtype==np.uint8 and d.dtype==np.uint16,"Feature geometry differs")
    for row,f in enumerate(focal):
        donors=[i for i,r in enumerate(roster) if r["reference_eligible"] and r["sample_id"]!=roster[f]["sample_id"]]
        expected=sorted(donors,key=lambda i:(tree.tmrca(f,i),roster[i]["stable_haplotype_id"]))[:25]
        g.need(list(nodes[row,1:26])==expected,"Donor identity exclusion or stable raw-time ranking failed")
        ancestor=f
        for column in range(26,29):
            ancestor=tree.parent(ancestor)
            g.need(nodes[row,column]==ancestor,"Ancestor chain differs")
        for i,left in enumerate(nodes[row]):
            for j,right in enumerate(nodes[row]):
                truth=min(abs(int(np.floor(tree.time(tree.mrca(int(left),int(right)))+.5))-int(np.floor(tree.time(int(left))+.5))),1000)
                g.need(int(d[row,i,j])==truth,"Directed distance disagrees with independent scalar implementation")
    g.need(np.any(d!=d.transpose(0,2,1)),"Asymmetric age distances disappeared")
    g.need(set(x[:,1:26].ravel()) <= set(features["tokens"][name] for name in cohort_order),"Donor tokens mixed with target classes")
    checks.append("25 distinct donors, both focal haplotypes excluded, stable ties, 3 ordered ancestors and 29-node asymmetric matrices")
    overflow=capped_distance(np.zeros(5),np.asarray([65535,65536,65537,1000000,999]))
    g.need(list(overflow)==[1000,1000,1000,1000,999],"Linear cap overflow regression")
    checks.append("Linear distance cap precedes uint16 conversion at 65535/65536/65537/1000000")
    # Paper Eq. 3 key term must use k_j with psi(i,j), not k_i or psi(j,i).
    hidden,heads,hops=4,2,9
    attention=SpatialBiasAttention(hidden,0.0,heads).eval()
    with torch.no_grad():
        for projection in (attention.k_proj,attention.v_proj,attention.out_proj):
            projection.weight.copy_(torch.eye(hidden)); projection.bias.zero_()
        attention.q_proj.weight.zero_(); attention.q_proj.bias.zero_()
    inputs=torch.randn(2,3,hidden)
    distance=torch.tensor([[[0,2,3],[5,0,1],[7,6,0]],[[0,3,2],[8,0,6],[1,4,0]]])
    key_emb=torch.randn(hops,hidden)
    zero=torch.zeros_like(key_emb)
    actual=attention(inputs,inputs,inputs,zero,key_emb,zero,distance,torch.ones(2,3,dtype=torch.bool))
    keys=inputs.view(2,3,heads,2).transpose(1,2)
    key_table=key_emb.view(hops,heads,2)
    logits=torch.empty(2,heads,3,3)
    for batch in range(2):
        for head in range(heads):
            for i in range(3):
                for j in range(3):
                    logits[batch,head,i,j]=(keys[batch,head,j]*key_table[distance[batch,i,j],head]).sum()*(2**-.5)
    expected=(logits.softmax(-1)@keys).transpose(1,2).reshape(2,3,hidden)
    g.need(torch.allclose(actual,expected,atol=1e-6),"Asymmetric Eq. 3 key-j correction failed")
    checks.append("Equation-corrected attention matches independent key-j / psi(i,j) scalar formula")
    model=TmrcaAttn(vocab_size=9,num_classes=4,num_layer=2,d_model=32,num_heads=4,dropout=0.0,attn_dropout=0.0,max_rel_distance=1000)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    tokens=torch.from_numpy(x.astype(np.int64))
    distances=torch.from_numpy(d.astype(np.int64))
    labels=torch.arange(6)%4
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    logits=model(tokens,distances,tokens!=2)
    loss=torch.nn.functional.cross_entropy(logits,labels)
    loss.backward()
    g.need(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),"Nonfinite backward gradients")
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
    optimizer.step()
    g.need(any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()),"No training parameter changed")
    model.eval()
    with torch.no_grad(),torch.autocast("cpu",dtype=torch.bfloat16):
        mixed_logits=model(tokens,distances,tokens!=2)
        g.need(torch.isfinite(mixed_logits).all().item(),"BF16 attention scatter accumulator failed")
    checks.append("BF16 autocast forward uses a compatible relative-value accumulator dtype")
    torch.save(model.state_dict(),output/"tiny_checkpoint.pt")
    reloaded=TmrcaAttn(vocab_size=9,num_classes=4,num_layer=2,d_model=32,num_heads=4,dropout=0.0,attn_dropout=0.0,max_rel_distance=1000)
    reloaded.load_state_dict(torch.load(output/"tiny_checkpoint.pt",weights_only=True)); reloaded.eval()
    with torch.inference_mode():
        g.need(torch.equal(model(tokens,distances,tokens!=2),reloaded(tokens,distances,tokens!=2)),"Checkpoint reload output differs")
    checks.append("CPU cross entropy, backward, finite gradients, optimizer update and checkpoint reload match")
    # tskit half-open joins select the right tree at an exact boundary.
    tables=ts.dump_tables()
    edges=tables.edges
    left,right,parent,child=edges.left,edges.right,edges.parent,edges.child
    tables.edges.clear()
    for lo,hi in ((0,10),(10,20)):
        for p,c in zip(parent,child): tables.edges.add_row(lo,hi,int(p),int(c))
    tables.sort()
    joined=tables.tree_sequence()
    join_tree=joined.first(); join_tree.seek(10)
    g.need(join_tree.interval.left==10 and join_tree.interval.right==20,"Half-open truth coordinate join failed")
    checks.append("Half-open interval join is right-continuous at exact boundary")
    result=dict(schema="paper_argmix_cpu_scientific_smoke_v1",status="pass",checks=checks,seconds=time.monotonic()-started,
                examples=6,genealogy_tokens=29,transformer_tokens=30,loss=float(loss),
                torch=torch.__version__,numpy=np.__version__,tskit=tskit.__version__,cuda_allocated=False,
                source_ref=g.file_ref(__file__),limitations=["No full parent generation or native Relate inference in this test","No GPU training or benchmark accuracy claim"])
    (output/"completed.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    run(parser.parse_args().output)
