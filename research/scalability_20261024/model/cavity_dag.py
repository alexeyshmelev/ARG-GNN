"""Isolated hash-consed full-depth cavity operator; no loader, training or launch API."""
from collections import defaultdict
import torch
from torch import nn

def need(ok,message):
    if not ok:raise ValueError(message)

class CavityDAG:
    """Immutable expression identities; learned values are fresh on every call."""
    def __init__(self,parent,base_ids,gate_ids,sample_index):
        self.parent=list(map(int,parent));self.base_ids=list(map(int,base_ids));self.gate_ids=list(map(int,gate_ids))
        self.sample_index=[list(map(int,row)) for row in sample_index]
        n=len(self.parent);need(len(base_ids)==len(gate_ids)==n,"Aligned occurrence sources")
        self.children=[[] for _ in range(n)]
        for child,p in enumerate(self.parent):
            if p>=0:
                need(p<n and p!=child,"Valid forest parent");self.children[p].append(child)
        self.records=[];self.keys={};self.depth=[];self.batches=defaultdict(list)
        self.up=[None]*n;self.aggregate=[None]*n;self.message=[None]*n;self.out=[None]*n
        self.base=[self.add("base",(),(source,)) for source in self.base_ids]
        self.zero=self.add("zero",(),())
        visiting=set()
        def upward(node):
            if self.up[node] is not None:return self.up[node]
            need(node not in visiting,"Acyclic native forest");visiting.add(node)
            messages=[];degree=max(1,len(self.children[node]))
            for child in self.children[node]:
                self.message[child]=self.add("message",(upward(child),),(self.gate_ids[child],degree))
                messages.append(self.message[child])
            self.aggregate[node]=self.add("sum",tuple(messages),())
            self.up[node]=self.add("up",(self.base[node],self.aggregate[node]),())
            visiting.remove(node);return self.up[node]
        for node in range(n):upward(node)
        def outside(node):
            if self.out[node] is not None:return self.out[node]
            p=self.parent[node]
            self.out[node]=self.zero if p<0 else self.add("outside",
                (outside(p),self.base[p],self.aggregate[p],self.message[node]),(self.gate_ids[node],))
            return self.out[node]
        for node in range(n):outside(node)
        self.outputs=[[self.add("leaf",(self.up[node],self.out[node]),()) for node in row] for row in self.sample_index]

    def add(self,operation,inputs,metadata):
        key=(operation,inputs,metadata)
        if key in self.keys:return self.keys[key]
        index=len(self.records);depth=1+max((self.depth[i] for i in inputs),default=-1)
        self.records.append(key);self.keys[key]=index;self.depth.append(depth)
        self.batches[(depth,operation,len(inputs))].append(index)
        return index

    def receipt(self):
        return dict(occurrences=len(self.parent),unique_operations=len(self.records),
            operation_counts={op:sum(record[0]==op for record in self.records) for op in sorted({r[0] for r in self.records})},
            unique_up_states=len(set(self.up)),unique_outside_states=len(set(self.out)),
            unique_sample_contexts=len({i for row in self.outputs for i in row}),
            numeric_states_cached_across_calls=False,persistent_sample_routes_preserved=True)

class CavityOperator(nn.Module):
    def __init__(self,width=8,classes=3):
        super().__init__();self.width=width
        self.up_gate=nn.Linear(9,width);self.down_gate=nn.Linear(9,width)
        self.up_message=nn.Linear(width,width,bias=False);self.up_update=nn.Linear(2*width,width)
        self.up_norm=nn.LayerNorm(width);self.down_message=nn.Linear(width,width,bias=False)
        self.down_update=nn.Linear(width,width);self.down_norm=nn.LayerNorm(width)
        self.leaf_context=nn.Sequential(nn.Linear(2*width,width),nn.Tanh())
        self.head=nn.Linear(width,classes)

    def forward(self,plan,base_sources,gate_sources):
        """Batch equal-depth/op/arity operations; autograd retains shared-use multiplicity."""
        values={};ug=self.up_gate(gate_sources).sigmoid();dg=self.down_gate(gate_sources).sigmoid()
        for _,ids in sorted(plan.batches.items()):
            records=[plan.records[i] for i in ids];op=records[0][0]
            args=[torch.stack([values[r[1][k]] for r in records]) for k in range(len(records[0][1]))]
            if op=="base":result=base_sources[[r[2][0] for r in records]]
            elif op in ("zero","sum"):
                result=base_sources.new_zeros(len(ids),self.width)
                for value in args:result=result+value
            elif op=="message":
                gates=ug[[r[2][0] for r in records]]
                degree=base_sources.new_tensor([r[2][1] for r in records])[:,None]
                result=self.up_message(args[0])*gates/degree
            elif op=="up":result=self.up_norm(args[0]+self.up_update(torch.cat(args,-1)).tanh())
            elif op=="outside":
                cavity=args[0]+args[1]+args[2]-args[3]
                result=self.down_norm(self.down_update(self.down_message(cavity)*dg[[r[2][0] for r in records]]).tanh())
            elif op=="leaf":result=self.leaf_context(torch.cat(args,-1))
            else:raise ValueError("Unknown static expression")
            values.update({index:result[row] for row,index in enumerate(ids)})
        return self.head(torch.stack([torch.stack([values[i] for i in row]) for row in plan.outputs]))

    def dense_reference(self,plan,base_sources,gate_sources):
        """Independent occurrence recursion, retaining a distinct state for each tree."""
        base=base_sources[plan.base_ids];ug=self.up_gate(gate_sources)[plan.gate_ids].sigmoid()
        dg=self.down_gate(gate_sources)[plan.gate_ids].sigmoid()
        up={};aggregate={};message={};out={}
        def visit(node):
            if node in up:return up[node]
            total=base.new_zeros(self.width)
            for child in plan.children[node]:
                message[child]=self.up_message(visit(child))*ug[child]/max(1,len(plan.children[node]))
                total=total+message[child]
            aggregate[node]=total
            up[node]=self.up_norm(base[node]+self.up_update(torch.cat((base[node],total),-1)).tanh())
            return up[node]
        def outside(node):
            if node in out:return out[node]
            p=plan.parent[node]
            out[node]=base.new_zeros(self.width) if p<0 else self.down_norm(self.down_update(
                self.down_message(outside(p)+base[p]+aggregate[p]-message[node])*dg[node]).tanh())
            return out[node]
        for node in range(len(plan.parent)):visit(node)
        return self.head(torch.stack([torch.stack([self.leaf_context(torch.cat((up[node],outside(node)),-1))
            for node in row]) for row in plan.sample_index]))
