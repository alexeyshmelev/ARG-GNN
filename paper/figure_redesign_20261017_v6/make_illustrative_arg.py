"""Create a small illustrative ARG only, not a scientific experiment."""
from pathlib import Path
import json
import msprime
OUT=Path(__file__).resolve().parent
attempts=[]
for seed in range(1,4097):
    arg=msprime.sim_ancestry(samples=5,ploidy=1,population_size=1000,
      sequence_length=10000,recombination_rate=3e-8,model="hudson",
      record_full_arg=True,random_seed=seed)
    ts=arg.simplify(keep_unary=False,record_provenance=False)
    if ts.num_trees!=3 or arg.num_trees!=3:
        continue
    signature=[]
    for t in ts.trees():
        signature.append(tuple(sorted(tuple(sorted(t.samples(u))) for u in t.nodes()
          if 1<len(list(t.samples(u)))<5)))
    maxage=max(n.time for n in ts.nodes())
    branches=[ts.node(t.parent(u)).time-ts.node(u).time
      for t in ts.trees() for u in t.nodes() if u>=5 and t.parent(u)>=0]
    accepted=(min(t.span for t in ts.trees())>=1200 and len(set(signature))==3
      and min(branches)/maxage>.025)
    attempts.append({"seed":seed,"accepted":accepted,"spans":[t.span for t in ts.trees()]})
    if accepted:break
else:raise RuntimeError("No acceptable illustrative ARG in fixed bounded search.")
tables=arg.dump_tables()
for i,t in enumerate(arg.trees()):
    candidates=[u for u in t.nodes() if 1<len(list(t.samples(u)))<5 and len(t.children(u))>=2]
    candidates.sort(key=lambda u:tuple(sorted(t.samples(u))))
    u=candidates[i%len(candidates)]
    site=tables.sites.add_row((t.interval.left+t.interval.right)/2,ancestral_state="0")
    tables.mutations.add_row(site=site,node=u,derived_state="1")
tables.sort();arg=tables.tree_sequence()
ts, node_map=arg.simplify(keep_unary=False,record_provenance=False,map_nodes=True)
assert arg.num_trees==ts.num_trees==3
assert all(t.num_roots==1 for t in ts.trees())
assert list(arg.breakpoints())==list(ts.breakpoints())
arg.dump(OUT/"illustrative_full_arg.trees");ts.dump(OUT/"illustrative_tree_sequence.trees")
metadata={"purpose":"Small illustration only, no fitted model, scientific experiment, metrics or targets",
"msprime_version":msprime.__version__,"simulation":{"samples":5,"ploidy":1,"population_size":1000,
"sequence_length":10000,"recombination_rate":3e-8,"model":"hudson","record_full_arg":True,"random_seed":seed},
"selection":"3 intervals, each at least1200bp,3 distinct proper-clade sets, minimum internal branch>2.5%maxage",
"three_tree_candidates_examined":attempts,"breakpoints_bp":list(ts.breakpoints()),
"sample_names":["q1","r1","r2","r3","r4"],"reference_label_examples":["Ana","CHG","WHG","EHG"],
"full_to_display_node_map":node_map.tolist(),
"full_nodes":[{"id":n.id,"time":n.time,"flags":n.flags} for n in arg.nodes()],
"display_trees":[{"index":t.index,"interval":list(t.interval),
"newick":t.as_newick(node_labels={u:f"s{u}" for u in ts.samples()})} for t in ts.trees()],
"mutations":"One hypothetical observed allele on a real branch per interval, not ancestry labels"}
(OUT/"illustrative_tree_sequence.json").write_text(json.dumps(metadata,indent=2)+"\n")
print("Illustrative seed",seed,"breakpoints",list(ts.breakpoints()),"nodes",arg.num_nodes,ts.num_nodes)

