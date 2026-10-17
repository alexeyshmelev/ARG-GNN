"""Narrow native-root bookkeeping. Pure integer/list helpers support source checks."""
def require(ok,message):
 if not ok:raise ValueError(message)
def classify_sample_roots(parent,head,root_of_node,row):
 """Connected root scalar, or tuple of actual isolated sampled roots."""
 nodes=tuple(int(u)for u in row)
 require(nodes and len(set(nodes))==len(nodes),"Distinct complete sampled row required")
 roots=tuple(int(root_of_node[u])for u in nodes)
 require(all(u>=0 for u in roots),"Acyclic rooted native components required")
 if len(set(roots))==1:return roots[0]
 require(all(int(parent[u])==-1 and int(head[u])==-1 and roots[i]==u for i,u in enumerate(nodes)),
  "Only all-sampled/all-isolated holds; mixed forest remains unsupported")
 return nodes
def validate_root_partition(ordered,native_roots,isolated_hold_indices):
 native=tuple(int(u)for u in native_roots);flattened=[];isolated=[]
 for at,root in enumerate(ordered):
  if isinstance(root,tuple):
   require(len(root)>1,"Isolated hold contains multiple actual native roots")
   isolated.append(at);flattened.extend(root)
  else:flattened.append(int(root))
 declared=tuple(isolated_hold_indices)
 require(all(type(i)is int for i in declared)and declared==tuple(isolated),
  "Exact upstream authenticated zero-SNP isolated-hold indices required")
 require(len(flattened)==len(set(flattened))==len(native)and set(flattened)==set(native),
  "Every actual native component assigned exactly once; no hidden/pruned component")
 return tuple(isolated)
def isolated_clades(root,row,parent,head):
 nodes=tuple(int(u)for u in row)
 require(isinstance(root,tuple)and root==nodes and len(set(nodes))==len(nodes)
  and all(int(parent[u])==-1 and int(head[u])==-1 for u in nodes),
  "Exact all-isolated sampled native roots; no virtual root or hidden descendants")
 return {},len(nodes),0
