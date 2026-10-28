"""TRAIN-only bounded observed-tree prototype; no fit, score or scientific writes.

The indexed fast path is exact for complete replacement intervals. Persistent
edges are detected by active-edge counts and rejected, never silently omitted.
Generic small tskit sequences can use tree_from_tskit as a reference/fallback.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import numpy as np

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
COHORTS = ('present_day','Bronze_Age','BAA','Yam','Neo','WHG','EHG','Ana','CHG')
CLASSES = ('Ana_nonBAA','CHG_nonBAA','WHG','EHG','Ana_viaBAA','CHG_viaBAA')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path_for(ref):
    p = Path(ref['path'])
    p = p if p.is_absolute() else ROOT / p
    require(p.resolve().is_relative_to(ROOT), 'Source must remain in project')
    return p


def small_json(ref, cap=12*2**20):
    p = path_for(ref)
    require(p.stat().st_size <= cap, 'Small metadata size cap')
    raw = p.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Small metadata SHA mismatch')
    return json.loads(raw)


def prefix_value(raw, key):
    # Only selected unique early fields are read; the 1.3m-row site_join is not.
    matches = list(re.finditer(r'"' + re.escape(key) + r'"\s*:', raw))
    require(len(matches) == 1, 'Expected unique field in bounded inference prefix: ' + key)
    return json.JSONDecoder().raw_decode(raw[matches[0].end():].lstrip())[0]


class KastoreArrays:
    """Read-only Kastore v1 column mappings, not an executable-code loader."""
    DTYPES = ('i1','u1','<i2','<u2','<i4','<u4','<i8','<u8','<f4','<f8')

    def __init__(self, path):
        self.path = Path(path)
        self.arrays = {}
        size = self.path.stat().st_size
        with self.path.open('rb') as stream:
            header = stream.read(64)
            require(header[:8] == b'\x89KAS\r\n\x1a\n', 'Kastore magic')
            require(struct.unpack_from('<H',header,8)[0] == 1, 'Kastore major version 1 required')
            count = struct.unpack_from('<I',header,12)[0]
            require(0 < count <= 128, 'Bounded Kastore descriptor count')
            require(struct.unpack_from('<Q',header,16)[0] == size, 'Kastore file size mismatch')
            descriptors = stream.read(count*64)
            require(len(descriptors) == count*64, 'Complete Kastore descriptors')
            for i in range(count):
                item = descriptors[64*i:64*(i+1)]
                type_id = item[0]
                key_offset,key_size,array_offset,length = struct.unpack_from('<QQQQ',item,8)
                require(type_id < len(self.DTYPES) and 0 < key_size <= 256, 'Kastore descriptor type/key')
                dtype = np.dtype(self.DTYPES[type_id])
                require(key_offset+key_size <= size and array_offset+length*dtype.itemsize <= size,
                        'Kastore array exceeds immutable source')
                stream.seek(key_offset)
                key = stream.read(key_size).decode('utf-8')
                require(key not in self.arrays, 'Duplicate Kastore key')
                self.arrays[key] = (np.memmap(self.path,mode='r',dtype=dtype,
                    offset=array_offset,shape=(length,)) if length else np.empty(0,dtype=dtype))
        self.snapshot = (size,self.path.stat().st_mtime_ns)

    def __getitem__(self,key):
        require((self.path.stat().st_size,self.path.stat().st_mtime_ns) == self.snapshot,
                'Mapped source changed')
        return self.arrays[key]

    def bound(self, order_key, value_key, x, inclusive):
        order,value = self[order_key],self[value_key]
        lo,hi = 0,len(order)
        while lo < hi:
            mid = (lo+hi)//2
            v = value[order[mid]]
            if v < x or (inclusive and v == x): lo = mid+1
            else: hi = mid
        return lo


@dataclass
class CompleteTree:
    left: float
    right: float
    original_ids: np.ndarray
    time: np.ndarray
    parent: np.ndarray
    sample_index: np.ndarray


def make_tree(left, right, original_ids, time, edge_parent, edge_child, sample_ids):
    ids = np.asarray(original_ids,dtype=np.int64)
    parent = np.full(len(ids),-1,dtype=np.int64)
    loc_p,loc_c = np.searchsorted(ids,edge_parent),np.searchsorted(ids,edge_child)
    require(len(np.unique(loc_c)) == len(loc_c), 'Multiple active parents')
    parent[loc_c] = loc_p
    require(np.all(time[loc_p] > time[loc_c]), 'Strict observed parent time ordering')
    sample_index = np.full(len(ids),-1,dtype=np.int64)
    sample_locs = np.searchsorted(ids,sample_ids)
    require(np.array_equal(ids[sample_locs],sample_ids), 'Every observed sample retained')
    sample_index[sample_locs] = np.arange(len(sample_ids))
    return CompleteTree(float(left),float(right),ids,np.asarray(time),parent,sample_index)


def tree_from_tskit(tree, sample_ids):
    """Generic reference for an already resident small TS, including unary/forest."""
    ids = np.asarray(sorted(set(tree.nodes()) | set(map(int,sample_ids))),dtype=np.int64)
    children = np.asarray([u for u in ids if tree.parent(int(u)) != -1],dtype=np.int64)
    parents = np.asarray([tree.parent(int(u)) for u in children],dtype=np.int64)
    times = np.asarray([tree.time(int(u)) for u in ids])
    return make_tree(tree.interval.left,tree.interval.right,ids,times,parents,children,sample_ids)


class ReplacementTreeReader:
    def __init__(self, path, sample_ids, max_edges=100000):
        self.columns = KastoreArrays(path)
        self.sample_ids = np.asarray(sample_ids,dtype=np.int64)
        self.sequence_length = float(self.columns['sequence_length'][0])
        self.max_edges = max_edges
        require(np.all((self.columns['nodes/flags'][self.sample_ids] & 1) == 1), 'Sample flags')

    def at(self, position):
        require(0 <= position < self.sequence_length,'Position outside inferred coverage')
        a = self.columns
        inserted = a.bound('indexes/edge_insertion_order','edges/left',position,True)
        removed = a.bound('indexes/edge_removal_order','edges/right',position,True)
        require(inserted > 0,'No inserted edges at position; use generic fallback')
        order = a['indexes/edge_insertion_order']
        left = float(a['edges/left'][order[inserted-1]])
        start = a.bound('indexes/edge_insertion_order','edges/left',left,False)
        count = inserted-start
        require(0 < count <= self.max_edges,'Per-tree edge cap; no truncation')
        require(inserted-removed == count,'Persistent edges require generic tskit fallback')
        edge_ids = order[start:inserted]
        rights = a['edges/right'][edge_ids]
        require(np.all(rights > position) and np.all(rights == rights[0]),
                'Not a complete replacement interval; use generic fallback')
        right = float(rights[0])
        # Active-count equality proves the graph at x, not its full interval.
        # Intervening events would make insertion-left/common-right too wide.
        if inserted < len(order):
            next_insertion=float(a['edges/left'][order[inserted]])
            require(next_insertion>=right,
                    'Partial future insertion requires generic tskit fallback')
        if removed:
            removal_order=a['indexes/edge_removal_order']
            previous_removal=float(a['edges/right'][removal_order[removed-1]])
            require(previous_removal<=left,
                    'Intervening previous removal requires generic tskit fallback')
        parents,children = a['edges/parent'][edge_ids],a['edges/child'][edge_ids]
        ids = np.unique(np.concatenate((parents,children,self.sample_ids)))
        return make_tree(left,right,ids,a['nodes/time'][ids],parents,children,self.sample_ids)

    def window(self,position,halo):
        require(isinstance(halo,int) and 0 <= halo <= 64,'Explicit bounded halo')
        core = self.at(position)
        before,after = [],[]
        tree = core
        for _ in range(halo):
            if tree.left == 0: break
            previous = self.at(np.nextafter(tree.left,-np.inf))
            require(previous.right == tree.left,'Left neighboring tree boundary mismatch')
            before.append(previous); tree = previous
        tree = core
        for _ in range(halo):
            if tree.right == self.sequence_length: break
            following = self.at(tree.right)
            require(following.left == tree.right,'Right neighboring tree boundary mismatch')
            after.append(following); tree = following
        return list(reversed(before))+[core]+after,len(before)


class SortedVCF:
    """Worker-local bounded byte-seek reader for one sorted plain-text chromosome."""
    def __init__(self,path,panel,max_line_bytes=2**20):
        self.path = Path(path)
        self.stream = self.path.open('rb')
        self.max_line_bytes = max_line_bytes
        self.bytes_read = 0
        try:
            self._header(panel)
        except Exception:
            self.stream.close()
            raise

    def _header(self,panel):
        for _ in range(10000):
            line = self._line()
            require(bool(line),'Missing VCF header')
            if line.startswith(b'##'): continue
            require(line.startswith(b'#CHROM\t'),'VCF column header')
            samples = line.decode().rstrip('\n').split('\t')[9:]
            require(samples == [r['sample_id'] for r in panel[::2]],'VCF sample order mismatch')
            require(all(panel[i]['sample_id']==panel[i+1]['sample_id'] and
                        [panel[i]['haplotype'],panel[i+1]['haplotype']]==[0,1]
                        for i in range(0,len(panel),2)),'Phased panel identity mismatch')
            self.data_start = self.stream.tell(); self.samples = samples
            first = self._line().split(b'\t'); self.chromosome = first[0]
            break
        else: raise ValueError('VCF header line cap')
        self.snapshot = (self.path.stat().st_size,self.path.stat().st_mtime_ns)

    def _line(self):
        line = self.stream.readline(self.max_line_bytes+1)
        require(len(line) <= self.max_line_bytes,'VCF line byte cap')
        self.bytes_read += len(line)
        return line

    def at(self,position_1based):
        require((self.path.stat().st_size,self.path.stat().st_mtime_ns)==self.snapshot,'VCF changed')
        lo,hi = self.data_start,self.snapshot[0]
        found = None
        found_start=None
        for _ in range(80):
            if lo >= hi: break
            mid = (lo+hi)//2
            self.stream.seek(mid)
            if mid != self.data_start: self._line()
            start = self.stream.tell()
            line = self._line()
            if not line: hi=mid; continue
            parts = line.rstrip(b'\n').split(b'\t')
            require(parts[0] == self.chromosome,'Single-chromosome VCF required')
            pos = int(parts[1])
            if pos == position_1based: found=parts;found_start=start;found_end=self.stream.tell();break
            if pos < position_1based: lo=self.stream.tell()
            else: hi=mid
        if found is None:
            self.stream.seek(lo)
            for _ in range(3):
                start=self.stream.tell()
                line=self._line()
                if not line: break
                parts=line.rstrip(b'\n').split(b'\t')
                pos=int(parts[1])
                if pos==position_1based:found=parts;found_start=start;found_end=self.stream.tell();break
                if pos>position_1based:break
        require(found is not None,'Exact observed VCF position not found')
        # Local neighbor checks reject duplicated POS and local ordering faults.
        # The source's whole-file sortedness remains an inherited input contract.
        if found_start>self.data_start:
            size=4096
            while True:
                begin=max(self.data_start,found_start-size)
                self.stream.seek(begin);block=self.stream.read(found_start-begin);self.bytes_read+=len(block)
                stripped=block.rstrip(b'\n')
                if b'\n' in stripped or begin==self.data_start:break
                size*=2;require(size<=2*self.max_line_bytes,'Previous VCF line bound')
            previous=stripped.rsplit(b'\n',1)[-1].split(b'\t')
            require(previous[0]==self.chromosome and int(previous[1])<position_1based,
                    'Duplicate POS or non-increasing VCF coordinate')
        self.stream.seek(found_end);following=self._line()
        if following:
            following=following.split(b'\t')
            require(following[0]==self.chromosome and int(following[1])>position_1based,
                    'Duplicate POS or non-increasing VCF coordinate')
        require(len(found)==9+len(self.samples),'VCF column count')
        alleles=[found[3].decode()]+found[4].decode().split(',')
        require(1 < len(alleles) <= 4 and len(set(alleles))==len(alleles)
                and all(v in 'ACGT' and len(v)==1 for v in alleles),'Nucleotide alleles required')
        format_keys=found[8].decode().split(':')
        require('GT' in format_keys,'GT required'); gi=format_keys.index('GT')
        calls=[]
        for field in found[9:]:
            gt=field.decode().split(':')[gi]; pair=gt.split('|')
            require(len(pair)==2,'Phased diploid genotypes required')
            for v in pair:
                call=-1 if v=='.' else int(v)
                require(-1 <= call < len(alleles),'Genotype allele index out of bounds')
                calls.append(call)
        return np.asarray(calls,dtype=np.int8),alleles

    def close(self): self.stream.close()


class TrainWindowDataset:
    """Registry-addressed dataset; labels never enter graph or genotype builders."""
    def __init__(self, parent_id, refs, halo=2):
        require(parent_id in tuple(f'ancestralpaths_european_train_{i:03d}' for i in range(3)),
                'Only the three frozen TRAIN parents are allowed')
        self.parent_id,self.refs,self.halo=parent_id,refs,halo
        registry=small_json(refs['sampled_registry_ref'])
        require(registry['split']=='train' and registry['parent_id']==parent_id and
                registry['observation_ref']==refs['inference_ref'] and
                registry['labels_present'] is False and registry['query_truth_read'] is False,
                'Observed TRAIN registry contract')
        self.records={r['example_index']:r for r in registry['records']}
        require(sorted(self.records)==list(range(registry['examples'])),'Exact registry identities')
        require(len(registry['records'])==len(self.records),'Duplicate example_index')
        targets=small_json(refs['training_targets_ref'])
        require(targets['split']=='train' and targets['registry_ref']==refs['sampled_registry_ref']
                and tuple(targets['native_path_names'])==CLASSES,'Frozen TRAIN targets contract')
        target_path=path_for(targets['arrays_ref']);raw=target_path.read_bytes()
        require(len(raw)<8*2**20 and hashlib.sha256(raw).hexdigest()==targets['arrays_ref']['sha256'],
                'Bounded authenticated target array')
        with np.load(target_path,allow_pickle=False) as arrays:
            self.labels=arrays['native_labels_0based'].copy()
        require(len(self.labels)==len(self.records) and np.all((self.labels>=0)&(self.labels<6)),
                'Missing/invalid labels fail closed; never drop targets')
        infer_path=path_for(refs['inference_ref'])
        with infer_path.open('rb') as stream: prefix=stream.read(16384).decode()
        self.tree_ref=prefix_value(prefix,'inferred_tree')
        panel_ref=prefix_value(prefix,'observed_sample_manifest')
        self.offset=float(prefix_value(prefix,'local_to_original_offset_bp'))
        require(prefix_value(prefix,'genotypes.vcf')==refs['observed_vcf_ref']['sha256'],
                'VCF does not match native inference input')
        self.panel=small_json(panel_ref)
        require([r['sample_index'] for r in self.panel]==list(range(len(self.panel))),'Panel indices')
        self.sample_ids=np.asarray([r['observed_node_id'] for r in self.panel])
        self.leaf_features=np.zeros((len(self.panel),17),dtype=np.float32)
        for i,row in enumerate(self.panel):
            self.leaf_features[i,COHORTS.index(row['observed_cohort'])]=1
            ref=row['observed_reference_class']
            require((ref is None if row['role']=='query' else ref in range(4)),
                    'Only known reference classes; query ancestry forbidden')
            self.leaf_features[i,9+(4 if ref is None else ref)]=1
            self.leaf_features[i,14]=np.log1p(row['age_generations'])/np.log(10001)
            self.leaf_features[i,15+(row['role']=='pure_source_reference')]=1
        self._pid=None;self._trees=None;self._vcf=None
        self.source_evidence=dict(refs=refs,tree_ref=self.tree_ref,panel_ref=panel_ref,
            inference_prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
            large_source_checksums='previous authenticated references; not rescanned in this bounded probe')

    def __getstate__(self):
        state=self.__dict__.copy()
        state.update(_pid=None,_trees=None,_vcf=None)
        return state

    def _handles(self):
        if self._pid != os.getpid():
            if self._vcf is not None:self._vcf.close()
            self._pid=os.getpid()
            self._trees=ReplacementTreeReader(path_for(self.tree_ref),self.sample_ids)
            self._vcf=SortedVCF(path_for(self.refs['observed_vcf_ref']),self.panel)
        return self._trees,self._vcf

    def __len__(self): return len(self.records)
    def __getitem__(self,index): return self.batch([int(index)])

    def batch(self,example_indices,context_specs=None,max_nodes=1000000):
        indices=list(map(int,example_indices));require(bool(indices),'Nonempty target draw list')
        trees,vcf=self._handles();windows={};entries=[];genotypes=[];alleles=[]
        for index in indices:
            row=self.records[index]; s=row['sample_index'];panel=self.panel[s]
            require(row['sample_id']==panel['sample_id'] and row['haplotype']==panel['haplotype']
                    and row['query_node_id']==self.sample_ids[s] and panel['role']=='query'
                    and row['cohort']==panel['observed_cohort'],'Focal identity join')
            position=row['position_local_bp']
            require(position+self.offset==row['position_original_bp'],'Coordinate offset mismatch')
            require(trees.columns['sites/position'][row['native_site_id']]==position,'Native site mismatch')
            core=trees.at(position)
            if context_specs is None:
                key=(core.left,core.right)
                if key not in windows:
                    window,core_index=trees.window(position,self.halo)
                    bounds=np.asarray([(t.left,t.right) for t in window])
                    spans=bounds[:,1]-bounds[:,0]
                    coords=bounds.mean(axis=1)
                    spec=dict(coordinates=coords,weights=spans/spans.sum(),
                              density=np.zeros(len(coords)),forced=np.zeros(len(coords),bool),
                              domain=np.asarray([bounds[0,0],bounds[-1,1]]),kind=0)
                    windows[key]=(window,core_index,spec)
            else:
                spec=context_specs[index]
                coords=np.asarray(spec['coordinates'],dtype=np.float64)
                require(0<len(coords)<=129 and np.all(np.isfinite(coords)) and
                        np.all((coords>=0)&(coords<trees.sequence_length)), 'Bounded context coordinates')
                require(len(spec['weights'])==len(spec['density'])==len(spec['forced'])==len(coords),
                        'Coordinate multiplicity and weighting dimensions')
                require(np.isclose(np.sum(spec['weights']),1) and np.all(np.asarray(spec['weights'])>=0),
                        'Normalized nonnegative context weights')
                require(len(spec['domain'])==2 and spec['domain'][0]<spec['domain'][1] and
                        np.all(np.isfinite(spec['density'])) and np.all(np.asarray(spec['density'])>=0),
                        'Finite proposal densities and nonempty coordinate domain')
                key=(core.left,core.right,float(position),tuple(coords),tuple(spec['weights']),
                     tuple(spec['density']),tuple(spec['forced']),tuple(spec['domain']),int(spec['kind']))
                if key not in windows:
                    by_interval={(core.left,core.right):core}
                    for x in coords:
                        tree=trees.at(float(x));by_interval[(tree.left,tree.right)]=tree
                    window=[by_interval[b] for b in sorted(by_interval)]
                    core_index=[(t.left,t.right) for t in window].index((core.left,core.right))
                    windows[key]=(window,core_index,spec)
            entries.append((row,key));g,a=vcf.at(exact_vcf_position(row['position_original_bp']))
            genotypes.append(g);alleles.append(a)
        require(sum(len(t.parent) for w,c,s in windows.values() for t in w)<=max_nodes,
                'Packed node budget exceeded; never truncate selected targets')
        node_ids=[];times=[];parents=[];samples=[];node_tree=[];intervals=[];window_ids=[];core_mask=[]
        leaf_nodes=[];tree_for_key={};node_offset=0
        context_coords=[];context_tree=[];context_weight=[];context_density=[];context_forced=[];context_window=[]
        context_domains=[];context_kinds=[]
        for wid,(key,(window,core_index,spec)) in enumerate(windows.items()):
            first_tree=len(intervals)
            context_domains.append(spec['domain']);context_kinds.append(spec['kind'])
            for x,w,q,forced in zip(spec['coordinates'],spec['weights'],spec['density'],spec['forced']):
                matching=[j for j,t in enumerate(window) if t.left<=x<t.right]
                require(len(matching)==1,'Exactly one containing context tree')
                context_coords.append(x);context_tree.append(first_tree+matching[0]);context_weight.append(w)
                context_density.append(q);context_forced.append(forced);context_window.append(wid)
            for j,tree in enumerate(window):
                tid=len(intervals)
                if j==core_index:tree_for_key[key]=tid
                parent=tree.parent.copy();parent[parent>=0]+=node_offset
                node_ids.extend(tree.original_ids);times.extend(tree.time);parents.extend(parent)
                samples.extend(tree.sample_index);node_tree.extend([tid]*len(parent))
                leaf_nodes.append(np.searchsorted(tree.original_ids,self.sample_ids)+node_offset)
                intervals.append((tree.left,tree.right));window_ids.append(wid);core_mask.append(j==core_index)
                node_offset+=len(parent)
        parents=np.asarray(parents,dtype=np.int64);children=np.flatnonzero(parents>=0)
        encoded_alleles=np.full((len(entries),4),'',dtype='U1')
        for i,a in enumerate(alleles):encoded_alleles[i,:len(a)]=a
        site_positions=trees.columns['sites/position']
        site_ids=[];site_trees=[];site_genotypes=[];site_allele_count=[]
        genotype_cache={r['native_site_id']:(g,a) for (r,key),g,a in zip(entries,genotypes,alleles)}
        for tid,(left,right) in enumerate(intervals):
            begin,end=np.searchsorted(site_positions,[left,right],side='left')
            require(end-begin<=10000 and len(site_ids)+end-begin<=50000,
                    'Bounded window site cap exceeded; do not truncate')
            for sid in range(int(begin),int(end)):
                if sid not in genotype_cache:
                    genotype_cache[sid]=vcf.at(exact_vcf_position(site_positions[sid]+self.offset))
                g,a=genotype_cache[sid]
                site_ids.append(sid);site_trees.append(tid);site_genotypes.append(g);site_allele_count.append(len(a))
        return dict(parent=parents,edge_index=np.stack((children,parents[children])),
            node_original_id=np.asarray(node_ids,dtype=np.int64),node_time=np.asarray(times,dtype=np.float64),
            node_sample_index=np.asarray(samples,dtype=np.int64),node_tree_index=np.asarray(node_tree,dtype=np.int64),
            leaf_node_index=np.asarray(leaf_nodes,dtype=np.int64),leaf_features=self.leaf_features.copy(),
            tree_interval_local_bp=np.asarray(intervals,dtype=np.float64),
            tree_interval_original_bp=np.asarray(intervals,dtype=np.float64)+self.offset,
            coordinate_offset_bp=np.asarray(self.offset),
            context_coordinate_local_bp=np.asarray(context_coords),
            context_tree_index=np.asarray(context_tree,dtype=np.int64),
            context_window_index=np.asarray(context_window,dtype=np.int64),
            context_quadrature_weight=np.asarray(context_weight),
            context_proposal_density_per_bp=np.asarray(context_density),
            context_forced_focal=np.asarray(context_forced,dtype=bool),
            window_context_domain_local_bp=np.asarray(context_domains),
            window_context_kind=np.asarray(context_kinds,dtype=np.int8),
            tree_window_index=np.asarray(window_ids,dtype=np.int64),tree_core_mask=np.asarray(core_mask,dtype=bool),
            tree_at_coverage_left=np.asarray([a==0 for a,b in intervals]),
            tree_at_coverage_right=np.asarray([b==trees.sequence_length for a,b in intervals]),
            focal_tree_index=np.asarray([tree_for_key[k] for r,k in entries]),
            focal_sample_index=np.asarray([r['sample_index'] for r,k in entries]),
            focal_example_index=np.asarray(indices,dtype=np.int64),
            focal_position_local_bp=np.asarray([r['position_local_bp'] for r,k in entries]),
            focal_position_original_bp=np.asarray([r['position_original_bp'] for r,k in entries]),
            focal_native_site_id=np.asarray([r['native_site_id'] for r,k in entries]),
            focal_genotypes=np.asarray(genotypes,dtype=np.int8),focal_alleles=encoded_alleles,
            focal_allele_count=np.asarray(list(map(len,alleles)),dtype=np.int8),
            reference_class=np.asarray([-1 if r['observed_reference_class'] is None else r['observed_reference_class'] for r in self.panel]),
            sample_age=np.asarray([r['age_generations'] for r in self.panel],dtype=np.float64),
            observed_cohort=np.asarray([COHORTS.index(r['observed_cohort']) for r in self.panel]),
            query_mask=np.asarray([r['role']=='query' for r in self.panel]),
            site_tree_index=np.asarray(site_trees,dtype=np.int64),
            site_native_id=np.asarray(site_ids,dtype=np.int64),
            site_position_local_bp=np.asarray(site_positions[site_ids],dtype=np.float64),
            site_position_original_bp=np.asarray(site_positions[site_ids],dtype=np.float64)+self.offset,
            site_genotypes=np.asarray(site_genotypes,dtype=np.int8).reshape(-1,len(self.panel)),
            site_allele_count=np.asarray(site_allele_count,dtype=np.int8),
            labels=self.labels[indices].copy())


def generic_small_reader(path,sample_ids,max_bytes=32*2**20):
    """Explicit bounded generic fallback, never auto-load the 8.8 GB TRAIN file."""
    import tskit
    require(Path(path).stat().st_size<=max_bytes,'Generic fallback size cap')
    ts=tskit.load(path)
    require(np.array_equal(ts.samples(),sample_ids),'Generic sample identity mismatch')
    return ts


def uniform_target_draws(parent_sizes,batch_size,rng):
    """IID draws from every original registry row, independent of labels/graph size."""
    parents=list(parent_sizes)
    sizes=np.asarray([parent_sizes[p] for p in parents],dtype=np.int64)
    require(np.all(sizes>0) and batch_size>0,'Positive complete registry sizes and batch')
    offsets=np.concatenate(([0],np.cumsum(sizes)))
    global_ids=rng.integers(0,offsets[-1],size=batch_size)
    parent_ids=np.searchsorted(offsets[1:],global_ids,side='right')
    return [(parents[p],int(g-offsets[p])) for g,p in zip(global_ids,parent_ids)]


def node_budget_microbatches(packet,max_nodes):
    """Partition complete windows without dropping target draws; no optimizer here.

    Return window and target index lists. An oversized window fails. Caller sums
    target losses divided by the ORIGINAL fixed draw count across microbatches,
    then makes one optimizer step. Independent steps per variable-size group
    would not preserve the same unbiased fixed-B stochastic objective.
    """
    require(max_nodes>0,'Positive node budget')
    windows=np.unique(packet['tree_window_index'])
    groups=[];current=[];nodes=0
    for w in windows:
        tree_ids=np.flatnonzero(packet['tree_window_index']==w)
        n=int(np.isin(packet['node_tree_index'],tree_ids).sum())
        require(n<=max_nodes,'Window exceeds node budget; do not drop targets')
        if current and nodes+n>max_nodes: groups.append(current);current=[];nodes=0
        current.append(int(w));nodes+=n
    if current:groups.append(current)
    result=[]
    for group in groups:
        tree_ids=np.flatnonzero(np.isin(packet['tree_window_index'],group))
        target_ids=np.flatnonzero(np.isin(packet['focal_tree_index'],tree_ids))
        result.append(dict(window_ids=group,target_indices=target_ids,
                           loss_denominator=len(packet['focal_example_index'])))
    require(sorted(np.concatenate([g['target_indices'] for g in result]).tolist())==
            list(range(len(packet['focal_example_index']))),'Every target draw must appear once')
    return result


def uniform_coordinate_context(rng,focal,radius,draw_count,sequence_length):
    """Forced focal plus K-1 IID uniform bp draws in the clipped radius domain.

    Density q=1/domain_length applies to random draws. Their normalized spatial
    mean weights are 1/(K-1). The forced focal has zero quadrature weight and zero continuous density,
    and remains available to the target head. Duplicate containing trees retain
    all coordinate draws and weights. Do NOT multiply again by tree span.
    """
    require(radius>0 and 2<=draw_count<=129 and 0<=focal<sequence_length,
            'Finite positive radius, focal and bounded context draw count')
    left,right=max(0.,focal-radius),min(sequence_length,focal+radius)
    coordinates=np.concatenate(([focal],rng.uniform(left,right,size=draw_count-1)))
    return dict(coordinates=coordinates,
                weights=np.concatenate(([0.],np.full(draw_count-1,1/(draw_count-1)))),
                density=np.concatenate(([0.],np.full(draw_count-1,1/(right-left)))),
                forced=np.arange(draw_count)==0,domain=np.array([left,right]),kind=1)


def exact_vcf_position(original_position):
    require(np.isfinite(original_position) and original_position>=0 and
            float(original_position).is_integer(),'Original VCF coordinate must be exactly integral')
    return int(original_position)+1
