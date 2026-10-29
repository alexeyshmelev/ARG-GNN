"""Observation-only, worker-local native site-window extraction.

No target label file is opened here. Complete native trees and all their SNPs
are retained. A bounded LRU caches immutable tree payloads, never activations.
"""
from collections import OrderedDict
import os
from pathlib import Path
import numpy as np
from research.redesign_20261028.data.window_data import (
    COHORTS, ReplacementTreeReader, SortedVCF, exact_vcf_position, path_for, small_json, require)

MODEL_KEYS = (
    'parent', 'node_time', 'node_tree_index', 'leaf_node_index', 'leaf_features',
    'tree_interval_local_bp', 'tree_window_index', 'tree_core_mask',
    'focal_tree_index', 'focal_sample_index', 'focal_site_index', 'focal_position_local_bp',
    'focal_genotypes', 'focal_allele_count', 'reference_class', 'sample_age',
    'observed_cohort', 'query_mask', 'site_tree_index', 'site_position_local_bp',
    'site_genotypes', 'site_allele_count', 'sample_id', 'tree_sample_id',
    'context_coordinate_local_bp', 'context_tree_index', 'context_window_index',
    'context_quadrature_weight', 'context_proposal_density_per_bp', 'context_forced_focal',
    'window_context_domain_local_bp', 'window_context_kind')


class ObservationSource:
    def __init__(self, parent_id, binding, cache_bytes=64*2**20):
        require(parent_id in tuple(f'ancestralpaths_european_train_{i:03d}' for i in range(3)),
                'Only registered original TRAIN observations are allowed')
        self.parent_id, self.binding, self.cache_limit = parent_id, binding, int(cache_bytes)
        self.registry = small_json(binding['refs']['sampled_registry_ref'])
        require(self.registry['split']=='train' and self.registry['parent_id']==parent_id and
                self.registry['labels_present'] is False and self.registry['query_truth_read'] is False,
                'Canonical observation registry contract')
        self.records = {int(row['example_index']): row for row in self.registry['records']}
        require(sorted(self.records)==list(range(self.registry['examples'])), 'Canonical contiguous target rows')
        require(self.registry['observation_ref']==binding['refs']['inference_ref'], 'Registry observation binding')
        self.panel = small_json(binding['panel_ref'])
        require([row['sample_index'] for row in self.panel]==list(range(len(self.panel))), 'Native panel row order')
        self.native_samples = np.asarray([row['observed_node_id'] for row in self.panel],dtype=np.int64)
        self.sample_id = np.asarray([f"{parent_id}:{row['sample_id']}:{row['haplotype']}" for row in self.panel])
        require(len(np.unique(self.native_samples))==len(self.panel) and len(set(self.sample_id))==len(self.panel),
                'Unique native nodes and namespaced haplotype identities')
        self.offset = float(binding['coordinate_offset_bp'])
        self.metadata = np.zeros((len(self.panel),17),dtype=np.float32)
        self.reference = np.asarray([-1 if row['observed_reference_class'] is None else row['observed_reference_class']
                                     for row in self.panel],dtype=np.int64)
        self.cohort = np.asarray([COHORTS.index(row['observed_cohort']) for row in self.panel],dtype=np.int64)
        self.age = np.asarray([row['age_generations'] for row in self.panel],dtype=np.float64)
        self.query = np.asarray([row['role']=='query' for row in self.panel],dtype=bool)
        require((self.reference[self.query]==-1).all() and np.isin(self.reference[~self.query],np.arange(4)).all(),
                'Query ancestry unknown; observed reference classes only')
        self.metadata[np.arange(len(self.panel)),self.cohort]=1
        self.metadata[np.arange(len(self.panel)),9+np.where(self.reference<0,4,self.reference)]=1
        self.metadata[:,14]=np.log1p(self.age)/np.log(10001.)
        self.metadata[:,15]=self.query
        self.metadata[:,16]=~self.query
        self._pid=None;self._reader=None;self._vcf=None
        self._cache=OrderedDict();self._cache_bytes=0
        self.counters=dict(tree_requests=0,tree_cache_hits=0,trees_read=0,VCF_sites_read=0)

    def __getstate__(self):
        state=self.__dict__.copy()
        state.update(_pid=None,_reader=None,_vcf=None,_cache=OrderedDict(),_cache_bytes=0)
        return state

    def _handles(self):
        if self._pid!=os.getpid():
            if self._vcf is not None:self._vcf.close()
            self._pid=os.getpid()
            self._reader=ReplacementTreeReader(path_for(self.binding['tree_ref']),self.native_samples)
            self._vcf=SortedVCF(path_for(self.binding['refs']['observed_vcf_ref']),self.panel)
            self._cache=OrderedDict();self._cache_bytes=0
        return self._reader,self._vcf

    def _tree(self, position):
        reader,vcf=self._handles()
        self.counters['tree_requests']+=1
        for key in tuple(self._cache):
            if key[0]<=position<key[1]:
                self.counters['tree_cache_hits']+=1
                self._cache.move_to_end(key)
                return self._cache[key][0]
        tree=reader.at(float(position))
        self.counters['trees_read']+=1
        leaves=np.searchsorted(tree.original_ids,self.native_samples)
        require(np.array_equal(tree.original_ids[leaves],self.native_samples), 'Actual native leaf identities')
        require(np.array_equal(tree.sample_index[leaves],np.arange(len(self.panel))), 'Native leaf metadata row join')
        require(np.array_equal(tree.time[leaves],self.age), 'Native sample ages match observed metadata')
        positions=reader.columns['sites/position']
        begin,end=np.searchsorted(positions,[tree.left,tree.right],side='left')
        site_ids=np.arange(begin,end,dtype=np.int64)
        genotype=[];alleles=[]
        for site_id in site_ids:
            calls,names=vcf.at(exact_vcf_position(positions[site_id]+self.offset))
            genotype.append(calls);alleles.append(len(names));self.counters['VCF_sites_read']+=1
        payload=dict(tree=tree,leaves=leaves,site_ids=site_ids,site_positions=np.array(positions[begin:end]),
                     genotype=np.asarray(genotype,dtype=np.int8).reshape(-1,len(self.panel)),
                     alleles=np.asarray(alleles,dtype=np.int8))
        size=sum(value.nbytes for value in payload.values() if isinstance(value,np.ndarray))
        size+=sum(value.nbytes for value in (tree.original_ids,tree.time,tree.parent,tree.sample_index))
        while self._cache and self._cache_bytes+size>self.cache_limit:
            _,(_,old_size)=self._cache.popitem(last=False);self._cache_bytes-=old_size
        if size<=self.cache_limit:
            self._cache[(tree.left,tree.right)]=(payload,size);self._cache_bytes+=size
        return payload

    def extract(self, site):
        require(site['parent_id']==self.parent_id,'Window belongs to this immutable parent')
        require(len(site['example_indices'])==len(site['sample_indices'])>0,'Complete nonempty target row mapping')
        rows=[self.records[int(index)] for index in site['example_indices']]
        for row,sample in zip(rows,site['sample_indices']):
            observed=self.panel[sample]
            require(row['sample_index']==sample and row['sample_id']==observed['sample_id'] and
                    row['haplotype']==observed['haplotype'] and row['query_node_id']==self.native_samples[sample]
                    and observed['role']=='query' and row['cohort']==observed['observed_cohort'], 'Exact focal identity join')
            require(row['native_site_id']==site['native_site_id'] and row['position_local_bp']==site['position_local_bp']
                    and row['position_original_bp']==site['position_original_bp'], 'Exact focal coordinate join')
        reader,vcf=self._handles()
        position=float(site['position_local_bp']);sid=int(site['native_site_id'])
        require(reader.columns['sites/position'][sid]==position and position+self.offset==site['position_original_bp'],
                'Native site and original VCF offset agreement')
        context=site['context'];coordinates=np.asarray(context['coordinates'],dtype=np.float64)
        require(len(coordinates)==5 and coordinates[0]==position and context['weights']==[0.,.25,.25,.25,.25],
                'Registered five-coordinate source measure')
        selected=[self._tree(x) for x in coordinates]
        unique={(item['tree'].left,item['tree'].right):item for item in selected}
        unique=[unique[key] for key in sorted(unique)]
        native_index={(item['tree'].left,item['tree'].right):i for i,item in enumerate(unique)}
        context_tree=np.asarray([native_index[(item['tree'].left,item['tree'].right)] for item in selected],dtype=np.int64)
        parts={key:[] for key in ('parent','node_time','node_tree_index','leaf_node_index','site_tree_index',
                                  'site_position_local_bp','site_genotypes','site_allele_count')}
        node_offset=0
        for tree_index,item in enumerate(unique):
            tree=item['tree']
            parts['parent'].append(np.where(tree.parent>=0,tree.parent+node_offset,-1))
            parts['node_time'].append(tree.time)
            parts['node_tree_index'].append(np.full(len(tree.parent),tree_index,dtype=np.int64))
            parts['leaf_node_index'].append((item['leaves']+node_offset)[None,:])
            parts['site_tree_index'].append(np.full(len(item['site_ids']),tree_index,dtype=np.int64))
            parts['site_position_local_bp'].append(item['site_positions'])
            parts['site_genotypes'].append(item['genotype']);parts['site_allele_count'].append(item['alleles'])
            node_offset+=len(tree.parent)
        packet={key:np.concatenate(value,axis=0) for key,value in parts.items()}
        focal=selected[0];focal_row=int(np.searchsorted(focal['site_ids'],sid))
        require(focal_row<len(focal['site_ids']) and focal['site_ids'][focal_row]==sid,'Focal SNP retained in its complete native tree')
        packet.update(leaf_features=self.metadata.copy(),
            tree_interval_local_bp=np.asarray([[item['tree'].left,item['tree'].right] for item in unique]),
            tree_window_index=np.zeros(len(unique),dtype=np.int64),tree_core_mask=np.arange(len(unique))==context_tree[0],
            focal_tree_index=context_tree[:1],focal_position_local_bp=np.asarray([position]),
            focal_site_index=np.zeros(len(rows),dtype=np.int64),focal_sample_index=np.asarray(site['sample_indices'],dtype=np.int64),
            focal_genotypes=focal['genotype'][focal_row:focal_row+1],focal_allele_count=focal['alleles'][focal_row:focal_row+1],
            reference_class=self.reference.copy(),sample_age=self.age.copy(),observed_cohort=self.cohort.copy(),query_mask=self.query.copy(),
            sample_id=self.sample_id.copy(),tree_sample_id=np.broadcast_to(self.sample_id,(len(unique),len(self.panel))).copy(),
            context_coordinate_local_bp=coordinates,context_tree_index=context_tree,context_window_index=np.zeros(5,dtype=np.int64),
            context_quadrature_weight=np.asarray(context['weights']),context_proposal_density_per_bp=np.asarray(context['density']),
            context_forced_focal=np.asarray(context['forced'],dtype=bool),
            window_context_domain_local_bp=np.asarray([context['domain']]),window_context_kind=np.asarray([context['kind']],dtype=np.int8))
        require(set(packet)==set(MODEL_KEYS), 'Exact observation-only model contract')
        site_positions=packet['site_position_local_bp']
        metadata=dict(parent_id=self.parent_id,native_site_id=sid,canonical_examples=len(rows),
            unique_native_trees=len(unique),coordinate_occurrences=len(coordinates),nodes=len(packet['parent']),
            site_occurrences=len(site_positions),native_tree_extent_local_bp=[float(packet['tree_interval_local_bp'][:,0].min()),
                                                                           float(packet['tree_interval_local_bp'][:,1].max())],
            observed_SNP_extent_local_bp=None if not len(site_positions) else [float(site_positions.min()),float(site_positions.max())],
            coordinate_domain_local_bp=context['domain'],source_offset_bp=self.offset,
            VCF_logical_bytes_read=vcf.bytes_read,reader_counters=self.counters.copy())
        return packet,metadata


def load_observations(path):
    with np.load(Path(path),allow_pickle=False) as archive:
        require(set(archive.files)==set(MODEL_KEYS),'Serialized packet contains only declared observations')
        return {key:archive[key].copy() for key in MODEL_KEYS}
