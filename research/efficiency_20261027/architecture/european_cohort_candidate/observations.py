"""Explicit all-panel public cohort binding around unchanged AP observations.

Only TRAIN or independently registered validation inputs are accepted. Cohort
labels never alter pure ancestry donor labels, focal selection or target joins.
"""
from dataclasses import dataclass
import torch
from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters import european_observations as original
from .model import COHORTS, CONDITION, need

PROTOCOL = 'ancestralpaths_paper200mb_native6'


def cohort_tensor(rows, *, reference_class, sample_age, query_columns, observed_nodes):
    need(len(rows)==952 and reference_class.shape==sample_age.shape==(952,), 'All952 original observed samples')
    need(query_columns.shape==(686,) and len(observed_nodes)==952, 'All686 query identities and every sampled node')
    result=torch.zeros((952,8),dtype=torch.float32)
    pure={'Ana':0,'CHG':1,'WHG':2,'EHG':3}
    query=[]
    for index,row in enumerate(rows):
        need(row['sample_index']==index and row['observed_node_id']==int(observed_nodes[index]),
             'Exact original sample-column and observed-node identity')
        cohort=row['observed_cohort']
        need(cohort in (*COHORTS,'present_day'), 'Only declared public sampling cohorts')
        if cohort!='present_day':
            result[index,COHORTS.index(cohort)]=1.
        label=row['observed_reference_class']
        if cohort in pure:
            need(row['role']=='pure_source_reference' and label==pure[cohort], 'Pure references remain original four classes')
        else:
            need(row['role']=='query' and label is None, 'Admixed cohorts remain unknown ancestry donors')
            query.append(index)
        need(int(reference_class[index])==(-1 if label is None else label)
             and float(sample_age[index])==row['age_generations'], 'Original age/reference inputs unchanged')
        need((row['age_generations']==0) == (cohort=='present_day'), 'Present-day distinction is explicit in metadata')
    need(query==query_columns.tolist(), 'Every original query and its original order retained')
    return result


@dataclass(frozen=True)
class CohortParent:
    original: object
    observed_cohort: torch.Tensor
    cohort_stamps: tuple
    sample_manifest_ref: dict
    scientific_condition: str = CONDITION

    @property
    def bank(self): return self.original.bank
    @property
    def episodes(self): return self.original.episodes
    @property
    def cut(self): return self.original.cut
    @property
    def parent_id(self): return self.original.parent_id
    @property
    def split(self): return self.original.split
    @property
    def native_holds(self): return self.original.native_holds
    @property
    def focal_hold(self): return self.original.focal_hold
    @property
    def focal_sample_column(self): return self.original.focal_sample_column
    @property
    def sampled_registry_ref(self): return self.original.sampled_registry_ref
    @property
    def focal_identity_sha256(self): return self.original.focal_identity_sha256

    def validate(self):
        self.original.validate()
        need(type(self.original) is original.EuropeanWholeParent and self.scientific_condition==CONDITION,
             'Exact original observed support and distinct successor condition')
        need(self.split in ('train','val_prediction'), 'No TEST development bundle')
        need(self.observed_cohort.shape==(952,8) and self.observed_cohort.dtype==torch.float32
             and self.observed_cohort.device.type=='cpu' and not self.observed_cohort.requires_grad
             and self.cohort_stamps==original.implementation.memory_core._stamps((self.observed_cohort,)),
             'Immutable observed cohort inputs for all952 original sample columns')
        need(self.sample_manifest_ref==self.original.original.source_refs['sample_manifest_ref'],
             'Cohort evidence binds the exact original public observed manifest')
        need(bool(((self.observed_cohort==0)|(self.observed_cohort==1)).all())
             and bool((self.observed_cohort.sum(-1)<=1).all()), 'One ancient cohort or known present-day zero vector')
        return self.bank


def load_relate_observation(inference_ref, observed_vcf_ref, *, max_sites, max_metadata_bytes, resource_check):
    receipt=original._json_ref(inference_ref,max_metadata_bytes)
    need(receipt['split'] in ('train','validation'), 'TRAIN or independently registered VAL inference only; no TEST')
    return original.load_relate_observation(inference_ref,observed_vcf_ref,protocol_id=PROTOCOL,
        max_sites=max_sites,max_metadata_bytes=max_metadata_bytes,resource_check=resource_check)


def prepare_observation(state, *, parent_id, split, source_refs, sampled_registry_ref,
                        max_nodes, max_holds, max_episode_payload_bytes, resource_check,
                        max_sampled_metadata_bytes=128*2**20):
    need(split in ('train','validation') and state['manifest']['split']==split, 'Original TRAIN or independent VAL split')
    feature=cohort_tensor(state['rows'],reference_class=torch.as_tensor(state['labels']),
        sample_age=torch.as_tensor(state['ages']),query_columns=torch.as_tensor(state['query_columns']),
        observed_nodes=state['nodes'])
    bundle,metadata=original.prepare_observation(state,protocol_id=PROTOCOL,parent_id=parent_id,split=split,
        source_refs=source_refs,sampled_registry_ref=sampled_registry_ref,max_nodes=max_nodes,max_holds=max_holds,
        max_episode_payload_bytes=max_episode_payload_bytes,resource_check=resource_check,
        max_sampled_metadata_bytes=max_sampled_metadata_bytes)
    result=CohortParent(bundle,feature,original.implementation.memory_core._stamps((feature,)),
                        dict(source_refs['sample_manifest_ref']))
    result.validate()
    metadata=dict(metadata,scientific_condition=CONDITION,observed_cohort_columns=list(COHORTS),
        observed_cohort_manifest_ref=result.sample_manifest_ref,
        observed_cohort_inputs_all952=True,cohort_is_sampling_metadata_not_ancestry_truth=True,
        original_reference_labels_and_donor_frequencies_unchanged=True)
    return result,metadata
