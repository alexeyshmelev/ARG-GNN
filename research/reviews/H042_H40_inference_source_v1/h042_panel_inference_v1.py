"""Deferred-runtime H042 frozen-state inference for an explicitly qualified panel.

No loader, file checkpoint reader, training, padding, calibration, admission or
launch is implemented. Module import is standard-library-only. Root must qualify
original-panel parity and a separate identity-preserving real observation loader.
"""
from pathlib import Path
from collections.abc import Mapping
from dataclasses import fields
import hashlib,json,math
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OWN=ROOT/'research/reviews/H042_H40_inference_source_v1'
SELECTION='earliest_maximum_equal_parent_Present182_exact_span_accuracy_then_lower_NLL_then_earlier_pass'
FEATURE_SEMANTICS='frozen_H042_site19_leaf6_edge4_support5_C4_alpha0.5_age350_logspan2000001_v1'


def _require(ok,message):
    if not ok:raise ValueError(message)


def _verify_frozen_sources():
    manifest=json.loads((OWN/'frozen_sources.json').read_text())
    for key,record in manifest['sources'].items():
        p=ROOT/record['path']
        _require(p.resolve()==p and hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256'],
            'Frozen H042/feature source changed: '+key)


def _counts(h,q):
    _require(type(h) is int and type(q) is int and h>=3 and 0<q<h,
        'Explicit actual panel/query counts required; no padded or invented rows')


def _ref(record):
    _require(type(record) is dict and set(record)=={'path','sha256'}
        and type(record['path']) is str and record['path'] and not Path(record['path']).is_absolute()
        and '..' not in Path(record['path']).parts and type(record['sha256']) is str
        and len(record['sha256'])==64 and all(c in '0123456789abcdef' for c in record['sha256']),
        'Explicit root-authenticated exact evidence reference required')


def _state_schema():return json.loads((OWN/'state_schema.json').read_text())


def _tensor_sha256(tensor):
    # Root calls this only on already authenticated in-memory CPU tensors.
    header=json.dumps(dict(dtype=str(tensor.dtype),shape=list(tensor.shape)),sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(header+b'\0'+tensor.detach().contiguous().numpy().tobytes(order='C')).hexdigest()


def _model_class_v1():
    import torch
    from arggnn.anonymous_clade_flag_control_v1 import (
        AnonymousCladeFlagControlGNNV1,RankEightCladeFlagAdapterV1)
    from arggnn.mutation_routed_full_panel_pyg_model_v1 import MutationRoutedFullPanelPyGGNNV1
    from arggnn.mutation_routed_whole_tree_gnn_v1 import _FLOAT_FIELDS,_LEVEL_FIELDS

    class PanelH042FrozenInferenceV1(AnonymousCladeFlagControlGNNV1):
        def __init__(self,expected_haplotypes,expected_queries):
            _counts(expected_haplotypes,expected_queries)
            # Exactly the original H042 module initialization order and layers.
            MutationRoutedFullPanelPyGGNNV1.__init__(self,hidden_dim=32,
                expected_haplotypes=expected_haplotypes,site_chunk_size=256,checkpoint_sites=True)
            self.clade_coupling=RankEightCladeFlagAdapterV1()
            self.expected_queries=expected_queries  # plain attribute, no state tensor

        def forward_holds(self, bank, *, flags, control="aligned"):
            parameter = next(self.parameters())
            t, h = bank.sample_index.shape
            if (parameter.dtype != torch.float32 or h != self.expected_haplotypes
                    or bank.query_column.numel() != self.expected_queries or not t
                    or bank.parent.device != parameter.device
                    or flags.native_holds != t or flags.haplotypes != h):
                raise ValueError("Complete original FP32 native bank and matching flags required")
            for name in _FLOAT_FIELDS:
                value = getattr(bank, name)
                if value.dtype != torch.float32 or value.device != parameter.device:
                    raise ValueError("Original floating observations/device differ")
            for f in fields(bank):
                value = getattr(bank, f.name)
                values = value if f.name in _LEVEL_FIELDS else (value,)
                if any(v.device != parameter.device for v in values):
                    raise ValueError("Complete observation bank device differs")
            context = self.encode_trees(bank, flags=flags, control=control)
            span = bank.spans.log1p() / math.log(2000001.)
            prev = torch.cat((
                bank.spans.new_zeros(1), .5 * (bank.spans[1:] + bank.spans[:-1]),
            ))
            nxt = torch.cat((
                .5 * (bank.spans[1:] + bank.spans[:-1]), bank.spans.new_zeros(1),
            ))
            geometry = torch.stack((
                span, prev.log1p() / math.log(2000001.),
                nxt.log1p() / math.log(2000001.),
            ), -1)
            sequence = torch.cat((context, geometry[:, None].expand(t, h, 3)), -1)
            history, _ = self.genomic_gru(sequence)
            return self.classifier(torch.cat((context, history), -1))

    return PanelH042FrozenInferenceV1


def make_frozen_h042_panel_v1(selected_model_state,*,selected_state_identity,
        expected_haplotypes,expected_queries):
    """Load the entire genuinely selected H042 state, never H038 warmstart.

    Root's authenticated checkpoint/selection reader must supply the CPU state
    and tensor digests. Evidence refs are checked structurally, never opened here.
    This source does not establish which checkpoint won the completed selection.
    """
    _verify_frozen_sources();_counts(expected_haplotypes,expected_queries)
    identity=selected_state_identity
    _require(type(identity) is dict and identity.get('schema')=='H042_completed_selected_best_state_identity_v1'
        and identity.get('root_authorizer')=='root_agent' and identity.get('qualified') is True
        and identity.get('hypothesis_id')=='H042' and identity.get('arm')=='exact_clade_flags'
        and identity.get('seed')==101 and identity.get('selection')==SELECTION
        and identity.get('selected_pass') in (5,10,15,20,25,30)
        and identity.get('completed_original_H042_selection_verified') is True
        and identity.get('retrained_or_calibrated') is False,
        'Genuine completed best H042 state identity from separate root reader required')
    _ref(identity['selected_checkpoint_ref']);_ref(identity['completed_selection_ref'])
    import torch
    schema=_state_schema();shapes=schema['shapes'];digests=identity['state_tensor_sha256']
    _require(isinstance(selected_model_state,Mapping) and set(selected_model_state)==set(shapes)==set(digests),
        'Exact42 complete learned H042 state keys required; no missing/new layers')
    for name,shape in shapes.items():
        value=selected_model_state[name]
        _require(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.float32
            and value.layout==torch.strided and list(value.shape)==shape and not value.requires_grad
            and bool(torch.isfinite(value).all()) and _tensor_sha256(value)==digests[name],
            'Exact authenticated original CPU H042 tensor shape/dtype/values required: '+name)
    _require(torch.get_default_dtype()==torch.float32,'Exact FP32 constructor default required')
    model_type=_model_class_v1()
    with torch.random.fork_rng(devices=[]),torch.device('cpu'):
        model=model_type(expected_haplotypes,expected_queries)
    actual=model.state_dict()
    _require(set(actual)==set(shapes) and len(actual)==42
        and sum(p.numel() for p in model.parameters())==30788,
        'Original H042 state graph and30,788 parameters required')
    for name,value in actual.items():
        _require(value.dtype==torch.float32 and value.device.type=='cpu' and list(value.shape)==shapes[name],
            'Original registered tensor shape/dtype changed: '+name)
    model.load_state_dict(selected_model_state,strict=True)
    _require(all(model.state_dict()[name].equal(value) and _tensor_sha256(model.state_dict()[name])==digests[name]
        for name,value in selected_model_state.items()),'Strict H042 loaded values differ')
    model.eval();model.requires_grad_(False)
    return model


def make_frozen_h042_h40_v1(selected_model_state,*,selected_state_identity):
    return make_frozen_h042_panel_v1(selected_model_state,selected_state_identity=selected_state_identity,
        expected_haplotypes=40,expected_queries=24)


def bank_fingerprint_sha256_v1(bank):
    """Root's future loader binds the exact CPU bank; no file or target reader."""
    _verify_frozen_sources()
    import torch
    from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
    _require(type(bank) is MutationRoutedWholeTreeInputsV1,'Exact original observation-only bank dataclass required')
    record={}
    for field in fields(bank):
        value=getattr(bank,field.name);values=value if isinstance(value,tuple) else (value,)
        _require(all(isinstance(v,torch.Tensor) and v.device.type=='cpu' and not v.requires_grad for v in values),
            'Unmodified read-only CPU observations required')
        record[field.name]=[_tensor_sha256(v) for v in values]
    return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_panel_bank_v1(bank,*,qualified_lineage,expected_haplotypes,expected_queries):
    """Shape/content gate after a separate loader has established real identity.

    A caller's booleans do not establish provenance. Root must retain the actual
    source/VCF/native sample/age/site/support joins named in this receipt.
    """
    _verify_frozen_sources();_counts(expected_haplotypes,expected_queries)
    proof=qualified_lineage
    required_true=('source_age_haplotype_join_qualified','native_sample_join_qualified',
        'all_original_native_holds_unaries_preserved','original_nucleotide_calls_and_missingness_preserved',
        'CPU_float64_chronology_qualified','physical_support_and_readout_identity_qualified',
        'frozen_feature_arithmetic_qualified')
    _require(type(proof) is dict and proof.get('schema')=='H042_panel_observation_identity_v1'
        and proof.get('root_authorizer')=='root_agent' and proof.get('qualified') is True
        and proof.get('haplotypes')==expected_haplotypes and proof.get('queries')==expected_queries
        and proof.get('feature_semantics')==FEATURE_SEMANTICS
        and proof.get('age_split_generations')==350. and proof.get('donor_alpha')==.5
        and proof.get('learned_span_log_denominator')==2000001.
        and proof.get('truth_or_AP_read') is False and proof.get('padding_duplication_or_fixture_bypass') is False
        and proof.get('retraining_or_calibration') is False and all(proof.get(k) is True for k in required_true),
        'Separate qualified original-observation loader/identity contract required')
    _ref(proof['observation_bridge_completion_ref']);_ref(proof['identity_contract_ref'])
    keys=proof['stable_haplotype_keys'];queries=proof['query_columns'];classes=proof['reference_class'];ages=proof['sample_age_generations']
    _require(type(keys) is list and len(keys)==expected_haplotypes
        and all(type(k) is list and len(k)==2 and type(k[0]) is str and k[0] and type(k[1]) is int and k[1] in (0,1) for k in keys)
        and len({tuple(k) for k in keys})==expected_haplotypes,
        'Unique genuine ordered subject/haplotype keys required')
    _require(type(queries) is list and len(queries)==expected_queries and len(set(queries))==expected_queries
        and all(type(q) is int and 0<=q<expected_haplotypes for q in queries)
        and type(classes) is list and len(classes)==expected_haplotypes
        and all(type(c) is int and -1<=c<=3 for c in classes)
        and type(ages) is list and len(ages)==expected_haplotypes
        and all(type(a) in (float,int) and math.isfinite(a) and a>=0 for a in ages),
        'Exact original query/proxy/date vectors required')
    import torch
    from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1,validate_inputs_v1
    _require(type(bank) is MutationRoutedWholeTreeInputsV1,'Exact original observation-only bank required')
    result=validate_inputs_v1(bank,expected_haplotypes=expected_haplotypes)
    _require(torch.equal(bank.query_column,torch.tensor(queries,dtype=torch.long))
        and torch.equal(bank.reference_class,torch.tensor(classes,dtype=torch.long))
        and torch.equal(bank.sample_age,torch.tensor(ages,dtype=torch.float32))
        and bank_fingerprint_sha256_v1(bank)==proof.get('bank_fingerprint_sha256'),
        'Actual CPU bank and root-qualified ordered identity/feature payload differ')
    return result


def validate_h018_h40_bank_v1(bank,*,qualified_lineage):
    result=validate_panel_bank_v1(bank,qualified_lineage=qualified_lineage,expected_haplotypes=40,expected_queries=24)
    proof=qualified_lineage
    _require(proof.get('condition_id')=='H018_R16_20_30Mb_extension_v1'
        and proof.get('proxy_pure_ancestry_status')=='unverified'
        and proof['query_columns']==list(range(16,40))
        and proof['reference_class']==[c for c in range(4) for _ in range(4)]+[-1]*24,
        'Exact real R16/Q24 order and unverified proxy-purity status required')
    ages=proof['sample_age_generations'];classes=proof['reference_class']
    keys=proof['stable_haplotype_keys']
    _require(len({keys[i][0] for i in range(0,40,2)})==20
        and all(keys[i][0]==keys[i+1][0] and keys[i][1]==0 and keys[i+1][1]==1
            and ages[i]==ages[i+1] for i in range(0,40,2)),
        'Twenty genuine ordered diploids with both original haplotypes and one unchanged source date required')
    groups=[sum(classes[i]==c and (ages[i]<350. if cohort==0 else ages[i]>=350.)
        for i in range(40)) for cohort in range(2) for c in range(4)]
    _require(groups==[2]*8 and proof.get('donor_group_counts')==groups,
        'All eight original age350 donor channels require their two actual H018 reference haplotypes')
    return dict(result,donor_group_counts=groups,proxy_pure_ancestry_status='unverified',real_accuracy_qualified=False)


def build_clade_flags_panel_v1(bank,*,qualified_lineage,expected_haplotypes,expected_queries):
    """Use unchanged exact-clade arrays, with an explicit new panel contract."""
    validate_panel_bank_v1(bank,qualified_lineage=qualified_lineage,
        expected_haplotypes=expected_haplotypes,expected_queries=expected_queries)
    from arggnn.anonymous_clade_flag_control_v1 import (
        _cpu_index_array,build_clade_flag_arrays_v1,AnonymousCladeFlagsV1)
    parent=_cpu_index_array(bank.parent);samples=_cpu_index_array(bank.sample_index)
    arrays,meta=build_clade_flag_arrays_v1(parent,samples)
    return AnonymousCladeFlagsV1.from_verified_arrays(arrays,meta),meta


def build_h018_h40_clade_flags_v1(bank,*,qualified_lineage):
    validate_h018_h40_bank_v1(bank,qualified_lineage=qualified_lineage)
    return build_clade_flags_panel_v1(bank,qualified_lineage=qualified_lineage,
        expected_haplotypes=40,expected_queries=24)
