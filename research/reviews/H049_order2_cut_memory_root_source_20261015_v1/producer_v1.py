"""H049 three_arm marker-grid TRAIN/EVAL producer. Inert without root context.

Reuses H046's exact complete-parent FP32 optimizer step, H044's original target
and preparation readers, and the original full-panel scorer. Root owns current
admission/guards/launcher/device and authenticates actual scientific payloads.
"""
from __future__ import annotations
from collections import OrderedDict
import argparse,hashlib,importlib.util,io,json,random,statistics,sys,time
from pathlib import Path
from research.reviews.H049_order2_cut_memory_root_source_20261015_v1 import cut_recipe_codec_v1 as codec
from research.reviews.order2_cut_memory_implementation_20261015_v1 import integration_hooks_v1 as hooks
from arggnn import train_continuous_clade_flank_v1 as h46
from arggnn import train_marker_resolution_readout_v1 as h44
from arggnn import train_anonymous_clade_flag_control_v1 as h42

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/H049_order2_cut_memory_root_source_20261015_v1/producer_v1.py"
ARMS=hooks.ARMS
TRAIN,VAL,PASSES=h42.TRAIN,h42.VAL,h42.PASSES
MODEL,SELECTION,METRICS=h42.MODEL,h42.SELECTION,h42.METRICS
READOUT=dict(hooks.READOUT)
WARM_CHECKPOINT,WARM_COMPLETION=h44.WARM_CHECKPOINT,h44.WARM_COMPLETION
require,bound=h42.require,h42.bound
REGISTRATION_REF=None
DESIGN_REF={'path': 'research/reviews/ordered_graph_operator_research_20261015_v1/operator_spec_v1.json', 'sha256': '73309b3e092d69d156f73e5ba29de286f7f2f865aaae3e115bb44b71c9a17bad'}
CORE_REF={'path': 'research/reviews/order2_cut_memory_implementation_20261015_v1/order2_cut_memory_v1.py', 'sha256': '3b496b163e11147643aa576dfe1f37a75544d466624ef2bd4377fc62d1533265'}
SCIENTIFIC_REFS=({'path': 'src/arggnn/H040_latent_clade_execution_v1.py', 'sha256': '28973fe0a9dae2c950c07dd463a23edcb6b45ab6659813f5ef31b9dc766bf84d'}, {'path': 'src/arggnn/age_interface.py', 'sha256': '15f9615a84a230386d8f71f5a317a46051c395f75408c79a6213f1b07f101f3d'}, {'path': 'src/arggnn/anchor_kernel.py', 'sha256': '2b8cc1c1726f746f961c0d63a4870e806a1a9d3d6e1fa6724dc0bda39967b083'}, {'path': 'src/arggnn/anchor_spectral_compact_v1.py', 'sha256': '35cd11e92499a997dd9623f937797f2c482886698e2a79b48d0a3eb8aa87bf9c'}, {'path': 'src/arggnn/anchor_temporal_baselines.py', 'sha256': '3e88095eb466ea25db5f69553dfa31e6e002fa55ce4d31c4ddb17cb73115518b'}, {'path': 'src/arggnn/anonymous_clade_flag_control_v1.py', 'sha256': 'd88a4639a53e63e2bff91e20d98e2af1f5cf8abc71e5fde2e02ac35e831eb8ce'}, {'path': 'src/arggnn/anonymous_clade_gnn_v1.py', 'sha256': '2f878e6334079cff083f67b6da8eda129dd30e7beebc62ae2a93d51796c3f137'}, {'path': 'src/arggnn/anonymous_clade_witness_v1.py', 'sha256': '0cdb438132e11c86fb61306a02244af0471f468c2ff2a57324644cafb97b6310'}, {'path': 'src/arggnn/anonymous_exact_clade_transport_v1.py', 'sha256': '441751a252f111fdf0d8af4e70e93e5a923b7bad7829f3aed7d5060d65727304'}, {'path': 'src/arggnn/biological_candidate.py', 'sha256': '6b3a25d0e59026cab7189032eb41221feb7d1bdf69e15f6c737a672dfb711dd3'}, {'path': 'src/arggnn/biological_gst_control_v1.py', 'sha256': '66788b41cabcb852c739505200a641bdc93fea3c9fbb642c291d52cefc77350f'}, {'path': 'src/arggnn/boundary_memory.py', 'sha256': 'a056cd10d1c848d7dabdc1daf646548876fd90e8f489667c46a374ac28cd11dc'}, {'path': 'src/arggnn/census_biological_bank_v1.py', 'sha256': '1e89e93a23827e15439eceaabea505c6fae088fc407b62e1fd06c71dd26460c6'}, {'path': 'src/arggnn/census_cut_exchange_v1.py', 'sha256': '2819e7cef7ded38968e019ec015f0b2d1d264502a0762bfb3cf795e90c183b62'}, {'path': 'src/arggnn/census_native_qualification_runner_v1.py', 'sha256': 'a63c9481c934247fabe077aed8dabe02bb1cc831f7bbfe441f5974b397346140'}, {'path': 'src/arggnn/census_tree_encoder_v1.py', 'sha256': '09351221d2d91f72c8b1df12bc3799ed72e538597c31a55737ed1bec111f886e'}, {'path': 'src/arggnn/census_tsinfer_fit_v1.py', 'sha256': 'fd55aaa6b8f88983514ff6888ad2efbadca4ee8b2e3acabd09fbd4a361b5acf3'}, {'path': 'src/arggnn/census_tsinfer_observation_v1.py', 'sha256': 'c32a0ed65500d6ea4a3f9ea9373463dbc7bda88eedace84a2d6538a3b3c8980a'}, {'path': 'src/arggnn/continuous_clade_flank_gnn_v1.py', 'sha256': '2c81b74c59a532b46ed75f7fd656ea526ec06fd9bf62af35855ab291f5451e20'}, {'path': 'src/arggnn/data.py', 'sha256': 'fb4e4b3ad8744731e6d2c5966cc2679f9f2d59dfd079facc280d69fbc61ef654'}, {'path': 'src/arggnn/development_execution.py', 'sha256': '4a2f87320615eb655a7700b6bbff4a7d293b1b16749468afa46f0de2d12e276f'}, {'path': 'src/arggnn/directed_clade_exchange_crf_v1.py', 'sha256': '81d2742c3d2d435d237a62f572babefd80b865a78eac1b1a7594d88425a8b3d0'}, {'path': 'src/arggnn/directed_clade_exchange_witness_v1.py', 'sha256': 'd8da7e9fbbca9c9f20b4ad49c26e66897e33fddc575cf8703b84fd14acc859c4'}, {'path': 'src/arggnn/full_panel_native_baseline_contracts_v1.py', 'sha256': 'b7ea16a43c93dca04663ebd99261e98b445cb08190e738d627ac8e0c948fe477'}, {'path': 'src/arggnn/genealogy_baselines.py', 'sha256': '388e84c183413a79db8e0e75c86f1ce202b4bb9999473b729be045d7459dc9f8'}, {'path': 'src/arggnn/gst_baseline.py', 'sha256': '68d7283b99dbabd6c9a3254f8a9bb71272b1288ccea25a6821610804c354cc73'}, {'path': 'src/arggnn/kron_interface.py', 'sha256': '09ec6286c185e7aa2b13d76dcbe8fea24b3a1c785a6619ac9bc5464e0de66d69'}, {'path': 'src/arggnn/marker_grid_reset_chain_gnn_v1.py', 'sha256': '341cbf32f10dfa3b0c941c9a70244758d3cc45d6362f33c6e74da5fd8d26acb4'}, {'path': 'src/arggnn/marker_resolution_memory_bounds_v2.py', 'sha256': 'f821c69037adf97a05bf6537c427ba41cebcff6f4256d24aa848d2315669e7ce'}, {'path': 'src/arggnn/marker_resolution_readout_v1.py', 'sha256': '27ecd30683024def2d408bf16250102b3252d0ae112687b9664fca086b141310'}, {'path': 'src/arggnn/models.py', 'sha256': '8429bd02a40950a3dc4d76046ab4b31b366871499d95e295a16ade730f567df5'}, {'path': 'src/arggnn/mutation_routed_fast_primary_v1.py', 'sha256': 'f4035cae8be87f3e7af7c6024ca997fdac7f88560e84e73efd4bfcad9295725f'}, {'path': 'src/arggnn/mutation_routed_full_panel_cache_v1.py', 'sha256': '1c1f1e5222df0e09e83b470d5375b8946cb3a89db1bc55e9d2c293a09037cb38'}, {'path': 'src/arggnn/mutation_routed_full_panel_fast_reader_v1.py', 'sha256': 'd6b963dc14e1f6752e310e627faf2e4f9a2e408fcb7dab17e3439771ab227e03'}, {'path': 'src/arggnn/mutation_routed_full_panel_observations_v1.py', 'sha256': 'a7184be9a735d95b80975f68666d51e9ed0316cc535d9f7b54baafea532a1107'}, {'path': 'src/arggnn/mutation_routed_full_panel_primary_v1.py', 'sha256': 'a94fd5af1cd10b1605f75621c36a130faa3f9d5ad71068f8b187d8e73c7e5374'}, {'path': 'src/arggnn/mutation_routed_full_panel_pyg_model_v1.py', 'sha256': '786864bbf47dadbd9668abf96119ac7a0d227d91b10a7831374b18a6e63cdefa'}, {'path': 'src/arggnn/mutation_routed_observation_cache_v1.py', 'sha256': 'f4adf6832b6d80bdf3233083e0aacc87ae27818314ca185021a9f4c8b0e97150'}, {'path': 'src/arggnn/mutation_routed_parent_loader_v1.py', 'sha256': 'bd9a55d1a7b40e5e643433238f84a32c8390ac0fe4a653cb12144a7137aaee48'}, {'path': 'src/arggnn/mutation_routed_pyg_dataset_v1.py', 'sha256': 'f7488ae9794c301d47ad66bab9caed5ff87ef09afcf96c5e48c9c00d350706ca'}, {'path': 'src/arggnn/mutation_routed_pyg_gnn_v1.py', 'sha256': '53866aa4616d12e7d6cbdd34160ee7ab8fb00999444c348ee4d251b8fdfd4337'}, {'path': 'src/arggnn/mutation_routed_tree_observations_v1.py', 'sha256': '922df286e595370ade23595420a325014e4df64aac88060e8a1a846684eab068'}, {'path': 'src/arggnn/mutation_routed_whole_tree_gnn_v1.py', 'sha256': '254f5155e31f81ae675e2bd08d38c97c9bf130d47e98efa2deea4c7344d9e073'}, {'path': 'src/arggnn/native_argformer_exec.py', 'sha256': '84ed405379ac56511c2f4722c130e8b884226018bbf8e2949ea2bb4dceeb99d0'}, {'path': 'src/arggnn/observed_variant_residual_v1.py', 'sha256': '695a78d266d8dc4cf226f6e8ff54669979eaf94f30e5cc8cb3062f14ec02c904'}, {'path': 'src/arggnn/persistence_trace_kernel_v1.py', 'sha256': '25bfc4719ae03621289856e09ed4b2040090568aaedee4d05a1e47b218398507'}, {'path': 'src/arggnn/project_budget_v4.py', 'sha256': 'a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47'}, {'path': 'src/arggnn/project_storage_closed_scopes_v1.py', 'sha256': '974bcf342965abb7e7e062547c68123de9bb9414bf9b1f97fd962d1488d3fdd7'}, {'path': 'src/arggnn/project_storage_full_reservation_v1.py', 'sha256': 'c0fdace58e26020c8227606e9b77325bc7308e7ec6691fe3a7e7c18cd4846105'}, {'path': 'src/arggnn/project_storage_terminal_retired_scopes_v1.py', 'sha256': 'bbbdf26206e4e1725c79d2558b4426af848df4cbbd5ac2d0f2bc7051c0606bab'}, {'path': 'src/arggnn/project_storage_terminal_scopes_v1.py', 'sha256': 'f06cf7028af7e86deb54caa1e8a01f0110db47b0d992bfe8aec8ee013ab7d79c'}, {'path': 'src/arggnn/project_storage_terminal_success_scopes_v1.py', 'sha256': '2dfbcc2711fc8310fd8a8b9866ee84e9107e1d2c4356396e0ebc0e99fdb7c3b3'}, {'path': 'src/arggnn/reference_encoder.py', 'sha256': '039f3b25383ea92f243090c2eaf592df590768fea7abf41d15fd7c34a7715138'}, {'path': 'src/arggnn/train_anonymous_clade_flag_control_v1.py', 'sha256': 'cb58ec3f81d81cdcd88637f3ade695a3e3ef97a2d0c10b7713e9bddb43fbb2ed'}, {'path': 'src/arggnn/train_continuous_clade_flank_v1.py', 'sha256': 'eda198ef04192d978433c64f921b865aa89a9304adcc76893fa2852a14d72ae4'}, {'path': 'src/arggnn/train_marker_resolution_readout_v1.py', 'sha256': '386814feb14a64310b8c51c2cfbaf681a59b0273f8be0f72ffde5a222eb08c3c'}, {'path': 'src/arggnn/train_mutation_routed_full_panel_pyg_v1.py', 'sha256': 'c01c4d9506faae7bf537282e4d722b115b3c69cc5de1d4570758136de292dae3'}, {'path': 'src/arggnn/train_mutation_routed_whole_tree_gnn_v1.py', 'sha256': 'bdc02b048b061031f9fb06bedb41668b1851347574e8188f18ef97c1355fd1e6'}, {'path': 'src/arggnn/train_persistence_trace_pilot_v1.py', 'sha256': '5c913f031dde31098fe71fd72bd4c7648b836f7669fe9cef0a8852fcc9a52231'}, {'path': 'research/reviews/order2_cut_memory_implementation_20261015_v1/order2_cut_memory_v1.py', 'sha256': '3b496b163e11147643aa576dfe1f37a75544d466624ef2bd4377fc62d1533265'}, {'path': 'research/reviews/order2_cut_memory_implementation_20261015_v1/integration_hooks_v1.py', 'sha256': '3fa7deaffd101de21f61064ddc9d5fc33f8cea6c9e0a05d79799710b68874acd'}, {'path': 'research/reviews/H049_order2_cut_memory_root_source_20261015_v1/cut_recipe_codec_v1.py', 'sha256': '1cc223028bec30faa74a35a8e98b4357354bbc8665f10f2c5fc2b1cda0c38207'})

RESOURCE_CAPS=("CPU_bank_cap_bytes","CPU_target_cap_bytes","marker_cache_cap_bytes",
    "marker_preparation_cap_bytes","CPU_coordinate_bulk_cap_bytes","CPU_binding_bulk_cap_bytes",
    "grid_cache_cap_bytes","grid_preparation_cap_bytes","GPU_residency_cap_bytes","GPU_grid_cap_bytes",
    "RAM_prediction_cap_bytes","cut_preparation_cap_bytes","cut_recipe_archive_cap_bytes",
    "output_cap_bytes","log_cap_bytes","max_cpu_rss_bytes","max_cuda_reserved_bytes",
    "checkpoint_file_cap_bytes","fixed_metadata_cap_bytes","qualification_allowance_bytes")
PREP_RESOURCE_CAPS=("CPU_bank_cap_bytes","cut_preparation_cap_bytes","cut_recipe_archive_cap_bytes",
    "output_cap_bytes","log_cap_bytes","max_cpu_rss_bytes","fixed_metadata_cap_bytes",
    "qualification_allowance_bytes")
UTILITY=dict(primary_accuracy_gain=.001,primary_boundary_F1_delta_minimum=0.,
    secondary_boundary_F1_gain=.01,secondary_accuracy_delta_minimum=-.0005,
    aggregation="equal_parent_mean_Present182_across8_reusedVAL",
    uncertainty="paired_sample_SD_across8_parents_one_optimizer_seed_development_evidence")

def bound_source_ref_v1(path):
    path=Path(path)
    path=path if path.is_absolute() else ROOT/path
    require(path.resolve()==path and path.is_relative_to(ROOT),"Canonical source")
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def science():
    global np,torch,h038,fit,core,grid_core
    h46.science()
    np,torch,h038,fit=h46.np,h46.torch,h46.h038,h46.fit
    from arggnn import marker_grid_reset_chain_gnn_v1 as grid_core
    from research.reviews.order2_cut_memory_implementation_20261015_v1 import order2_cut_memory_v1 as core
    require(grid_core.original_grid_span_nll_v1 is h46.core.original_grid_span_nll_v1,
        "Exact inherited H044/H046 full-parent loss object")

def fixed_protocol_v1():
    return dict(schema="H049_marker_grid_reset_execution_config_v1",hypothesis_id="H049",
        linked_hypothesis_ids=["H029","H030","H042","H043","H045","H046","H048"],arms=list(ARMS),
        seed=101,dtype="float32",passes=30,updates_per_pass=32,updates_per_arm=960,
        checkpoint_passes=list(PASSES),TRAIN_parent_ids=list(TRAIN),VAL_parent_ids=list(VAL),
        execution_mode="three_arm_sequential_one_parent",parent_order="shuffle_seed101_plus_pass",
        model=MODEL,readout=READOUT,parameters_per_arm=32446,base_parameters=30788,added_parameters=1658,
        common_grid_parameters=31933,added_memory_parameters=513,common_decoder="graph_to_transition",
        rank=8,horizon_generations=350,ell_bp=dict(min=100,max=100000,initial=5000),
        checkpoint_memory=True,updates_all_arms=2880,saved_checkpoints=18,VAL_prediction_products_required_before_truth=144,
        learning_rate=.001,weight_decay=.0001,AdamW_betas=[.9,.999],AdamW_epsilon=1e-8,
        gradient_clip_norm=1.,early_stopping=False,
        first_step_qualification="within_continuous_training_no_extra_updates",
        VAL_influences_optimization=False,same_VAL_is_development_evidence=True,
        selection=SELECTION,metrics=METRICS,core_ref=CORE_REF,scientific_refs=list(SCIENTIFIC_REFS),
        design_ref=DESIGN_REF,
        warmstart_checkpoint_ref=WARM_CHECKPOINT,warmstart_completion_ref=WARM_COMPLETION,
        TRAIN_target_policy="one_complete_original100bp_parent_mass_streamed_once_shared_between_arms",
        dense_prediction_arrays_saved=False,grid_features_saved_to_disk=False,outer_truth_read=False,
        original_haplotypes=952,original_queries=686,original_grid_bp=100,original_region_length_bp=2000000,
        marker_callability="query_called_and_at_least_one_called_reference_in_each_of4classes",
        marker_evidence="disjoint_half_open_original_scoring_bins_FP32",
        chain="analytic_rank_one_reset_generator_exact_balanced_full_position_scan",
        utility_criteria=UTILITY,no_new_operator_priority_claim=True)

def validate_protocol(cfg,phase=None):
    require(isinstance(cfg,dict) and all(cfg.get(k)==v for k,v in fixed_protocol_v1().items())
        and cfg.get("execution_enabled") is True,"Exact registered full-budget H049 protocol")
    for name in (PREP_RESOURCE_CAPS if phase=="prepare" else RESOURCE_CAPS):
        require(type(cfg.get(name)) is int and cfg[name]>0,"Positive reviewed resource: "+name)
    require(cfg.get("root_resource_caps_reviewed") is True,"Root-reviewed genuine count-derived caps")
    require(isinstance(cfg.get("registration_ref"),dict) and cfg["registration_ref"].get("path")
        and cfg.get("producer_ref",{}).get("path")==SOURCE,"Fresh registration and producer binding")


def bind_registration_v1(cfg,reader):
    global REGISTRATION_REF
    REGISTRATION_REF=cfg["registration_ref"]
    registered=reader.decode_json(REGISTRATION_REF)
    require(registered.get("schema")=="H049_order2_cut_memory_registration_v1"
        and registered.get("hypothesis_id")=="H049"
        and registered.get("status")=="registered_before_execution"
        and registered.get("scientific_protocol")==fixed_protocol_v1()
        and registered.get("registered_entry",{}).get("id")=="H049"
        and registered["registered_entry"].get("source_ref")==CORE_REF,
        "Root H049 preregistration exactly binds this scientific protocol")


def grid_upper_bounds_v1(bank,positions):
    return hooks.parent_bounds_v1(bank,positions)


def _extra_bytes(value):
    return sum(t.numel()*t.element_size() for t in value.tensors())

class MarkerGridCacheV1:
    """RAM-only bounded LRU with frozen observation recipes before phase truth."""
    def __init__(self,ctx,reader):
        self.ctx,self.cfg=ctx,ctx["cfg"]
        self.markers=h44.MarkerCacheV1(ctx,reader)
        self.values,self.proofs=OrderedDict(),{}
        self.bytes=self.peak=self.builds=self.hits=0
    def get(self,pid,bank,parent):
        self.ctx["guard"]()
        if pid in self.values:
            value=self.values[pid];value.validate();self.values.move_to_end(pid);self.hits+=1
            return value
        original=self.markers.get(pid,bank,parent)
        limits=grid_upper_bounds_v1(bank,parent["positions"])
        require(limits["named_preparation_allowance_bytes"]<=self.cfg["grid_preparation_cap_bytes"],
            "Count-derived grid/flag preparation allowance exceeds root cap")
        self.ctx["before_grid_build"](pid,bank,limits)
        entry=self.ctx["preparation"]["parents"][pid]
        edges=self.markers.coordinates[entry["coordinates_member_selector"]["bin_edges_bp"]]
        grid=grid_core.prepare_marker_grid_reset_v1(original,edges)
        prepared=self.ctx["cut_preparation"]["parents"][pid]
        cut_receipt=self.ctx["reader"].decode_json(prepared["receipt_ref"])
        require(cut_receipt.get("lossless_roundtrip_verified") is True,
            "Actual CPU preparation verified the compact recipe round trip")
        plan=codec.decode_v1(self.ctx["reader"].read(prepared["arrays_ref"]),cut_receipt,grid.bank)
        value=core.Order2CutBundleV1(grid,plan);value.validate()
        extra=_extra_bytes(value)
        require(extra<=limits["extra_retained_tensor_upper_bytes"],"Actual complete grid/flag tensor bound")
        proof=dict(logical_bounds=limits,actual_extra_tensor_bytes=extra,
            cut_recipe=prepared,cut_metadata=cut_receipt["cut_metadata"],
            runtime_array_identities=[h44.core.tensor_identity_v1(t) for t in value.tensors()],
            marker_recipe=self.markers.proofs[pid],original_observed_entry_ref=self.ctx["entry_refs"][pid],
            targets_read=False,full_backbone_trees_retained=True)
        if pid in self.proofs:require(proof==self.proofs[pid],"Immutable rebuilt H049 observation recipe")
        else:self.proofs[pid]=proof
        size=extra+value.markers.tensor_bytes()
        require(size<=self.cfg["grid_cache_cap_bytes"],"One full grid+marker parent exceeds CPU cache cap")
        while self.values and self.bytes+size>self.cfg["grid_cache_cap_bytes"]:
            _,old=self.values.popitem(last=False)
            self.bytes-=_extra_bytes(old)+old.markers.tensor_bytes()
        self.values[pid]=value;self.bytes+=size;self.peak=max(self.peak,self.bytes);self.builds+=1
        self.ctx["guard"]()
        return value

def load_observations_v1(ctx,reader,ids,cache):
    from arggnn.mutation_routed_full_panel_fast_reader_v1 import read_full_cache_index_v1,read_full_cache_parent_v1
    index=read_full_cache_index_v1(ctx["observation_cache_ref"],reader,registry=ctx["registry"],
        entry_refs=ctx["entry_refs"],source_refs=ctx["observation_source_refs"])
    banks,parents,total={},{},0
    for pid in ids:
        ctx["guard"]()
        item=read_full_cache_parent_v1(pid,index,reader);bank,parent=item["model_inputs"],item["scoring"]
        require(parent["split"]==("train" if pid in TRAIN else "val"),"Original same-parent observed split")
        total+=h038.tensor_bytes_v1(bank)
        require(total<=ctx["cfg"]["CPU_bank_cap_bytes"],"Complete phase CPU-bank cap")
        banks[pid],parents[pid]=bank,parent;cache.get(pid,bank,parent);del item
    require(set(cache.proofs)==set(ids) and set(cache.markers.proofs)==set(ids),
        "All phase observation recipes before any phase truth")
    seal=ctx["output"].save("all_%d_H049_observation_recipes.json"%len(ids),dict(
        schema="H049_complete_observation_barrier_v1",hypothesis_id="H049",parent_ids=list(ids),
        preparation_completion_ref=ctx["preparation_completion_ref"],
            cut_preparation_completion_ref=ctx["cut_preparation_completion_ref"],
        preparation_actual_wait_ref=ctx["preparation_actual_wait_ref"],CPU_bank_tensor_bytes=total,
        grid_recipes=cache.proofs,truth_read=False,full_backbone_trees_retained=True,
        grid_arrays_durable=False))
    return banks,parents,seal

def build_family_v1(payload,device,guard):
    return hooks.build_family_for_engine_v1(payload,device,guard,
        checkpoint_ref=WARM_CHECKPOINT,completion_ref=WARM_COMPLETION)


def device_parent_v1(ctx,bundle,mass_cpu=None):
    bundle.validate()
    persistent=_extra_bytes(bundle)-bundle.features_cpu.numel()*4
    require(persistent<=ctx["cfg"]["GPU_grid_cap_bytes"],"Complete persistent GPU flags/grid cap")
    needed=h038.tensor_bytes_v1(bundle.source_bank)+persistent
    if mass_cpu is not None:needed+=mass_cpu.numel()*4
    require(needed<=ctx["cfg"]["GPU_residency_cap_bytes"],"One shared whole-parent input residency cap")
    moved=bundle.to(ctx["device"])
    require(moved.features_cpu.device.type=="cpu","Full feature tensor stays CPU; only query chunks transfer")
    return moved,None if mass_cpu is None else mass_cpu.to(ctx["device"])

def update_v1(model,optimizer,bundle,mass,guard,*,qualify=False,after_optimizer_step=None):
    """Exact H046 step plus first scheduled-step gradient/state qualification."""
    handles=[];gradient_checks={}
    before={k:p.detach().cpu().clone() for k,p in model.named_parameters()} if qualify else {}
    def hook(name):
        def record(gradient):
            gradient_checks[name]=dict(dtype=str(gradient.dtype),finite=bool(torch.isfinite(gradient).all()),
                nonzero=bool(gradient.count_nonzero()))
            return gradient
        return record
    if qualify:
        for name,p in model.named_parameters():handles.append(p.register_hook(hook(name)))
    try:
        row=h46.update_v1(model,optimizer,bundle,mass,guard,qualify=qualify,
            after_optimizer_step=after_optimizer_step)
    finally:
        for handle in handles:handle.remove()
    if qualify:
        named=dict(model.named_parameters())
        require(set(gradient_checks)==set(named) and all(v["dtype"]=="torch.float32" and v["finite"]
            for v in gradient_checks.values()),"Every trainable parameter receives finite FP32 full-model gradient")
        require(len(optimizer.state)==len(named) and all(int(state["step"].item())==1
            for state in optimizer.state.values()),"First actual complete-model independent AdamW step")
        changed={name:not torch.equal(p.detach().cpu(),before[name]) for name,p in named.items()}
        require(any(changed[n] for n in changed if n.startswith("common.backbone.")),"Actual full-body warmstart updated")
        row["full_model_qualification"]=dict(parameters=32446,trainable_parameter_tensors=len(named),
            gradient_checks=gradient_checks,updated_parameter_tensors=changed,
            inherited_real_H042_pass30_checkpoint_ref=WARM_CHECKPOINT,
            inherited_real_H042_completion_ref=WARM_COMPLETION,
            actual_optimizer_step=1,all_parameters_have_finite_FP32_gradients=True,
            complete_H042_body_trainable=True,all_position_history_retained=True,
            no_disposable_optimizer_update=True,scientific_accuracy_measured=False)
    return row

def run_train_v1(ctx,input_ref):
    cfg,output,guard,device=ctx["cfg"],ctx["output"],ctx["guard"],ctx["device"]
    reader=h42.PhaseReader(ctx["reader"],"train",ctx["bindings"])
    steps={a:0 for a in ARMS};history=[];checkpoints={};qualifications={}
    def record_step(arm):steps[arm]+=1
    try:
        cache=MarkerGridCacheV1(ctx,reader)
        banks,parents,observation_seal=load_observations_v1(ctx,reader,TRAIN,cache)
        payload=h46.load_warmstart_v1(reader)
        models,optimizers,initialization,hashes=build_family_v1(payload,device,guard)
        del payload
        output.save("initialization.json",dict(schema="H049_three_arm_initialization_v1",hypothesis_id="H049",
            initialization=initialization,initial_state_sha256=hashes,observation_seal=observation_seal,
            complete_H042_body_trainable=True,independent_model_and_optimizer_states=True,
            optimizer_moments_and_step_reset=True,VAL_influences_optimization=False,
            scientific_refs=list(SCIENTIFIC_REFS),registration_ref=REGISTRATION_REF,producer_ref=cfg['producer_ref']))
        reader.truth_gate=True
        for at in range(1,31):
            order=list(TRAIN);random.Random(101+at).shuffle(order)
            means={a:0. for a in ARMS};started=time.monotonic()
            for number,pid in enumerate(order,1):
                guard()
                cpu_bundle=cache.get(pid,banks[pid],parents[pid])
                mass_cpu=h44.original_mass_v1(ctx,reader,pid,parents[pid])
                bundle,mass=device_parent_v1(ctx,cpu_bundle,mass_cpu)
                first=at==1 and number==1
                for arm in ARMS:
                    require(all(p.grad is None for m in models.values() for p in m.parameters()),
                            "No gradient lifetime may cross an independent arm step")
                    row=update_v1(models[arm],optimizers[arm],bundle,mass,guard,qualify=first,
                        after_optimizer_step=lambda a=arm:record_step(a))
                    means[arm]+=row["loss"]/32.
                    if first:qualifications[arm]=dict(row,parent_id=pid,optimizer_step=steps[arm],
                        native_nodes=int(bundle.bank.parent.numel()),native_holds=len(bundle.bank.spans),
                        grid_payload_bytes=_extra_bytes(cpu_bundle),
                        actual_counts=cache.proofs[pid]["logical_bounds"],
                        part_of_continuous_960_updates=True,no_extra_qualification_updates=True)
                if first:output.save("first_real_parent_step_qualification.json",dict(hypothesis_id="H049",
                    arms=qualifications,continuous_training=True,no_extra_updates=True))
                del bundle,mass,mass_cpu,cpu_bundle
                if number%8==0:print(json.dumps(dict(stage="H049_three_arm_TRAIN",pass_index=at,
                    complete_parents=number,steps=steps.copy(),seconds=time.monotonic()-started)),flush=True)
            require(steps=={a:32*at for a in ARMS},"Exactly32 actual whole-parent optimizer steps per arm/pass")
            row=dict(pass_index=at,steps=steps.copy(),mean_TRAIN_NLL=means,full_parent_updates_per_arm=32,
                     seconds=time.monotonic()-started)
            history.append(row);output.save("TRAIN_pass_%02d.json"%at,row)
            if at in PASSES:
                groups={}
                for arm in ARMS:
                    guard()
                    cp=dict(schema="H049_marker_grid_reset_checkpoint_v1",hypothesis_id="H049",
                        arm=arm,phase="train",pass_index=at,additional_optimizer_steps=steps[arm],
                        seed=101,parameter_count=32446,model_state={k:v.detach().cpu() for k,v in models[arm].state_dict().items()},
                        optimizer_state=optimizers[arm].state_dict(),optimizer_state_reset=True,
                        complete_H042_body_trainable=True,teacher_used=False,initial_state_sha256=hashes[arm],
                        input_ref=input_ref,config_ref=ctx["config_ref"],core_ref=CORE_REF,readout=READOUT,
                        observation_seal=observation_seal,preparation_completion_ref=ctx["preparation_completion_ref"],
            cut_preparation_completion_ref=ctx["cut_preparation_completion_ref"],
                        cut_preparation_actual_wait_ref=ctx["cut_preparation_actual_wait_ref"],
                        warmstart_checkpoint_ref=WARM_CHECKPOINT,warmstart_completion_ref=WARM_COMPLETION,
                        scientific_refs=list(SCIENTIFIC_REFS),registration_ref=REGISTRATION_REF,producer_ref=cfg['producer_ref'])
                    buffer=io.BytesIO();torch.save(cp,buffer)
                    require(buffer.tell()<=cfg["checkpoint_file_cap_bytes"],"Actual checkpoint byte cap")
                    del buffer
                    groups[arm]=output.save("%s.pass_%02d.pt"%(arm,at),cp,"torch")
                checkpoints[str(at)]=groups
        require(steps=={a:960 for a in ARMS} and set(checkpoints)=={str(p) for p in PASSES},
                "Fixed three_arm30pass/960update/six-checkpoint schedule")
        result=dict(schema="H049_three_arm_train_completion_v1",hypothesis_id="H049",
            status="completed_fixed30pass_three_arm_TRAIN",input_ref=input_ref,config_ref=ctx["config_ref"],
            arms=list(ARMS),steps=steps,history=history,checkpoint_groups=checkpoints,
            initial_state_sha256=hashes,initialization=initialization,observations_seal=observation_seal,
            preparation_completion_ref=ctx["preparation_completion_ref"],
            cut_preparation_completion_ref=ctx["cut_preparation_completion_ref"],
            cut_preparation_actual_wait_ref=ctx["cut_preparation_actual_wait_ref"],warmstart_checkpoint_ref=WARM_CHECKPOINT,
            warmstart_completion_ref=WARM_COMPLETION,core_ref=CORE_REF,scientific_refs=list(SCIENTIFIC_REFS),
            registration_ref=REGISTRATION_REF,producer_ref=cfg['producer_ref'],VAL_truth_read=False,outer_truth_read=False,selection_deferred=True,
            first_step_qualification=qualifications,same_VAL_is_development_evidence=True,
            grid_cache_peak_bytes=cache.peak,marker_cache_peak_bytes=cache.markers.peak,
            grid_builds=cache.builds,grid_cache_hits=cache.hits,grid_features_saved_to_disk=False)
        output.save("completed.json",result);return result
    except Exception as error:
        output.save("failed.json",dict(schema="H049_three_arm_failure_v1",hypothesis_id="H049",phase="train",
            status="failed_partial_preserved",error_type=type(error).__name__,error=str(error),
            logical_steps=steps.copy(),completed_checkpoint_groups=checkpoints,automatic_replay=False,
            VAL_truth_read=False,outer_truth_read=False))
        raise

class H049PhaseReaderV1(h42.PhaseReader):
    """Exact known RAM packet bytes only; inherited target allowlist/gate unchanged."""
    def __init__(self,base,phase,bindings,*,virtual_prefix):
        super().__init__(base,phase,bindings)
        prefix=Path(virtual_prefix)
        require(not prefix.is_absolute() and ".." not in prefix.parts and "truth" not in prefix.parts,
            "Owned canonical RAM packet namespace")
        self.virtual_prefix=prefix;self.packet_bytes={};self.packet_refs={}
    def register_packet_v1(self,key,packet):
        require(self.phase=="evaluate" and not self.truth_gate and self.ram is not None
            and not self.ram.verified and key in {(a,at,p) for a in ARMS for at in PASSES for p in VAL},
            "Exact observation-only probability packet producer")
        arm,at,pid=key
        require(packet["schema"]=="H049_marker_original_grid_RAM_seal_v1"
            and packet["hypothesis_id"]=="H049" and packet["arm"]==arm
            and packet["pass_index"]==at and packet["parent_id"]==pid,"Packet/key identity")
        path=str(self.virtual_prefix/("%s.pass_%02d.%s.json"%(arm,at,pid)))
        require(path not in self.packet_bytes,"RAM packet replacement denied")
        payload=(json.dumps(packet,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
        ref=dict(path=path,sha256=hashlib.sha256(payload).hexdigest())
        self.packet_bytes[path]=payload;self.packet_refs[path]=dict(ref)
        return ref
    def read(self,record,**kwargs):
        path=record["path"]
        if path in self.packet_bytes:
            payload=self.packet_bytes[path]
            require(self.phase=="evaluate" and record==self.packet_refs[path]
                and hashlib.sha256(payload).hexdigest()==record["sha256"],
                "Exact registered immutable RAM packet bytes")
            return payload
        return super().read(record,**kwargs)

class DenseRAMMatrixV1:
    """All3arms x6checkpoints x8 original dense grids before any VAL truth."""
    def __init__(self,ctx,reader,parents,banks):
        self.ctx,self.reader,self.parents,self.banks=ctx,reader,parents,banks
        self.values,self.records,self.lookup,self.packets={},{},{},{}
        self.bytes,self.verified,self.barrier_ref=0,False,None
    def seal(self,arm,at,pid,value,checkpoint):
        key=(arm,at,pid);require(key not in self.values,"Immutable product replacement denied")
        parent=self.parents[pid]
        require(self.bytes+4*len(parent["positions"])*686*4<=self.ctx["cfg"]["RAM_prediction_cap_bytes"],
            "Preflight complete dense prediction matrix cap")
        value=np.array(value,dtype=np.float32,copy=True,order="C")
        require(value.shape==(len(parent["positions"]),686,4) and np.isfinite(value).all()
            and np.all(value>=0) and np.allclose(value.sum(-1),1,rtol=1e-5,atol=1e-6),"Complete original dense FP32 probabilities")
        self.bytes+=int(value.nbytes);require(self.bytes<=self.ctx["cfg"]["RAM_prediction_cap_bytes"],"All144 dense RAM cap")
        value.setflags(write=False)
        geometry={k:h42.shared.array_identity(parent[k]) for k in
            ("positions","edges","retained_indexes","query_slots","raw_query_nodes","cohorts")}
        geometry["readout_hold"]=h42.shared.array_identity(self.banks[pid].readout_hold.numpy())
        packet=dict(schema="H049_marker_original_grid_RAM_seal_v1",hypothesis_id="H049",arm=arm,pass_index=at,
            parent_id=pid,checkpoint_ref=checkpoint,probability=h42.shared.array_identity(value),geometry=geometry,
            core_ref=CORE_REF,producer_ref=self.ctx["cfg"]["producer_ref"],prediction_array_durable=False,targets_opened_before_seal=False)
        ref=self.reader.register_packet_v1(key,packet)
        self.values[key],self.records[key],self.packets[key]=value,ref,packet;self.lookup[ref["path"]]=key
    def verify(self):
        require(set(self.values)=={(a,at,p) for a in ARMS for at in PASSES for p in VAL},"All144 products before truth")
        for key,value in self.values.items():
            self.ctx["guard"]()
            require(not value.flags.writeable and h42.shared.array_identity(value)==self.packets[key]["probability"]
                and self.reader.decode_json(self.records[key])==self.packets[key],"Frozen probability content/record changed")
        self.barrier_ref=self.ctx["output"].save("all144_original_grid_VAL_RAM_sealed_before_truth.json",dict(
            schema="H049_marker_all144_VAL_barrier_v1",hypothesis_id="H049",arms=list(ARMS),
            checkpoint_passes=list(PASSES),VAL_parent_ids=list(VAL),producer_ref=self.ctx["cfg"]["producer_ref"],RAM_prediction_bytes=self.bytes,
            complete_dense_prediction_products=144,all144_sealed_before_any_VAL_truth=True,
            seals=[self.records[k] for k in sorted(self.records)],
            packets=[dict(record=self.records[k],packet=self.packets[k]) for k in sorted(self.packets)],
            virtual_packet_bytes_resolved_only_by_known_RAM_reader=True,outer_truth_read=False))
        self.verified=True
    def grid(self,record,parent):
        require(self.verified and record["path"] in self.lookup,"Complete verified144-product barrier")
        key=self.lookup[record["path"]];value=self.values[key]
        require(record==self.records[key] and key[2]==parent["parent_id"] and not value.flags.writeable
            and h42.shared.array_identity(value)==self.packets[key]["probability"],"Current exact original-grid same-parent product")
        return value

def load_checkpoints_v1(reader,train,cfg):
    require(train.get("schema")=="H049_three_arm_train_completion_v1"
        and train.get("hypothesis_id")=="H049" and train.get("status")=="completed_fixed30pass_three_arm_TRAIN"
        and train.get("arms")==list(ARMS) and train.get("steps")=={a:960 for a in ARMS}
        and train.get("core_ref")==CORE_REF and train.get("scientific_refs")==list(SCIENTIFIC_REFS)
        and train.get("registration_ref")==REGISTRATION_REF and train.get("producer_ref")==cfg["producer_ref"]
        and train.get("VAL_truth_read") is False and train.get("outer_truth_read") is False
        and train.get("preparation_completion_ref")==cfg["preparation_completion_ref"]
        and train.get("cut_preparation_completion_ref")==cfg["cut_preparation_completion_ref"]
        and train.get("cut_preparation_actual_wait_ref")==cfg["cut_preparation_actual_wait_ref"]
        and set(train.get("checkpoint_groups",{}))=={str(p) for p in PASSES}
        and len(set(train["initial_state_sha256"].values()))==1,
        "Actual full matched H049 TRAIN completion and immutable original preparation")
    train_cfg=reader.decode_json(train["config_ref"]);validate_protocol(train_cfg);validate_protocol(cfg)
    require({k:v for k,v in train_cfg.items() if k not in RESOURCE_CAPS}
            =={k:v for k,v in cfg.items() if k not in RESOURCE_CAPS},
            "Only reviewed resource caps may differ between TRAIN/evaluation")
    require(len(train.get("history",[]))==30 and all(r["pass_index"]==i
        and r["steps"]=={a:32*i for a in ARMS} and r["full_parent_updates_per_arm"]==32
        for i,r in enumerate(train["history"],1)) and set(train.get("first_step_qualification",{}))==set(ARMS),
        "All30 completed passes and actual scheduled first-step qualifications")
    states,refs={},{}
    for arm in ARMS:
        for at in PASSES:
            require(set(train["checkpoint_groups"][str(at)])==set(ARMS),"Exactly3 arms per checkpoint group")
            ref=train["checkpoint_groups"][str(at)][arm]
            cp=torch.load(io.BytesIO(reader.read(ref)),map_location="cpu",weights_only=False)
            require(cp.get("schema")=="H049_marker_grid_reset_checkpoint_v1"
                and cp.get("hypothesis_id")=="H049" and cp.get("arm")==arm and cp.get("phase")=="train"
                and cp.get("pass_index")==at and cp.get("additional_optimizer_steps")==32*at
                and cp.get("seed")==101 and cp.get("parameter_count")==32446
                and cp.get("optimizer_state_reset") is True and cp.get("teacher_used") is False
                and cp.get("complete_H042_body_trainable") is True
                and cp.get("core_ref")==CORE_REF and cp.get("readout")==READOUT
                and cp.get("initial_state_sha256")==train["initial_state_sha256"][arm]
                and cp.get("input_ref")==train["input_ref"] and cp.get("config_ref")==train["config_ref"]
                and cp.get("preparation_completion_ref")==cfg["preparation_completion_ref"]
                and cp.get("cut_preparation_completion_ref")==cfg["cut_preparation_completion_ref"]
                and cp.get("cut_preparation_actual_wait_ref")==cfg["cut_preparation_actual_wait_ref"]
                and cp.get("warmstart_checkpoint_ref")==WARM_CHECKPOINT
                and cp.get("warmstart_completion_ref")==WARM_COMPLETION
                and cp.get("scientific_refs")==list(SCIENTIFIC_REFS)
                and cp.get("registration_ref")==REGISTRATION_REF and cp.get("producer_ref")==cfg["producer_ref"]
                and all(isinstance(v,torch.Tensor) and v.dtype==torch.float32
                        and bool(torch.isfinite(v).all()) for v in cp["model_state"].values()),
                "Exact complete FP32 H049 checkpoint provenance")
            states[arm,at],refs[arm,at]=cp["model_state"],ref
    return states,refs


def practical_criteria_v1(full):
    result=hooks.paired_effects_v1(full)
    for row in result["comparisons"].values():
        accuracy=row["mean_Present182_accuracy_delta"]
        boundary=row["mean_1kb_boundary_F1_delta"]
        row["meets_primary_utility"]=(accuracy>=UTILITY["primary_accuracy_gain"]
            and boundary is not None and boundary>=UTILITY["primary_boundary_F1_delta_minimum"])
        row["meets_secondary_boundary_utility"]=(boundary is not None
            and boundary>=UTILITY["secondary_boundary_F1_gain"]
            and accuracy>=UTILITY["secondary_accuracy_delta_minimum"])
    result.update(registered_criteria=UTILITY,threshold_registration_required=False,
        threshold_decision="registered_criteria_applied_to_each_planned_comparator",
        primary_utility_against_both=all(r["meets_primary_utility"] for r in result["comparisons"].values()),
        secondary_boundary_utility_against_both=all(r["meets_secondary_boundary_utility"] for r in result["comparisons"].values()),
        global_best_H045_and_native_baselines="separate_root_reporting_required_no_global_top1_claim",
        priority_claim=False,domain_pooling_denoising_utility_is_empirical=True)
    return result


def run_evaluate_v1(ctx,input_ref):
    from arggnn.mutation_routed_full_panel_primary_v1 import FullPanelPrimaryValidationV1
    cfg,output,guard=ctx["cfg"],ctx["output"],ctx["guard"]
    reader=H049PhaseReaderV1(ctx["reader"],"evaluate",ctx["bindings"],
        virtual_prefix=str(output.path.relative_to(ROOT)/"H049_RAM_probability_packets"))
    train=reader.decode_json(ctx["train_completion_ref"])
    states,checkpoints=load_checkpoints_v1(reader,train,cfg)
    cache=MarkerGridCacheV1(ctx,reader)
    banks,parents,observation_seal=load_observations_v1(ctx,reader,VAL,cache)
    ram=DenseRAMMatrixV1(ctx,reader,parents,banks);reader.ram=ram
    models={a:hooks.evaluation_model_v1(a,ctx["device"]) for a in ARMS}
    for pid in VAL:
        cpu_bundle=cache.get(pid,banks[pid],parents[pid])
        bundle,_=device_parent_v1(ctx,cpu_bundle)
        for arm in ARMS:
            for at in PASSES:
                guard();models[arm].load_state_dict(states[arm,at],strict=True)
                with torch.no_grad(),torch.autocast(device_type=ctx["device"].type,enabled=False):
                    probability=models[arm](bundle).softmax(-1)
                ram.seal(arm,at,pid,probability.cpu().numpy(),checkpoints[arm,at]);del probability
                print(json.dumps(dict(stage="H049_VAL_prediction",sealed_products=len(ram.values),total_products=144)),flush=True)
        del bundle,cpu_bundle
    del models,states
    ram.verify();reader.truth_gate=True;ctx["VAL_truth_gate_opened"]=True
    selected,histories,full,metric_receipts={},{},{},{}
    for arm in ARMS:
        validation=FullPanelPrimaryValidationV1(parents,ctx["bindings"],reader,fit.metric_helpers_v1(),cfg)
        rows=[]
        for at in PASSES:
            guard();cells,means=validation.score([(p,ram.records[arm,at,p]) for p in VAL])
            rows.append(dict(pass_index=at,cells=cells,equal_parent_metrics=means,
                checkpoint=checkpoints[arm,at],selection_rule=SELECTION))
        best=min(rows,key=lambda row:h038.selection_key_v1(row["equal_parent_metrics"],row["pass_index"]))
        cells,means=validation.full_final([(p,ram.records[arm,best["pass_index"],p]) for p in VAL])
        full[arm]=dict(cells=cells,equal_parent_metrics=means,selection_revised=False)
        selected[arm],histories[arm],metric_receipts[arm]=best,rows,validation.receipt()
    result=dict(schema="H049_three_arm_VAL_completion_v1",hypothesis_id="H049",
        status="completed_original8VAL_three_arm_development_selection",input_ref=input_ref,
        train_completion_ref=ctx["train_completion_ref"],observations_seal=observation_seal,
        producer_ref=cfg["producer_ref"],scientific_refs=list(SCIENTIFIC_REFS),registration_ref=REGISTRATION_REF,
        selection_rule=SELECTION,selected=selected,history=histories,selected_full_diagnostics=full,
        metric_receipts=metric_receipts,registered_practical_criteria=practical_criteria_v1(full),
        optimization_performed=False,all144_RAM_seals_before_any_VAL_truth=True,
        RAM_dense_probability_bytes=ram.bytes,complete_probability_matrix_manifest_ref=ram.barrier_ref,
        virtual_seals_durable_in_matrix_manifest=True,dense_prediction_arrays_saved=False,
        outer_truth_read=False,same_VAL_is_development_evidence=True)
    output.save("completed.json",result);return result


def run_context_v1(ctx,input_ref):
    require(ctx.get("root_authorizer")=="root_agent" and ctx.get("admitted") is True
        and ctx.get("phase") in ("train","evaluate") and ctx.get("arm")=="joint_three_arm"
        and ctx.get("physical_gpu") in (4,5) and callable(ctx.get("guard"))
        and callable(ctx.get("before_grid_build")),"Independently admitted three_arm H049 root context")
    ctx["guard"]()
    validate_protocol(ctx["cfg"])
    bind_registration_v1(ctx["cfg"],ctx["reader"])
    for record in (*SCIENTIFIC_REFS,REGISTRATION_REF,DESIGN_REF,ctx["cfg"]["producer_ref"]):bound(record)
    require(ctx["reader"].root.resolve()==ROOT and ctx["device"].type=="cuda"
        and ctx["reader"].decode_json(ctx["config_ref"])==ctx["cfg"],"Root-leased CUDA reader and exact config")
    require(ctx["cfg"]["preparation_completion_ref"]==ctx["preparation_completion_ref"],
        "Exact immutable all40 original H044 preparation")
    ids=TRAIN if ctx["phase"]=="train" else VAL
    require(set(ctx["bindings"])==set(ids) and set(ctx["entry_refs"])==set(TRAIN+VAL),
        "Exact phase targets and40 original observations")
    science();ctx["guard"]()
    if ctx["phase"]=="train":return run_train_v1(ctx,input_ref)
    try:return run_evaluate_v1(ctx,input_ref)
    except Exception as error:
        ctx["output"].save("failed.json",dict(schema="H049_three_arm_failure_v1",hypothesis_id="H049",
            phase="evaluate",status="failed_partial_preserved",error_type=type(error).__name__,error=str(error),
            VAL_truth_gate_opened=ctx.get("VAL_truth_gate_opened",False),outer_truth_read=False,automatic_replay=False))
        raise

def describe_v1():
    return dict(status="source_ready_no_execution_authority",hypothesis_id="H049",arms=list(ARMS),
        parameters_per_arm=32446,TRAIN_parents=32,VAL_parents=8,passes=30,updates_per_arm=960,
        checkpoint_passes=list(PASSES),first_scheduled_update_qualifies_no_extra_updates=True,
        all144_products_before_VAL_truth=True,Torch_imported=("torch" in sys.modules),
        root_context_factory_and_admission_required=True,core_ref=CORE_REF,
        registration_ref=REGISTRATION_REF,resource_caps=list(RESOURCE_CAPS))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true");parser.add_argument("--input")
    parser.add_argument("--expected-input-sha256")
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(describe_v1(),indent=2));return 0
    require(args.input and args.expected_input_sha256,"Exact pinned active root input required")
    record=dict(path=args.input,sha256=args.expected_input_sha256)
    spec=json.loads(bound(record))
    require(spec.get("schema")=="H049_marker_grid_root_input_v1" and spec.get("status")=="active"
        and spec.get("root_authorizer")=="root_agent" and spec.get("hypothesis_id")=="H049"
        and spec.get("arm")=="joint_three_arm" and spec.get("physical_gpu") in (4,5),
        "Active separate H049 root capability required")
    require(spec.get("producer_ref")==dict(path=SOURCE,sha256=hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest()),
        "Actual new H049 producer binding")
    require(isinstance(spec.get("source_sha256"),dict) and SOURCE in spec["source_sha256"],
        "Complete root execution closure must include actual producer")
    for path,digest in spec["source_sha256"].items():bound(dict(path=path,sha256=digest))
    for source in SCIENTIFIC_REFS:
        require(spec["source_sha256"].get(source["path"])==source["sha256"],"Inherited scientific closure bound by root")
    context_ref=spec["root_context_module_ref"];bound(context_ref)
    require(spec["source_sha256"].get(context_ref["path"])==context_ref["sha256"],
        "Actual current root context belongs to execution closure")
    loader=importlib.util.spec_from_file_location("_H049_current_root_context",ROOT/context_ref["path"])
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    context=module.build_context_v1(record,spec)
    require(context.get("physical_gpu")==spec["physical_gpu"] and context.get("phase")==spec["phase"]
        and context["cfg"]["producer_ref"]==spec["producer_ref"],"Same actual capability and root context")
    run_context_v1(context,record);return 0

if __name__=="__main__":raise SystemExit(main())
