"""New current-four-head TreeLoom application to existing H056 observed banks.

Descriptive CPU application only: no optimization, truth, calibration, checkpoint
selection, native inference or data preparation. Historical H045 outputs retain
their old model identity. Each genuine chromosome window is one full context.
"""
from __future__ import annotations
import copy
import importlib
import os
from pathlib import Path
import resource
import tempfile
import uuid

from . import contracts as c
from . import treeloom
from . import treeloom_fresh as fresh
from . import european_training as training
from . import european_c4_training as current
from . import american_preparation as generation
from . import treeloom_training as memory

SCHEMA = "current_TreeLoom_C4_H056_descriptive_application_v1"
CONFIG_REF = dict(path="research/reviews/H056_three_bank_application_actual_root_20261016_v2/config.json",
    sha256="36d64ed1990597154bd44feeeac6987600cecb23d1d0404fc907931db8e8c849")
MANIFEST_REF = dict(path="research/reviews/H056_three_bank_application_actual_root_20261016_v2/source_manifest.json",
    sha256="6e4f2de045d5e59d30f62370d867318390be96acc065ac01c1d42ebec774563c")
CHROMOSOMES = ("20","21","22")
CONDITION = "current_TreeLoom_C4_H056_H36"
CLASSES = current.CLASSES
_MODEL_MODULE = None


def _need(ok,message):
    if not ok:
        raise c.ContractError(message)


def requirements():
    manifest = c.read_ref(MANIFEST_REF)
    refs = [dict(path=p,sha256=h) for p,h in manifest["source_sha256"].items()]
    for ref in refs:
        _need(c.file_ref(ref["path"]) == ref,"Frozen H056 observed-reader source changed")
    base = current.requirements()
    source_refs = [*base["source_refs"],*refs,CONFIG_REF,MANIFEST_REF,c.file_ref(__file__)]
    return dict(source_refs=list({r["path"]:r for r in source_refs}.values()),
        model_parameters=32325,trainable_tensors=49,model_family="current_frozen_TreeLoom_H050_C4",
        model_class_order=list(CLASSES),observation_class_order=["ANA","CHG","WHG","EHG"],
        class_name_alias_metadata_only_no_column_permutation=True,
        observation_haplotypes=36,reference_haplotypes=16,query_haplotypes=20,
        chromosomes=list(CHROMOSOMES),whole_genuine_window_context=True,
        optimization_steps=0,query_ancestry_truth_access=False,ancestry_accuracy_available=False,
        old_H045_result_reused_or_relabelled=False,scientific_payloads_opened=False,jobs_started=0)


def _inputs():
    cfg = c.read_ref(CONFIG_REF)
    _need(set(cfg["chromosomes"]) == set(CHROMOSOMES),"All original fixed observed windows required")
    return copy.deepcopy(cfg["chromosomes"])


def _completion(ref):
    current.original.gpu._public_ref(ref)
    complete = c.read_ref(ref)
    _need(complete.get("protocol_id") == current.PROTOCOL
          and complete.get("class_order") == list(CLASSES)
          and complete.get("optimizer_updates") == 90
          and complete.get("selected_epoch") == complete.get("completed_epochs") == 30,
          "Actual separately trained current four-head final30/90 checkpoint required")
    current._parent_plan(complete["sampling_plan_ref"],current.PROTOCOL)
    training._fit_completion(ref,protocol_id=current.PROTOCOL,
        sampling_plan_ref=complete["sampling_plan_ref"],train_count=3)
    _need(complete["checkpoint_ref"]["sha256"] !=
          "bfd8a87632015a900d9da6009d4b0ee037958460038d7949d0998e80b770796b",
          "Historical H045 checkpoint cannot be substituted or relabelled")
    return complete


def build_spec(*,fit_completion_ref,output_directory,max_rss_bytes=4*2**30,
               max_bank_read_bytes=18*2**20,max_checkpoint_read_bytes=8*2**20,
               max_episode_payload_bytes=2**20,max_nodes=100000,max_holds=1000,
               max_output_bytes=32*2**20):
    _completion(fit_completion_ref)
    _need(all(type(value) is int and value > 0 for value in
              (max_rss_bytes,max_bank_read_bytes,max_checkpoint_read_bytes,
               max_episode_payload_bytes,max_nodes,max_holds,max_output_bytes))
          and max_output_bytes >= 8*2**20,"Explicit complete CPU application caps required")
    out = generation._path(output_directory)
    _need(out.is_relative_to(c.ROOT/"experiments/results") and not out.exists(),
          "Fresh current-model empirical scope required")
    return dict(schema=SCHEMA,fit_completion_ref=copy.deepcopy(fit_completion_ref),
        chromosomes=_inputs(),output_directory=str(out.relative_to(c.ROOT)),
        max_rss_bytes=max_rss_bytes,max_bank_read_bytes=max_bank_read_bytes,
        max_checkpoint_read_bytes=max_checkpoint_read_bytes,
        max_episode_payload_bytes=max_episode_payload_bytes,max_nodes=max_nodes,max_holds=max_holds,
        max_output_bytes=max_output_bytes,requirements=requirements(),
        optimization_steps=0,query_ancestry_truth_access=False,ancestry_accuracy_available=False)


def _model_module():
    global _MODEL_MODULE
    if _MODEL_MODULE is None:
        requirements()
        original = treeloom._european_module()
        conditions = dict(original.CONDITIONS)
        conditions[CONDITION] = dict(haplotypes=36,queries=20,length_bp=10000000,
                                    output_classes=CLASSES,parameters=32325)
        _MODEL_MODULE = fresh._private_module(original,"H056_current_C4_"+uuid.uuid4().hex,
                                             {"CONDITIONS":conditions})
    return _MODEL_MODULE


def build_variable_model():
    import torch
    implementation = _model_module()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        model = implementation.FrozenEuropeanTreeLoomV1(condition=CONDITION)
    _need(sum(p.numel() for p in model.parameters()) == 32325
          and len(list(model.parameters())) == 49 and model.output_classes == CLASSES,
          "Same current four-class state keys/shapes and learned architecture required")
    return model


def _metadata(chrom,item):
    from research.reviews.H056_H36_implementation_20261016_v1 import contract_v1 as old
    done = c.read_ref(item["bank_completion_ref"])
    wait = c.read_ref(item["preparation_wait_ref"])
    counts = old.validate_counts_v1(item["counts"],graph=True)
    _need(done.get("schema") == "H056_empirical_CPU_preparation_completed_v1"
          and done.get("status") == "completed_observation_preparation"
          and done.get("protocol_ref") == old.PROTOCOL_REF
          and done.get("chromosome") == chrom and done.get("counts") == counts
          and done.get("bank_ref") == item["bank_ref"]
          and done.get("no_query_ancestry_truth") is True
          and done.get("no_model_or_checkpoint_executed") is True and done.get("no_GPU_used") is True
          and done["metadata"]["feature_semantics"] == old.FEATURE_SEMANTICS
          and done["metadata"]["protocol_ref"] == old.PROTOCOL_REF
          and done["metadata"]["chromosome"] == chrom
          and done["metadata"]["expected_counts"] == counts
          and done["metadata"]["observation_refs"]["roster"] == item["roster_ref"]
          and done["metadata"]["no_query_truth_read"] is True
          and wait.get("actual_exit_code") == 0 and wait.get("actual_direct_wait_observed") is True
          and wait.get("process_group_drained") is True
          and wait.get("completed_ref") == item["bank_completion_ref"],
          "Exact completed original observed H056 bank, sample roster and genuine drain required")
    return done


def load_bank(chrom,item,*,max_bank_read_bytes,resource_check):
    """Only original observed tensors; no VCF/tree preparation or old-model call."""
    import torch
    from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1,validate_inputs_v1
    from arggnn.anonymous_clade_flag_control_v1 import AnonymousCladeFlagsV1
    from research.reviews.H056_H36_implementation_20261016_v1.panel_h36_v1 import panel_metadata_v1
    from research.reviews.H056_H36_implementation_20261016_v1.reader_h36_v1 import bank_array_manifest_v1
    done = _metadata(chrom,item)
    path = generation._path(item["bank_ref"]["path"])
    _need(path.stat().st_size <= max_bank_read_bytes
          and generation._reference(path) == item["bank_ref"],"Exact bounded original observed bank SHA required")
    resource_check("current_TreeLoom_before_original_observed_bank")
    saved = torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    _need(set(saved) == {"schema","protocol_ref","chromosome","input_ref","metadata","bank_tensors","flag_node_flags"}
          and saved["schema"] == "H056_empirical_CPU_observation_bank_v1"
          and saved["protocol_ref"] == done["protocol_ref"] and saved["chromosome"] == chrom
          and saved["input_ref"] == done["input_ref"] and saved["metadata"] == done["metadata"],
          "Only sealed observed bank tensors/flags/metadata accepted")
    bank = MutationRoutedWholeTreeInputsV1(**saved["bank_tensors"])
    report = validate_inputs_v1(bank,expected_haplotypes=36)
    counts = item["counts"]
    _need(bank.sample_index.shape == (counts["native_trees"],36)
          and bank.genotype.shape == (counts["sites"],36)
          and bank.parent.numel() == counts["vertex_occurrences"]
          and torch.equal(bank.query_column,torch.arange(16,36))
          and torch.equal(bank.readout_hold,torch.arange(counts["native_trees"]))
          and bank_array_manifest_v1(bank) == done["metadata"]["model_input_tensor_manifest"],
          "Every original site/node/hold/sample and tensor hash must remain exact")
    from .european_treeloom import _json_ref
    roster = _json_ref(item["roster_ref"],2*2**20)
    panel = panel_metadata_v1(roster)
    _need(torch.equal(bank.reference_class,torch.tensor(panel["reference_class"],dtype=torch.long))
          and all(float(bank.sample_age[i]) == float(torch.tensor(row["age_generations"],dtype=torch.float32))
                  for i,row in enumerate(roster)),
          "Exact actual query/reference class/date/haplotype mapping required")
    flags = AnonymousCladeFlagsV1.from_verified_arrays(saved["flag_node_flags"].numpy(),
                                                     done["metadata"]["exact_clade_flags"])
    return bank,flags,done,roster


def make_bundle(bank,known_flags,*,chromosome,source_refs,max_nodes,max_holds,max_episode_payload_bytes):
    import torch
    from arggnn.ordered_clade_episode_gnn_v1 import build_clade_episode_arrays_v1,OrderedCladeEpisodesV1
    implementation = _model_module()
    _need(bank.sample_index.shape[1] == 36 and bank.query_column.numel() == 20
          and bank.parent.numel() <= max_nodes and len(bank.spans) <= max_holds,
          "Complete genuine H36 bank must fit explicit bounds; never reduce context")
    arrays,metadata = build_clade_episode_arrays_v1(
        bank.parent.numpy(),bank.sample_index.numpy(),bank.spans.numpy(),
        max_payload_bytes=max_episode_payload_bytes)
    episodes = OrderedCladeEpisodesV1.from_verified_arrays(arrays,metadata)
    _need(torch.equal(episodes.flags.node_flags,known_flags.node_flags),
          "Rebuilt observed episode flags must equal exact saved aligned flags")
    cut,cut_metadata = implementation.memory_core.prepare_cut_plan_v1(
        bank,expected_haplotypes=36,max_nodes=max_nodes,max_holds=max_holds)
    parent = "existing_H056_observed_chr"+chromosome
    refs = dict(source_refs,parent_id=parent,query_truth_in_inputs=False,
        all_observed_trees_nodes_sites_samples_retained=True,
        supported_length_bp=float(bank.spans.double().sum()),requested_length_bp=10000000)
    bundle = implementation.FrozenEuropeanBundleV1(bank,episodes,cut,CONDITION,parent,
        "evaluation_prediction",refs,
        implementation.memory_core._stamps(implementation._bank_tensors(bank)),
        implementation.memory_core._stamps(implementation._episode_tensors(episodes)))
    bundle.validate()
    return bundle,metadata,cut_metadata


def descriptive_summary(chromosome_results,subjects):
    """Span→haplotype pair→covered-bp chromosome→equal-person means; no accuracy."""
    import numpy as np
    _need(set(chromosome_results) == set(CHROMOSOMES) and len(subjects) == 10,
          "All fixed windows and original ten query people required")
    people,weights,windows = [],[],[]
    for chrom in CHROMOSOMES:
        item = chromosome_results[chrom]
        values = np.asarray(item["haplotype_proportions"],dtype=np.float64)
        weight = float(item["covered_bp"])
        _need(values.shape == (20,4) and np.isfinite(values).all() and weight > 0
              and np.allclose(values.sum(-1),1.,atol=1e-6,rtol=0),
              "Every original query probability mean and genuine physical span required")
        persons = values.reshape(10,2,4).mean(1)
        people.append(persons);weights.append(weight)
        windows.append(dict(chromosome=chrom,covered_bp=weight,
            person_probabilities=persons.tolist(),
            later_minus_earlier=(persons[5:].mean(0)-persons[:5].mean(0)).tolist()))
    pooled = np.einsum("c,cpa->pa",weights,np.asarray(people))/sum(weights)
    return dict(schema="current_TreeLoom_C4_H056_descriptive_summary_v1",
        class_order=list(CLASSES),query_subject_ids=list(subjects),chromosomes=windows,
        pooled_person_probabilities=pooled.tolist(),earlier_mean=pooled[:5].mean(0).tolist(),
        later_mean=pooled[5:].mean(0).tolist(),
        later_minus_earlier=(pooled[5:].mean(0)-pooled[:5].mean(0)).tolist(),
        pooling="genuine_span_bp_then_two_haplotypes_per_person_then_covered_bp_chromosomes_then_equal_people",
        optimization_steps=0,ancestry_accuracy_available=False,mixture_calibration_claimed=False,
        old_H045_result_reused_or_relabelled=False,uncertainty_or_significance_claimed=False)


def run(spec,*,resource_check):
    kwargs = {key:spec[key] for key in ("fit_completion_ref","output_directory","max_rss_bytes",
        "max_bank_read_bytes","max_checkpoint_read_bytes","max_episode_payload_bytes",
        "max_nodes","max_holds","max_output_bytes")}
    _need(spec == build_spec(**kwargs),"Exact current-model descriptive application specification required")
    _need(callable(resource_check) and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
          "ROOT-supervised CPU-only empirical worker required")
    def check(*args):
        resource_check(*args)
        _need(max(memory._current_rss_bytes(),resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
              <= spec["max_rss_bytes"],"Whole CPU application RSS envelope exceeded")
    check("current_TreeLoom_before_application")
    writer = generation._Writer(generation._path(spec["output_directory"]),spec["max_output_bytes"],check)
    temp = writer.path/".tmp";temp.mkdir();tempfile.tempdir=str(temp)
    for key in ("TMPDIR","TMP","TEMP"):
        os.environ[key] = str(temp)
    import numpy as np
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    _need(torch.get_default_dtype() == torch.float32 and not torch.is_autocast_enabled("cpu"),
          "Original full FP32 CPU model arithmetic required")
    complete = _completion(spec["fit_completion_ref"])
    checkpoint = generation._path(complete["checkpoint_ref"]["path"])
    _need(checkpoint.stat().st_size <= spec["max_checkpoint_read_bytes"]
          and generation._reference(checkpoint) == complete["checkpoint_ref"],"Actual bounded currentC4checkpoint SHA required")
    saved = torch.load(checkpoint,map_location="cpu",weights_only=True,mmap=True)
    model = build_variable_model()
    training._validate_saved(saved,model,protocol_id=current.PROTOCOL,
        sampling_plan_ref=complete["sampling_plan_ref"],updates=90,barrier_ref=complete["observation_barrier_ref"])
    del saved
    model.eval()
    results,products,subjects = {},{},None
    for chrom in CHROMOSOMES:
        check("current_TreeLoom_before_chromosome")
        item = spec["chromosomes"][chrom]
        bank,flags,done,roster = load_bank(chrom,item,max_bank_read_bytes=spec["max_bank_read_bytes"],
                                         resource_check=check)
        bundle,episode_meta,cut_meta = make_bundle(bank,flags,chromosome=chrom,source_refs=item,
            max_nodes=spec["max_nodes"],max_holds=spec["max_holds"],
            max_episode_payload_bytes=spec["max_episode_payload_bytes"])
        # One original whole-window tree/history/GRU forward. No2Mb resets,
        # cross-chromosome carry, gradients or optimizer are introduced.
        with torch.inference_mode():
            probabilities = model(bank,episodes=bundle).softmax(-1).numpy()
        t = item["counts"]["native_trees"]
        _need(probabilities.shape == (t,20,4) and probabilities.dtype == np.float32
              and np.isfinite(probabilities).all() and np.allclose(probabilities.sum(-1),1.,atol=1e-6,rtol=0),
              "Every genuine native-hold/query FP32 prediction required")
        intervals = np.asarray(done["metadata"]["graph_projection"]["coordinate_view"]["intervals_regional_bp0"],
                               dtype=np.float64)
        spans = np.diff(intervals,axis=1)[:,0]
        _need(intervals.shape == (t,2) and np.isfinite(intervals).all() and (spans > 0).all()
              and np.array_equal(intervals[:-1,1],intervals[1:,0])
              and np.array_equal(spans,bank.spans.numpy().astype(np.float64))
              and [float(intervals[0,0]),float(intervals[-1,1])] == item["genuine_regional_bp0"],
              "Exact genuine regional coordinate support; no tail extrapolation")
        query_subjects = [row["subject_id"] for row in roster[16:]]
        persons = query_subjects[::2]
        _need(all(query_subjects[i] == query_subjects[i+1] for i in range(0,20,2))
              and (subjects is None or subjects == persons),"Original diploid identities must match all windows")
        subjects = persons
        prediction_ref = writer.npz("chr"+chrom+"_all_native_holds.npz",probabilities=probabilities,
            intervals_regional_bp0=intervals,intervals_absolute_bp0=intervals+20000000,
            query_columns=bank.query_column.numpy(),class_order=np.asarray(CLASSES),
            query_subject_ids=np.asarray(query_subjects),
            query_haplotypes=np.asarray([row["haplotype"] for row in roster[16:]],dtype=np.int8))
        results[chrom] = dict(covered_bp=float(spans.sum()),
            haplotype_proportions=(np.einsum("t,tqc->qc",spans,probabilities.astype(np.float64))/spans.sum()).tolist())
        products[chrom] = dict(prediction_ref=prediction_ref,**item,
            episode_metadata=episode_meta,cut_metadata=cut_meta,whole_observed_context=True)
        del bank,flags,bundle,probabilities,done,roster
        check("current_TreeLoom_chromosome_complete")
    summary_ref = writer.json("summary.json",descriptive_summary(results,subjects))
    provenance_ref = writer.json("provenance.json",dict(
        application=SCHEMA,fit_completion_ref=spec["fit_completion_ref"],checkpoint_ref=complete["checkpoint_ref"],
        observation_config_ref=CONFIG_REF,observation_manifest_ref=MANIFEST_REF,
        model_parameters=32325,trainable_tensors=49,model_class_order=list(CLASSES),
        observed_class_order=["ANA","CHG","WHG","EHG"],class_aliases_metadata_only=True,
        products=products,optimization_steps=0,query_ancestry_truth_read=False,
        old_H045_outputs_unchanged=True,ancestry_accuracy_available=False,
        domain_shift="H36_R16_Q20_empirical10Mb_pre_size_Relate_vs_H952_simulated200Mb_post_five_size_training"))
    result = dict(schema=SCHEMA,status="completed_current_C4_descriptive_application_no_accuracy",
        fit_completion_ref=spec["fit_completion_ref"],checkpoint_ref=complete["checkpoint_ref"],
        prediction_refs={chrom:products[chrom]["prediction_ref"] for chrom in CHROMOSOMES},
        summary_ref=summary_ref,provenance_ref=provenance_ref,
        chromosomes=list(CHROMOSOMES),optimization_steps=0,query_ancestry_truth_read=False,
        ancestry_accuracy_available=False,old_H045_result_reused_or_relabelled=False,
        actual_worker_wait_and_owned_cleanup_still_required=True)
    ref = writer.json("application_completed.json",result)
    _need(not writer.pending,"Every new current-model descriptive artifact must finish")
    return dict(result,completion_ref=ref)


def factory(**kwargs):
    spec = build_spec(**kwargs)
    return dict(module="arggnn.benchmarking.empirical_treeloom",function="run",
        spec=spec,python=".runtime/env/bin/python",source_refs=spec["requirements"]["source_refs"])
