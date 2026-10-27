"""New European native-six condition with all-panel observed cohort inputs.

Cohort membership is public sampling metadata, not local-ancestry truth. All
952 observed samples, including ancient query/self/ancestor occurrences, carry
these inputs through both original shared tree sweeps. The original four pure
reference labels, unknown-query flag, age and donor frequencies remain separate.
No checkpoint reader, target reader, launcher or historical module mutation.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model_binding_v1 import ShortDevelopmentTreeLoomV1

CONDITION = 'AP_short10Mb_development_native6_observed_cohort8_v1'
PANEL_BINDING = 'AP_short10Mb_development_native6'
CLASSES = ('Ana_nonBAA','CHG_nonBAA','WHG','EHG','Ana_viaBAA','CHG_viaBAA')
COHORTS = ('Bronze_Age','BAA','Yam','Neo','WHG','EHG','Ana','CHG')
PARAMETERS = 32647  # Original32391 plus32x8 new cohort input weights.


def need(ok, message):
    if not ok:
        raise ValueError(message)


class ShortDevelopmentCohortTreeLoom(ShortDevelopmentTreeLoomV1):
    def __init__(self):
        super().__init__(condition=PANEL_BINDING)
        # condition remains the inherited physical panel binding only. Every
        # successor artifact must carry the distinct scientific_condition.
        self.scientific_condition = CONDITION
        self.output_classes = CLASSES
        old = self.leaf_projection[0]
        need(type(old) is nn.Linear and old.in_features == 38 and old.out_features == 32,
             'Exact original leaf projection contract')
        widened = nn.Linear(46, 32)
        with torch.no_grad():
            widened.weight[:, :38].copy_(old.weight)
            widened.weight[:, 38:].zero_()
            widened.bias.copy_(old.bias)
        self.leaf_projection[0] = widened
        need(sum(p.numel() for p in self.parameters()) == PARAMETERS and len(list(self.parameters())) == 49,
             'Original49 parameter tensors with256 additional cohort weights')

    def base_from_sites(self, bank, site_leaf, observed_cohort):
        t,h = bank.sample_index.shape
        n,d = len(bank.parent),32
        need(observed_cohort.shape == (h,8) and observed_cohort.dtype == torch.float32
             and observed_cohort.device == site_leaf.device and not observed_cohort.requires_grad,
             'Complete immutable all-panel cohort inputs on the execution device')
        labels=bank.reference_class
        onehot=F.one_hot(labels.clamp_min(0),4).to(site_leaf.dtype)*(labels>=0)[:,None]
        unknown=(labels<0).to(site_leaf.dtype)[:,None]
        age=(bank.sample_age.log1p()/math.log(10001.))[:,None]
        observed=torch.cat((onehot,unknown,age,observed_cohort),-1)[None].expand(t,h,14)
        leaf=self.leaf_projection(torch.cat((site_leaf,observed),-1))
        sampled=bank.node_time.new_zeros(n).index_fill(0,bank.sample_index.flatten(),1.)
        node=torch.stack((bank.node_time.log1p()/math.log(10001.),(bank.parent<0).to(sampled.dtype),sampled),-1)
        base=self.node_projection(node).index_add(0,bank.sample_index.flatten(),leaf.reshape(-1,d))
        raw=bank.branch_support_true
        support=torch.cat((raw[:,:3],raw[:,3:4].log1p()/math.log(100001.),raw[:,4:5]),-1)
        gate_input=torch.cat((bank.edge_geometry,support),-1)
        up_gate=self.up_gate(gate_input).sigmoid()
        down_gate=self.down_gate(gate_input).sigmoid()
        degree=torch.bincount(bank.parent[bank.parent>=0],minlength=n).to(base.dtype).clamp_min(1)
        return base,up_gate,down_gate,degree

    def prepare_tree_base_v1(self, bank, *, control='aligned'):
        raise ValueError('Use the explicit cohort-aware executor; the legacy API has no cohort argument')

    def contract(self):
        return dict(scientific_condition=CONDITION,physical_panel_binding=PANEL_BINDING,
            classes=list(CLASSES),observed_cohort_columns=list(COHORTS),haplotypes=952,queries=686,
            requested_length_bp=10000000,development_model_selection_only=True,parameters=PARAMETERS,trainable_tensors=49,
            all_sampled_node_occurrences_receive_cohort=True,ancient_query_self_included=True,
            present_day_encoding='All eight ancient-cohort channels zero; age0 and original unknown flag retained.',
            non_sampled_ancestor_cohort='No ancestry/cohort label invented. Original tree message passing propagates observed sampled metadata.',
            pure_reference_classes=4,donor_frequency_axes=8,reference_labels_unchanged=True,
            fresh_initialization='Original seed101 parameters, with256 new cohort weights initialized zero. No trained checkpoint loaded.',
            change='Cohort metadata is concatenated before the original leaf projection tanh in both shared tree rounds.',
            equivalent_to_trained_original=False,accuracy_preservation_established=False)


def build_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        return ShortDevelopmentCohortTreeLoom().float()
