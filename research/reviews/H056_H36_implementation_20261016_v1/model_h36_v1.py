"""Explicit H36/R16/Q20 count interface; all frozen H045 learned operations reused.

No checkpoint reader, optimizer, launcher, guard or authority is provided here.
The old40/24 and952/686 paths are retained solely for root compatibility checks.
"""
import torch
from arggnn.ordered_clade_episode_gnn_v1 import (OrderedCladeEpisodeGNNV1,
    RankEightCladeEpisodeAdapterV1,build_clade_episode_arrays_v1,OrderedCladeEpisodesV1)
from arggnn.anonymous_clade_flag_control_v1 import RankEightCladeFlagAdapterV1
from arggnn.mutation_routed_full_panel_pyg_model_v1 import MutationRoutedFullPanelPyGGNNV1
from research.reviews.H018_H045_empirical_application_source_v1.variable_panel_model_v1 import VariablePanelEpisodeInterfaceV1
from research.reviews.H056_H36_implementation_20261016_v1.contract_v1 import H,Q,require

class H36EpisodeInterfaceV1(OrderedCladeEpisodeGNNV1):
    def __init__(self,*,expected_haplotypes=H,expected_queries=Q):
        pair=(expected_haplotypes,expected_queries)
        require(pair in ((H,Q),(40,24),(952,686)),"Only explicit36/20,40/24,952/686 interfaces")
        if pair==(H,Q):
            MutationRoutedFullPanelPyGGNNV1.__init__(self,hidden_dim=32,
                expected_haplotypes=H,site_chunk_size=256,checkpoint_sites=True)
            self.expected_queries=Q
            self.clade_coupling=RankEightCladeFlagAdapterV1()
            self.episode_coupling=RankEightCladeEpisodeAdapterV1()
            self.arm="foreign_episode";self.float()
        else:
            VariablePanelEpisodeInterfaceV1.__init__(self,
                expected_haplotypes=expected_haplotypes,expected_queries=expected_queries)
        require(sum(p.numel() for p in self.parameters())==31812,"Exact original31812 learned values")
    # This frozen method calls the original variable-panel full-history GRU/head;
    # self.encode_trees is inherited unchanged from the original H045 class.
    forward_holds=VariablePanelEpisodeInterfaceV1.forward_holds

def episode_inputs_v1(bank,*,known_flags,max_payload_bytes):
    pair=(bank.sample_index.shape[1],bank.query_column.numel())
    require(pair in ((H,Q),(40,24),(952,686)) and bank.parent.device.type=="cpu"
        and type(max_payload_bytes)is int and max_payload_bytes>0,"Explicit complete CPU panel/episode bound")
    arrays,meta=build_clade_episode_arrays_v1(bank.parent.detach().numpy(),
        bank.sample_index.detach().numpy(),bank.spans.detach().numpy(),max_payload_bytes=max_payload_bytes)
    episodes=OrderedCladeEpisodesV1.from_verified_arrays(arrays,meta)
    require(torch.equal(episodes.flags.node_flags,known_flags.node_flags),
        "Original episode flags equal prepared aligned flags")
    return episodes,meta
