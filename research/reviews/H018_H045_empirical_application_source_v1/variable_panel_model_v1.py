"""Disclosed count-only H045 application interface; original learned arithmetic.

Qualified count pairs: empirical H40/Q24 and original H952/Q686. No padding,
new learned layer, source edit, optimizer or weight transformation is used.
"""
import torch
from arggnn.ordered_clade_episode_gnn_v1 import (
    OrderedCladeEpisodeGNNV1,RankEightCladeEpisodeAdapterV1,
    build_clade_episode_arrays_v1,OrderedCladeEpisodesV1)
from research.reviews.H018_variable_panel_adapter_source_v1.flag_interface_proposal import VariablePanelFlagInterfaceV1

class VariablePanelEpisodeInterfaceV1(OrderedCladeEpisodeGNNV1):
    def __init__(self,*,expected_haplotypes=40,expected_queries=24):
        if (expected_haplotypes,expected_queries) not in ((40,24),(952,686)):
            raise ValueError("Only explicitly qualified40/Q24 or952/Q686 count interfaces")
        if expected_haplotypes==40:
            VariablePanelFlagInterfaceV1.__init__(self,hidden_dim=32,
                expected_haplotypes=40,expected_queries=24,site_chunk_size=256,checkpoint_sites=True)
            self.episode_coupling=RankEightCladeEpisodeAdapterV1()
            self.arm="foreign_episode";self.float()
        else:
            OrderedCladeEpisodeGNNV1.__init__(self,hidden_dim=32,expected_haplotypes=952,
                site_chunk_size=256,checkpoint_sites=True,arm="foreign_episode")
            self.expected_queries=686
        if sum(p.numel() for p in self.parameters())!=31812:
            raise ValueError("Exact original31812 learned parameters required")

    def forward_holds(self,bank,*,episodes,control="aligned"):
        # Existing H018 count/device gates and identical original whole-span
        # geometry/GRU/head; self.encode_trees is the unchanged H045 method.
        return VariablePanelFlagInterfaceV1.forward_holds(self,bank,flags=episodes,control=control)

def episode_inputs_v1(bank,*,known_flags,max_payload_bytes=1000000):
    h=bank.sample_index.shape[1];q=bank.query_column.numel()
    if (h,q) not in ((40,24),(952,686)) or bank.parent.device.type!="cpu":
        raise ValueError("Qualified complete CPU bank, no padding or panel selection")
    arrays,meta=build_clade_episode_arrays_v1(bank.parent.detach().numpy(),
        bank.sample_index.detach().numpy(),bank.spans.detach().numpy(),max_payload_bytes=max_payload_bytes)
    episodes=OrderedCladeEpisodesV1.from_verified_arrays(arrays,meta)
    if not torch.equal(episodes.flags.node_flags,known_flags.node_flags):
        raise ValueError("Model episode flags differ from original prepared aligned flags")
    return episodes,meta
