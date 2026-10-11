"""H038 unchanged-capacity PyG model with a declared concordance ablation."""
from dataclasses import replace
from .mutation_routed_pyg_gnn_v1 import MutationRoutedPyGGNNV1
ARMS=("aligned","zero_concordance")


def prepare_control_bank_v1(bank,control):
    if control not in ARMS:raise ValueError("Registered aligned/zero_concordance arm required")
    if control=="aligned":return bank
    support=bank.branch_support_true.clone()
    support[:,:3]=0
    return replace(bank,branch_support_sham=support)


class MutationRoutedFullPanelPyGGNNV1(MutationRoutedPyGGNNV1):
    """Same38statekeys/29764parameters; PyG message passing remains actual."""
    def forward(self,bank,*,control="aligned"):
        if control not in ARMS:raise ValueError("Registered full-panel arm required")
        return super().forward(bank,control="aligned" if control=="aligned" else "shuffled")


    def forward_train_holds(self,bank,*,control="aligned"):
        """Complete tree/GRU computation with one query readout per native hold."""
        if control not in ARMS:raise ValueError("Registered full-panel arm required")
        holds=super().forward_holds(bank,control="aligned" if control=="aligned" else "shuffled")
        return holds.index_select(1,bank.query_column)
