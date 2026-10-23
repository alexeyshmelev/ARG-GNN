"""Exact plaintext copies of the three pinned native author path functions."""
import random
import tskit
SPECIAL_TOKENS={"[PAD]":0,"[MASK]":1,"[CLS]":2,"[SEP]":3}
SPECIAL_OFFSET=4
RANDOM_SEED=42

def get_path_key(tree, leaf_node_id):
    path_nodes = []
    u = int(leaf_node_id)
    while u != tskit.NULL:
        path_nodes.append(u)
        u = tree.parent(u)
    return tuple(path_nodes)

def build_vocab_from_sequences(sequences):
    """
    Build vocabularies from the observed raw tokens in the sequences.
    Sequences contain only node IDs.
    Assign contiguous indices to unique node tokens (after reserved special tokens).
    """
    node_set = set()
    for seq in sequences:
        for token in seq:
            node_set.add(int(token))
    # Sort tokens to ensure reproducibility
    sorted_nodes = sorted(node_set)

    # Assign node tokens
    current_index = SPECIAL_OFFSET
    node_id_tokens = {}
    for n in sorted_nodes:
        key = f"node_{n}"
        node_id_tokens[key] = current_index
        current_index += 1

    vocab = {
        "special": SPECIAL_TOKENS,
        "node_id_tokens": node_id_tokens,
    }
    return vocab

def split_unique_sequences(sequences, train_frac=0.9):
    """
    Split the unique sequences (already deduplicated) into train and validation.
    """
    random.seed(RANDOM_SEED)
    random.shuffle(sequences)
    n = len(sequences)
    train_end = int(train_frac * n)
    train_seq_local = sequences[:train_end]
    val_seq_local = sequences[train_end:]
    return train_seq_local, val_seq_local
