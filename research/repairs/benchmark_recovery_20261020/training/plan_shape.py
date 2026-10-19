"""Read small existing observed plans only; never load tensors or truth."""
import argparse
import json
import math
from pathlib import Path


def estimate(plan):
    holds = int(plan["native_holds"])
    return dict(parent_id=plan["parent_id"], native_holds=holds,
        original_sites=plan["original_sites"],
        original_tree_blocks=math.ceil(holds/16),
        tree_blocks_at_128=math.ceil(holds/128), tree_blocks_at_256=math.ceil(holds/256),
        complete_context_bytes=holds*1600*32*4,
        complete_context_VJP_bytes=holds*1600*32*4,
        decoder_tile_input_bytes={str(tile):holds*tile*32*4 for tile in (8,32,64)},
        original_decoder_tiles=200,
        maximum_nonempty_decoder_tiles_at_32=math.ceil(1600/32)-(600//32),
        caution="Tensor dimensions only; GRU reserve, autograd and CUDA peak need actual qualification")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plans_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps([estimate(json.loads(path.read_text()))
        for path in sorted(args.plans_directory.glob("*.json"))], sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
