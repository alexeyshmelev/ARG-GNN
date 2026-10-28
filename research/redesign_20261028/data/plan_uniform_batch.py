"""Print a deterministic target-draw plan; opens no scientific input arrays."""
import argparse
import json
import numpy as np
from research.redesign_20261028.data.window_data import uniform_target_draws


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--count',type=int,default=32)
    parser.add_argument('--seed',type=int,default=20261028)
    parser.add_argument('--parents',choices=['train000_001','all_train'],default='train000_001')
    args=parser.parse_args()
    if not 1<=args.count<=128:raise ValueError('Bounded plan: 1 through 128 target draws')
    sizes={f'ancestralpaths_european_train_{i:03d}':n for i,n in enumerate([16667,16667,16666])
           if i<2 or args.parents=='all_train'}
    draws=uniform_target_draws(sizes,args.count,np.random.default_rng(args.seed))
    print(json.dumps(dict(seed=args.seed,parent_sizes=sizes,draws=draws,
        target_probability=1/sum(sizes.values()),
        support='two TRAIN parents only' if args.parents=='train000_001' else 'exact full 50000 TRAIN registry',
        prospective_internal_development_parent='ancestralpaths_european_train_002' if args.parents=='train000_001' else None,
        original_registered_split_unchanged=True,scientific_data_opened=False),indent=2))

if __name__=='__main__':main()
