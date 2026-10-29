"""Prepare a25GiB cache admission proposal from current closed-writer accounting.

This helper never authorizes a cache or writes the root admission path. Root
reviews the immutable ledger and explicitly publishes root_confirmed=true.
"""
import argparse
import json
from pathlib import Path
import time
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue import successor_v2_support as support


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--prepare-proposal',action='store_true')
    args=parser.parse_args()
    config_ref=r.ref(args.config);config=r.read(config_ref)
    plan=dict(r.read(config['successor_template_ref']),handoff_config_ref=config_ref)
    accounting=support.commitment(plan,argmix_override_bytes=25*2**30)
    if not args.prepare_proposal:
        print(json.dumps(dict(production_reserved_bytes=25*2**30,accounting=accounting,root_admission_written=False),sort_keys=True));return
    protocol_ref=r.ref('research/joint_evaluation_20261029/argmix/protocol_v1.json')
    directory=Path(__file__).resolve().parent/'argmix_upgrade_proposals'/str(time.time_ns())
    directory.mkdir(parents=True,exist_ok=False)
    ledger=dict(schema='argmix_production_actual_ledger_v1',created_unix=time.time(),admitted=accounting['admitted'],
                successor_config_ref=config_ref,cache_scope='experiments/cache/joint_argmix_20261029',cache_max_bytes=24*2**30,
                combined_reserved_bytes=accounting['total_bytes'],accounting=accounting,
                root_authorization_performed=False)
    ledger_ref=r.save(directory/'ledger.json',ledger)
    proposal=dict(schema='joint_argmix_cache_admission_v1',admitted=accounting['admitted'],root_confirmed=False,
                  protocol_ref=protocol_ref,prior_allocation_ref=config['argmix_joint_allocation_ref'],successor_config_ref=config_ref,
                  cache_scope='experiments/cache/joint_argmix_20261029',results_scope='experiments/results/joint_argmix_20261029',
                  cache_max_bytes=24*2**30,results_max_bytes=2**30,owned_RAM_max_bytes=64*2**30,
                  combined_reserved_bytes=accounting['total_bytes'],owner_ceiling_bytes=1_200_000_000_000,
                  actual_ledger_receipt_ref=ledger_ref)
    proposal_ref=r.save(directory/'admission_proposal.json',proposal)
    print(json.dumps(dict(proposal_ref=proposal_ref,ledger_ref=ledger_ref,admitted=accounting['admitted'],root_admission_written=False)))


if __name__=='__main__':main()
