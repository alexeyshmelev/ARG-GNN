"""Paired original common four-path score after both methods seal all5parents."""
import argparse
import json
from pathlib import Path
from arggnn.benchmarking.paper_ancestralpaths_four.workflow import score
from arggnn.benchmarking.paper_ancestralpaths_four import CONDITION
from .dataset import PLAN_REF
from .common import BASE,RESULTS,MODEL_ID,freeze,ref,read,write,check_ram,check_disk,require


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cnn-predictions',required=True,help='JSON mapping of all5parent IDs to authenticated CNN4 prediction product refs')
    args=parser.parse_args();_,p,frozen_ref=freeze()
    targets=read(p['score_metadata_ref']);cnn=json.loads(Path(args.cnn_predictions).read_text())
    require(set(cnn)==set(targets),'Complete five-parent CNN4 mapping')
    predictions={MODEL_ID:{parent:ref(RESULTS/'predictions'/parent/'product.json') for parent in targets},CONDITION:cnn}
    def guard(additional):check_ram();check_disk(additional)
    result=score(PLAN_REF,predictions,targets,[MODEL_ID,CONDITION],str(RESULTS/'paired_score'),guard=guard,execution_check=check_ram,write_headroom_bytes=4*2**20,max_truth_uncompressed_bytes=16*2**30)
    write(RESULTS/'paired_score_disclosure.json',dict(score_ref=result,freeze_ref=frozen_ref,
       comparison='Matched canonical non-BAA target rows and four direct output classes; model-specific input representations differ.',
       joint_observed_context='Full native trees and genotype context include BAA query leaves with unknown ancestry. No BAA supervision or focal BAA outputs.',
       CNN_observed_context='Original CNN4 recomputed five-by-eight historical four-path features.',probability_merging=False))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
