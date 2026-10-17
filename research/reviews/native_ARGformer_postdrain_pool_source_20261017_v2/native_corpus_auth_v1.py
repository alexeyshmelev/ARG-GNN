from pathlib import Path
from . import native_common_v1 as c
AUTH={"path":"research/reviews/American_C3_native_MLM_amendment_v12/corpus_auth_v2.py","sha256":"2f02c99d49502493abab82e5b6c08c8abc782538d7d821a71cd5116f62b0cd98"}
def authenticate_inputs(prep,prepared_dir,counts):
    original=c.load(AUTH,"_native_parent_original_streaming_corpus_auth")
    entries={}
    for key in ("mlm/train","mlm/val"):
        row=counts["datasets"][key];record=prep["native_dataset_refs"][key]
        c.require(row["dataset_ref"]==record,"Original actual count/source ref")
        entries[key]=original.authenticate_file(record,prepared_dir/key/("raw_"+key.split("/")[1]+"_sequences_and_vocab.pkl"),row["raw_pickle_bytes"])
    entries["vocab"]=original.authenticate_file(prep["vocab_ref"],prepared_dir/"vocab.pkl",64*counts["vocabulary_nodes"]+16777216)
    result=dict(schema="native_full_parent_original_stream_auth_v1",inputs=entries,all_consumed_MLM_raw_files_authenticated=True,standalone_vocab_authenticated=True,corpus_copied=False,decoded=False,verification_count_per_input=1,original_authentication_source_ref=AUTH,legacy_preparation_not_modified=True)
    original.confirm_unchanged(result,prepared_dir)
    return result
def confirm_unchanged(result,prepared_dir):
    return c.load(AUTH,"_native_parent_original_confirmation").confirm_unchanged(result,prepared_dir)
