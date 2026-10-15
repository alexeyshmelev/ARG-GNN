#!/usr/bin/env python3
"""Deferred numeric entrypoints for prospective1611 native adapters.

Not run by source-only selftests. Imports native dependencies only inside commands.
No scheduler/install/admission/guard/GPU-probe code is included.
"""
from __future__ import annotations
import argparse
import collections
import copy
import json
import math
import random
import sys
from pathlib import Path
from panel_adapter import (
    ROOT,HERE,ARGMIX,Panel,LABELS,sha256,load_json,write_json,read_jsonl,
    canonical_sha,validate_targets,validate_labels,argmix_request,
    native_argmix_iterator,native_pop_token_lookup,unpack_native_flat,
    ap_features,project_ap_probabilities,verify_sources,token_vocab)

def finite_config():
    config=load_json(HERE/"finite_fit_config.v1.json")
    if config["schema"]!="prospective1611_finite_native_fit_v1":
        raise ValueError("wrong finite fit schema")
    return config

def native_imports():
    verify_sources()
    # Frozen author directory is imported, never rewritten or copied over.
    sys.path.insert(0,str(ARGMIX/"src"))

def validate_rebased_map_rows(path,length_bp):
    """Metadata map-format check prevents native fallback extrapolation.

    This checks local bytes/coverage, not public source or derivation provenance.
    """
    lines=[line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    if not lines or lines[0]!=["position","COMBINED_rate(cM/Mb)","Genetic_Map(cM)"]:
        raise ValueError("rebased map requires canonical position/rate/cM header")
    rows=[]
    for fields in lines[1:]:
        if len(fields)!=3:
            raise ValueError("map row needs three fields")
        x,rate,cm=map(float,fields)
        if not all(math.isfinite(v) for v in (x,rate,cm)) or x!=int(x) or x<0 or rate<0 or cm<0:
            raise ValueError("invalid map metadata")
        rows.append((int(x),rate,cm))
    if len(rows)<2 or rows[0][0]!=0 or rows[-1][0]!=length_bp or rows[0][2]!=0:
        raise ValueError("map must cover exact [0,L] with cumulative cM rebased to zero")
    if any(b[0]<=a[0] or b[2]<a[2] for a,b in zip(rows,rows[1:])):
        raise ValueError("map positions/cumulative distance must be monotone")
    return {"rows":len(rows),"first_bp":0,"last_bp":length_bp,"cM_span":rows[-1][2]}

def read_map_binding(path):
    doc=load_json(path)
    required={"schema","chromosome","genome_build","source_url","source_sha256",
              "source_interval_bp","rebased_map_path","rebased_map_sha256","length_bp"}
    if set(doc)!=required or doc["schema"]!="prospective_chr22_GRCh37_map_binding_v1":
        raise ValueError("map binding fields/schema mismatch")
    if doc["chromosome"]!="22" or doc["genome_build"]!="GRCh37":
        raise ValueError("explicit GRCh37 chr22 map required")
    a,b=doc["source_interval_bp"]
    if not 0<=a<b or doc["length_bp"]!=b-a:
        raise ValueError("inconsistent map interval/length")
    for key in ("source_sha256","rebased_map_sha256"):
        if not isinstance(doc[key],str) or len(doc[key])!=64 or any(c not in "0123456789abcdef" for c in doc[key]):
            raise ValueError("invalid map SHA256")
    if not isinstance(doc["source_url"],str) or not doc["source_url"].startswith("https://"):
        raise ValueError("explicit HTTPS public map source URL required; origin/transform verification is a root-owned precondition")
    if sha256(doc["rebased_map_path"])!=doc["rebased_map_sha256"]:
        raise ValueError("rebased map hash mismatch")
    validate_rebased_map_rows(doc["rebased_map_path"],doc["length_bp"])
    return doc

def check_observed_mapping(ts,panel,targets):
    if set(map(int,ts.samples()))!=set(panel.by_node):
        raise ValueError("observed tree sample nodes differ from exact roster mapping")
    for nid,row in panel.by_node.items():
        if float(ts.node(nid).time)!=row["age_generations"]:
            raise ValueError("observed sample age differs from roster")
    known={}
    for row in targets:
        idx=row["tree_index"]
        if idx not in known:
            tree=ts.at_index(idx)
            known[idx]=(float(tree.interval.left),float(tree.interval.right))
        if known[idx]!=(float(row["left"]),float(row["right"])):
            raise ValueError("target tree index/interval differs from observed bank")

def extract(args):
    """Future root-run feature extraction. Labels cannot be provided here."""
    verify_sources()
    panel=Panel(load_json(args.roster))
    targets=read_jsonl(args.targets)
    validate_targets(targets,panel,parent_id=args.parent)
    # No simulation truth path exists in this interface.
    import tskit
    ts=tskit.load(args.trees)
    check_observed_mapping(ts,panel,targets)
    target_by_key={(r["haplotype_id"],float(r["left"]),float(r["right"])):r for r in targets}
    if len(target_by_key)!=len(targets):
        raise ValueError("ambiguous interval target join")
    seen=set();written=0
    with Path(args.output).open("x") as output:
        if args.method=="ap":
            for target in targets:
                feat=ap_features(ts.at_index(target["tree_index"]),panel,target["haplotype_id"])
                row={"parent_id":args.parent,"haplotype_id":target["haplotype_id"],
                     "node_id":panel.query(target["haplotype_id"])["node_id"],
                     "left":target["left"],"right":target["right"],
                     "target_row_id":target["row_id"],"features":feat}
                output.write(json.dumps(row,separators=(",",":"))+"\n")
                seen.add(target["row_id"]);written+=1
        else:
            native_imports()
            import ancpath
            lookup=native_pop_token_lookup(panel)
            for hid in sorted({r["haplotype_id"] for r in targets}):
                request=argmix_request(panel,hid)
                processor=native_argmix_iterator(ancpath,args.trees,request)
                emitted_intervals=set()
                while True:
                    payload=processor.next_flat()
                    if payload is None:
                        break
                    for raw in unpack_native_flat(payload,lookup):
                        if raw["node_id"]!=panel.query(hid)["node_id"]:
                            raise ValueError("native output query-node mapping mismatch")
                        iv=(raw["left"],raw["right"])
                        if iv in emitted_intervals:
                            raise ValueError("native duplicate unmerged interval")
                        emitted_intervals.add(iv)
                        target=target_by_key.get((hid,*iv))
                        # All observed interval rows for queried haplotypes are context;
                        # only exact common targets receive external labels later.
                        row={"parent_id":args.parent,"haplotype_id":hid,
                             "node_id":raw["node_id"],"left":iv[0],"right":iv[1],
                             "target_row_id":target["row_id"] if target else None,
                             "input_ids":raw["input_ids"],"tmrca":raw["tmrca"]}
                        output.write(json.dumps(row,separators=(",",":"))+"\n")
                        written+=1
                        if target:
                            if target["row_id"] in seen:
                                raise ValueError("target emitted twice")
                            seen.add(target["row_id"])
    expected={r["row_id"] for r in targets}
    if seen!=expected:
        raise ValueError("feature/target join incomplete; partial output must not be used")
    meta={"schema":"prospective1611_native_features_v1","complete":True,
          "method":args.method,"method_label":"AncestralPaths-derived C4 adapter" if args.method=="ap" else "ARGMix current-release count/role adapter","parent_id":args.parent,
          "target_rows":len(targets),"feature_rows":written,
          "targets_sha256":sha256(args.targets),"roster_sha256":sha256(args.roster),
          "observed_trees_sha256":sha256(args.trees),"features_sha256":sha256(args.output),
          "source_pins_sha256":sha256(HERE/"source_pins.v1.json"),
          "fit_config_sha256":sha256(HERE/"finite_fit_config.v1.json"),
          "argmix_context":"all observed intervals per requested query; labels joined later only at shared midpoint targets",
          "paper_reproduction":False}
    write_json(args.output+".metadata.json",meta)

def load_inputs(path,method,*,with_labels):
    """Bundles name explicit files; no discovery/glob of payload directories."""
    bundle=load_json(path)
    if bundle.get("schema")!="prospective1611_feature_input_bundle_v1":
        raise ValueError("wrong input bundle schema")
    rows=[];all_targets=[];parents=set();signatures=[];seen_feature_keys=set()
    for entry in bundle["parents"]:
        need={"parent_id","roster","targets","features","metadata"}
        if with_labels:
            need.update(("labels","supervision_contract"))
        if set(entry)!=need:
            raise ValueError("input entry fields mismatch")
        parent=entry["parent_id"]
        if parent in parents:
            raise ValueError("duplicate parent")
        parents.add(parent)
        panel=Panel(load_json(entry["roster"]))
        targets=read_jsonl(entry["targets"])
        validate_targets(targets,panel,parent_id=parent)
        target_index={r["row_id"]:r for r in targets}
        labels=validate_labels(read_jsonl(entry["labels"]),targets) if with_labels else {}
        meta=load_json(entry["metadata"])
        if meta.get("schema")!="prospective1611_native_features_v1" or not meta.get("complete") or meta.get("method")!=method or meta.get("parent_id")!=parent:
            raise ValueError("feature metadata mismatch/incomplete")
        for key,filename in (("features_sha256","features"),("targets_sha256","targets"),("roster_sha256","roster")):
            if sha256(entry[filename])!=meta[key]:
                raise ValueError(key+" mismatch")
        if meta["source_pins_sha256"]!=sha256(HERE/"source_pins.v1.json") or meta["fit_config_sha256"]!=sha256(HERE/"finite_fit_config.v1.json"):
            raise ValueError("feature source/config identity mismatch")
        supervision_signature={}
        if with_labels:
            from truth_adapter import validate_supervision_contract
            supervision_signature=validate_supervision_contract(load_json(entry["supervision_contract"]),parent_id=parent,
                targets_path=entry["targets"],roster_path=entry["roster"],labels_path=entry["labels"],observed_metadata=meta)
        seen=set();before=len(rows)
        for row in read_jsonl(entry["features"]):
            need_fields={"parent_id","haplotype_id","node_id","left","right","target_row_id"}
            need_fields |= {"features"} if method=="ap" else {"input_ids","tmrca"}
            if set(row)!=need_fields or row["parent_id"]!=parent:
                raise ValueError("feature row fields/parent mismatch")
            q=panel.query(row["haplotype_id"])
            if row["node_id"]!=q["node_id"]:
                raise ValueError("feature node/identity mismatch")
            l,r=float(row["left"]),float(row["right"])
            if not math.isfinite(l+r) or not 0<=l<r:
                raise ValueError("feature interval invalid")
            key=(parent,row["node_id"],l,r)
            if key in seen_feature_keys:
                raise ValueError("duplicate context row")
            seen_feature_keys.add(key)
            tid=row["target_row_id"]
            if tid is not None:
                if tid not in target_index or tid in seen:
                    raise ValueError("duplicate/unexpected target output ID")
                target=target_index[tid]
                if (target["haplotype_id"],float(target["left"]),float(target["right"]))!=(row["haplotype_id"],l,r):
                    raise ValueError("feature interval/identity does not match target")
                seen.add(tid)
            if method=="ap":
                a=row["features"]
                if tid is None or len(a)!=5 or any(len(block)!=9 for block in a) or any(not math.isfinite(float(v)) for block in a for v in block):
                    raise ValueError("invalid 5x9 AP feature")
            else:
                if len(row["input_ids"])!=16 or len(row["tmrca"])!=16 or any(len(block)!=16 for block in row["tmrca"]):
                    raise ValueError("invalid 16-node ARGMix feature shape")
                if any(isinstance(v,bool) or not isinstance(v,int) or not 0<=v<len(token_vocab()) for v in row["input_ids"]):
                    raise ValueError("invalid ARGMix token")
                if any(isinstance(v,bool) or not isinstance(v,int) or not -32768<=v<=32767 for block in row["tmrca"] for v in block):
                    raise ValueError("invalid native int16 TMRCA matrix")
                if row["input_ids"][0]!=token_vocab()["<SAMPLE>"]:
                    raise ValueError("ARGMix focal token is not <SAMPLE>")
            row["label"]=labels[tid] if with_labels and tid is not None else None
            rows.append(row)
        if seen!=set(target_index) or len(rows)-before!=meta["feature_rows"] or len(seen)!=meta["target_rows"]:
            raise ValueError("features do not cover target support exactly")
        all_targets.extend(targets)
        signatures.append({"parent_id":parent,"targets_sha256":meta["targets_sha256"],
                           "roster_sha256":meta["roster_sha256"],
                           "observed_trees_sha256":meta["observed_trees_sha256"],**supervision_signature})
    if not rows or len(all_targets)!=len({r["row_id"] for r in all_targets}):
        raise ValueError("empty input or cross-parent target ID collision")
    return rows,all_targets,signatures

def native_chunks(rows,map_binding,config):
    import torch
    from joint_interval_model import build_joint_input_chunks
    converted=[]
    for r in rows:
        converted.append({"shard_id":r["parent_id"],"node_id":r["node_id"],
                          "tree_start":r["left"],"tree_end":r["right"],
                          "input_ids":torch.tensor(r["input_ids"],dtype=torch.long),
                          "tmrca":torch.tensor(r["tmrca"],dtype=torch.int16),
                          "label":r["label"]})
    j=config["argmix"]["joint_model"]
    chunks=build_joint_input_chunks(converted,recomb_map_path=map_binding["rebased_map_path"],
                                    chunk_rows=j["chunk_rows"],chunk_stride=j["chunk_stride"],
                                    pad_id=token_vocab()["<PAD>"])
    return chunks,converted

def validate_fit_support(train_targets,val_targets):
    # Published-counts phase, not an implicit small diagnostic fit.
    tc=collections.Counter(r["parent_id"] for r in train_targets)
    vc=collections.Counter(r["parent_id"] for r in val_targets)
    expected={"published_recon_train_%03d"%i:100000 for i in range(5)}
    if tc!=expected or vc!={"published_recon_val_000":100000}:
        raise ValueError("fit requires exactly the registered five TRAIN and one VAL target sets")
    if set(tc)&set(vc) or {r["row_id"] for r in train_targets}&{r["row_id"] for r in val_targets}:
        raise ValueError("training/validation leakage")

def fit(args):
    verify_sources()
    config=finite_config()
    train,tr_targets,tr_sig=load_inputs(args.train_inputs,args.method,with_labels=True)
    val,va_targets,va_sig=load_inputs(args.val_inputs,args.method,with_labels=True)
    validate_fit_support(tr_targets,va_targets)
    out=Path(args.output_dir);out.mkdir(exist_ok=False)
    if args.method=="ap":
        import numpy as np
        import tensorflow as tf
        ap=config["ap"]
        random.seed(ap["seed"]);np.random.seed(ap["seed"])
        tf.keras.utils.set_random_seed(ap["seed"])
        x=np.asarray([r["features"] for r in train],dtype=np.float32)
        vx=np.asarray([r["features"] for r in val],dtype=np.float32)
        y=tf.keras.utils.to_categorical([r["label"] for r in train],6)
        vy=tf.keras.utils.to_categorical([r["label"] for r in val],6)
        with tf.device(args.device):
            # Exact released classifier architecture; no four-head substitution.
            model=tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(5,9)),
                tf.keras.layers.Dense(512,activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(6,activation=tf.nn.softmax)])
            model.compile(loss="categorical_crossentropy",
                          optimizer=tf.keras.optimizers.Adam(**ap["adam"]),
                          metrics=["accuracy"])
            callback=tf.keras.callbacks.ModelCheckpoint(str(out/"best_model.h5"),
                        monitor="val_loss",mode="min",save_best_only=True)
            history=model.fit(x,y,validation_data=(vx,vy),epochs=5,
                              batch_size=32,shuffle=True,callbacks=[callback],verbose=2)
        losses=list(map(float,history.history["val_loss"]))
        best=min(range(len(losses)),key=losses.__getitem__)
        summary={"best_epoch":best+1,"best_val_loss":losses[best],"epochs_ran":len(losses)}
    else:
        native_imports()
        import numpy as np
        import torch
        from torch.utils.data import DataLoader,TensorDataset
        from joint_interval_model import build_joint_model_from_config,save_joint_model_checkpoint
        from train_joint_model import _masked_joint_loss
        a=config["argmix"];seed=config["argmix_seed"]
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        binding=read_map_binding(args.map_binding)
        for r in train+val:
            if r["right"]>binding["length_bp"]:
                raise ValueError("observed feature interval outside frozen map length")
        tr_chunks,_=native_chunks(train,binding,config)
        va_chunks,_=native_chunks(val,binding,config)
        def loader(chunks,shuffle):
            # Context participates in GRU, but a wholly unlabeled chunk has no loss.
            keep=(chunks["labels"]!=-100).any(dim=1)
            fields=("input_ids","tmrca","labels","row_mask","cm_features")
            return DataLoader(TensorDataset(*(chunks[k][keep] for k in fields)),
                              batch_size=a["joint_model"]["train"]["batch_size"],
                              shuffle=shuffle,num_workers=0)
        tr_loader=loader(tr_chunks,True);va_loader=loader(va_chunks,False)
        device=torch.device(args.device)
        model=build_joint_model_from_config(a,device=device)
        t=a["joint_model"]["train"]
        optimizer=torch.optim.AdamW(model.parameters(),lr=t["learning_rate"],weight_decay=t["weight_decay"])
        best_loss=math.inf;best_state=None;best_epoch=0;left=t["early_stop_patience"];history=[]
        for epoch in range(1,t["num_epochs"]+1):
            epoch_losses=[]
            for training,dl in ((True,tr_loader),(False,va_loader)):
                model.train(training);total=0.0;batches=0
                with torch.set_grad_enabled(training):
                    for batch in dl:
                        ids,tmrca,labels,mask,cm=(v.to(device) for v in batch)
                        if training:
                            optimizer.zero_grad()
                        _,logits=model.forward_chunks(ids,tmrca,mask,cm,pad_id=0)
                        loss=_masked_joint_loss(logits,labels,mask,cm,
                               label_smoothing=t["label_smoothing"],
                               loss_weighting=t["loss_weighting"])
                        if not torch.isfinite(loss):
                            raise ValueError("non-finite native training loss")
                        if training:
                            loss.backward();optimizer.step()
                        total+=float(loss.item());batches+=1
                if not batches:
                    raise ValueError("no labeled chunks")
                epoch_losses.append(total/batches)
            train_loss,val_loss=epoch_losses
            history.append({"epoch":epoch,"train_loss":train_loss,"val_loss":val_loss})
            if val_loss<best_loss:
                best_loss=val_loss;best_epoch=epoch
                best_state=copy.deepcopy(model.state_dict());left=t["early_stop_patience"]
            else:
                left-=1
                if left<=0:
                    break
        if best_state is None:
            raise ValueError("no finite best checkpoint")
        model.load_state_dict(best_state,strict=True)
        save_joint_model_checkpoint(out/"best_model.pt",model=model,config=a,
                                    epoch=best_epoch,val_loss=best_loss,
                                    run_name="prospective1611_count_role_adapter_v1")
        summary={"best_epoch":best_epoch,"best_val_loss":best_loss,
                 "epochs_ran":len(history),"history":history,
                 "native_overlap_note":"1024-row chunks, stride512; overlapping target occurrences inherit native training weights",
                 "map_binding_sha256":sha256(args.map_binding),
                 "map_lineage_verified_by_adapter":False}
    summary.update({"schema":"prospective1611_native_fit_summary_v1","method":args.method,
                    "method_label":"AncestralPaths-derived C4 adapter" if args.method=="ap" else "ARGMix current-release count/role adapter",
                    "train_shared_signatures":tr_sig,"val_shared_signatures":va_sig,
                    "logical_train_targets":len(tr_targets),"logical_val_targets":len(va_targets),
                    "fit_config_sha256":sha256(HERE/"finite_fit_config.v1.json"),
                    "paper_reproduction":False,
                    "checkpoint_sha256":sha256(out/("best_model.h5" if args.method=="ap" else "best_model.pt"))})
    write_json(out/"summary.json",summary)

def predict(args):
    verify_sources();config=finite_config()
    fit_summary=load_json(args.fit_summary)
    if fit_summary.get("method")!=args.method or fit_summary.get("fit_config_sha256")!=sha256(HERE/"finite_fit_config.v1.json") or fit_summary.get("checkpoint_sha256")!=sha256(args.checkpoint):
        raise ValueError("prediction checkpoint/fit/config identity mismatch")
    if args.method=="argmix" and fit_summary.get("map_binding_sha256")!=sha256(args.map_binding):
        raise ValueError("prediction map binding differs from training")
    rows,targets,sig=load_inputs(args.inputs,args.method,with_labels=False)
    probabilities={}
    if args.method=="ap":
        import numpy as np
        import tensorflow as tf
        with tf.device(args.device):
            model=tf.keras.models.load_model(args.checkpoint)
            # Batch output order joins by target row ID, not caller array offsets.
            for start in range(0,len(rows),32):
                group=rows[start:start+32]
                pred=model.predict(np.asarray([r["features"] for r in group],dtype=np.float32),verbose=0)
                for row,p in zip(group,pred):
                    probabilities[row["target_row_id"]]=project_ap_probabilities(p.tolist())
    else:
        native_imports()
        import numpy as np
        import torch
        from joint_interval_model import (load_joint_model_checkpoint,_build_chunk_index_specs)
        from config_loader import get_class_labels
        binding=read_map_binding(args.map_binding)
        if any(r["right"]>binding["length_bp"] for r in rows):
            raise ValueError("prediction feature interval outside frozen map length")
        device=torch.device(args.device)
        # Native loader returns (model, checkpoint dictionary); keep strict format.
        loaded=load_joint_model_checkpoint(args.checkpoint,device=device)
        model=loaded[0]
        checkpoint=loaded[1]
        if canonical_sha(checkpoint["config"])!=canonical_sha(config["argmix"]):
            raise ValueError("checkpoint configuration differs from finite native fit")
        if tuple(get_class_labels(checkpoint["config"])[i] for i in range(4))!=LABELS:
            raise ValueError("checkpoint founder-label order differs")
        chunks,converted=native_chunks(rows,binding,config)
        j=config["argmix"]["joint_model"]
        specs=_build_chunk_index_specs(converted,chunk_rows=j["chunk_rows"],chunk_stride=j["chunk_stride"])
        sums=np.zeros((len(rows),4),dtype=np.float64);counts=np.zeros(len(rows),dtype=np.int64)
        model.eval()
        with torch.no_grad():
            for start in range(0,len(specs),16):
                sl=slice(start,start+16)
                _,logits=model.forward_chunks(chunks["input_ids"][sl].to(device),
                    chunks["tmrca"][sl].to(device),chunks["row_mask"][sl].to(device),
                    chunks["cm_features"][sl].to(device),pad_id=0)
                p=torch.softmax(logits,dim=-1).cpu().numpy()
                for b,idxs in enumerate(specs[sl]):
                    sums[idxs]+=p[b,:len(idxs)];counts[idxs]+=1
        for i,row in enumerate(rows):
            tid=row["target_row_id"]
            if tid is not None:
                if counts[i]<1:
                    raise ValueError("native prediction missed target")
                probabilities[tid]=(sums[i]/counts[i]).tolist()
    if set(probabilities)!={r["row_id"] for r in targets}:
        raise ValueError("prediction target support differs")
    with Path(args.output).open("x") as out:
        for t in targets:
            p=probabilities[t["row_id"]]
            if len(p)!=4 or not all(math.isfinite(v) and v>=0 for v in p) or not math.isclose(sum(p),1.0,abs_tol=1e-5):
                raise ValueError("invalid four-founder probability")
            out.write(json.dumps({"row_id":t["row_id"],"labels":LABELS,"probabilities":p},
                                 separators=(",",":"))+"\n")
    write_json(args.output+".metadata.json",{"schema":"prospective1611_native_predictions_v1",
               "method":args.method,"method_label":"AncestralPaths-derived C4 adapter" if args.method=="ap" else "ARGMix current-release count/role adapter","shared_signatures":sig,"target_rows":len(targets),
               "predictions_sha256":sha256(args.output),"checkpoint_sha256":sha256(args.checkpoint),
               "paper_reproduction":False})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    e=sub.add_parser("extract")
    e.add_argument("--method",choices=("argmix","ap"),required=True)
    for name in ("roster","targets","trees","parent","output"):
        e.add_argument("--"+name,required=True)
    f=sub.add_parser("fit")
    f.add_argument("--method",choices=("argmix","ap"),required=True)
    for name in ("train-inputs","val-inputs","output-dir","device"):
        f.add_argument("--"+name,required=True)
    f.add_argument("--map-binding")
    d=sub.add_parser("predict")
    d.add_argument("--method",choices=("argmix","ap"),required=True)
    for name in ("inputs","checkpoint","fit-summary","output","device"):
        d.add_argument("--"+name,required=True)
    d.add_argument("--map-binding")
    args=p.parse_args()
    if args.command in ("fit","predict") and args.method=="argmix" and not args.map_binding:
        p.error("ARGMix requires --map-binding")
    {"extract":extract,"fit":fit,"predict":predict}[args.command](args)

if __name__=="__main__":
    main()
