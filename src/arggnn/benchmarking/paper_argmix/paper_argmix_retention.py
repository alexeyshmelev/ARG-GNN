"""Root-invoked release of only declared new scratch after verified success.

This operation is never run automatically from a scientific producer. The
campaign scheduler must explicitly invoke the release-scratch stage. Historical
sources, observations, truth, logs, real waits and manifests are never targets.
"""
from __future__ import annotations

import gzip
import json
import shutil
from . import paper_argmix_generator as g


def release(config_path,parent_id,parent_output,native_work,inferred_output,output,*,guard):
    config,parent,observed,manifest,rows=g.observed_assets(config_path,parent_id,parent_output)
    campaign=g.confined(config["campaign_root"])
    native_work=g.confined(native_work)
    inferred_output=g.confined(inferred_output)
    g.need(native_work==campaign/parent_id/"relate_work" and
           inferred_output==campaign/parent_id/"inferred" and
           g.confined(parent_output)==campaign/parent_id/"generated" and
           g.confined(output)==campaign/parent_id/"scratch_release",
           "Only this newly declared campaign parent may release scratch")
    bridge=g.metadata(inferred_output/"completed.json")
    commands=g.metadata(native_work/"commands.json")
    g.need(bridge["parent_id"]==parent_id and bridge["config_ref"]==g.file_ref(config_path) and
           bridge["native_commands_ref"]==g.file_ref(native_work/"commands.json") and
           bridge["truth_read"] is False,"Successful inferred observation bridge required")
    terminal=g.metadata(g.verify_ref(bridge["native_terminal_ref"]))
    g.need(terminal["actual_wait_consumed"] and terminal["returncodes"]==[0,0],"Actual successful native waits required")
    g.verify_ref(bridge["tree_ref"]); g.verify_ref(bridge["roster_ref"])
    # Check gzip CRC/trailer as well as the immutable compressed-file checksum.
    compressed=manifest["observed_files"].get("genotypes.vcf.gz")
    if compressed:
        with gzip.open(g.verify_ref(compressed),"rb") as stream:
            while stream.read(4*1024*1024):
                guard(0)
    else:
        g.need(parent_id=="published_recon_train_000" and "external_observed_vcf_ref" in manifest,
               "Verified gzip observations or authenticated historical TRAIN0 required")
        g.verify_ref(manifest["external_observed_vcf_ref"])
    candidates=[native_work/"input.haps",native_work/"inferred",native_work/"inferred.anc",
                native_work/"inferred.mut",native_work/"inferred.trees"]
    inventory=[]
    for candidate in candidates:
        g.need(not candidate.is_symlink(),"Scratch target is a symlink")
        if not candidate.exists():
            continue
        descendants=list(candidate.rglob("*")) if candidate.is_dir() else [candidate]
        for item in descendants:
            g.need(not item.is_symlink(),"Scratch descendant is a symlink")
            if item.is_file():
                inventory.append(dict(path=str(item.relative_to(g.ROOT)),bytes=item.stat().st_size))
    output=g._fresh_directory(output,guard)
    evidence=dict(schema="paper_argmix_new_scratch_release_v1",parent_id=parent_id,
                  config_ref=g.file_ref(config_path),source_ref=g.file_ref(__file__),
                  observed_manifest_ref=g.file_ref(observed/"observation_manifest.json"),
                  inferred_bridge_ref=g.file_ref(inferred_output/"completed.json"),
                  native_terminal_ref=bridge["native_terminal_ref"],inventory=inventory,
                  released_logical_bytes=sum(row["bytes"] for row in inventory),historical_targets=False,
                  retained="gzip or original TRAIN0 VCF, compact genotypes, truth, sanitized trees, rosters, support, features, logs, commands and real wait manifests")
    budget=g.Writes(16*1024*1024,guard)
    budget.json(output/"verified_before_release.json",evidence)
    for candidate in candidates:
        guard(0)
        if candidate.is_dir():
            shutil.rmtree(candidate)
        elif candidate.is_file():
            candidate.unlink()
    evidence["status"]="new_declared_scratch_released_after_verified_conversion"
    evidence["remaining_native_inputs"]={p.name:g.file_ref(p) for p in native_work.iterdir() if p.is_file()}
    budget.json(output/"completed.json",evidence)
    return evidence
