"""Unmodified original VCF loop, copied solely as a test oracle."""
import numpy as np
from arggnn.benchmarking.european_treeloom import _need

def read_observed_calls(vcf_path, rows, wanted, n_sites, *, resource_check):
    h = len(rows)
    genotype, allele_count = np.empty((n_sites,h),dtype=np.int8), np.empty(n_sites,dtype=np.int8)
    seen, header, previous = set(), None, 0
    with vcf_path.open() as stream:
        for line in stream:
            if line.startswith("##"):
                continue
            fields = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                header = fields[9:]
                _need(h % 2 == 0 and header == [r["sample_id"] for r in rows[::2]]
                      and all(rows[i]["sample_id"] == rows[i+1]["sample_id"]
                              and [rows[i]["haplotype"],rows[i+1]["haplotype"]] == [0,1]
                              for i in range(0,h,2)), "Original phased diploid sample columns required")
                continue
            _need(header is not None and len(fields) == len(header)+9, "Complete original VCF rows required")
            position = int(fields[1])
            _need(position > previous, "Original observed VCF positions must remain ordered")
            previous = position
            index = wanted.get(position)
            if index is None:
                continue
            if len(seen) % 4096 == 0:
                resource_check("European_TreeLoom_complete_VCF_join")
            alleles = [fields[3]] + ([] if fields[4] == "." else fields[4].split(","))
            _need(0 < len(alleles) <= 4 and all(a in {"A","C","G","T"} for a in alleles)
                  and len(set(alleles)) == len(alleles), "Original nucleotide allele order required")
            format_names = fields[8].split(":")
            _need("GT" in format_names, "Actual observed GT calls required")
            gt_index = format_names.index("GT")
            calls = []
            for sample in fields[9:]:
                gt = sample.split(":")[gt_index]
                _need("|" in gt, "Observed haplotypes must remain phased")
                pair = gt.split("|")
                _need(len(pair) == 2, "Two actual haplotypes per VCF sample required")
                for call in pair:
                    value = -1 if call == "." else int(call)
                    _need(-1 <= value < len(alleles), "Observed nucleotide genotype index required")
                    calls.append(value)
            genotype[index] = calls
            allele_count[index] = len(alleles)
            seen.add(index)
    _need(header is not None and len(seen) == n_sites, "Every native SNP requires original observed calls")
    return genotype, allele_count
