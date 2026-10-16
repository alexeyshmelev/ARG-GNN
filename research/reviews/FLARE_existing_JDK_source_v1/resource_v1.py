"""Pure FLARE byte plans from exact root-observed counts; no data or tool I/O."""
def plan_v1(sites,reference_haplotypes,query_haplotypes,focals):
    need=lambda ok: (_ for _ in ()).throw(ValueError("Exact positive paired observation dimensions")) if not ok else None
    need(all(type(v) is int and v>0 for v in (sites,reference_haplotypes,query_haplotypes,focals))
         and reference_haplotypes%2==query_haplotypes%2==0 and reference_haplotypes//2>50)
    gzip=lambda n:n+(n+99)//100+65536
    raw=lambda n:4194304+sites*(256+4*n)
    conversion=gzip(raw(reference_haplotypes//2))+gzip(raw(query_haplotypes//2))+16*sites+16777216
    native_vcf=gzip(4194304+sites*(256+64*(query_haplotypes//2)))
    native=native_vcf+2*1048576+2*16777216+1073741824+16777216
    seal=focals*512+16777216
    return dict(sites=sites,reference_haplotypes=reference_haplotypes,query_haplotypes=query_haplotypes,
        focal_records=focals,conversion_original_formula_bytes=conversion,
        native_VCF_original_formula_bytes=native_vcf,native_original_formula_with1GiB_tmp_bytes=native,
        seal_original_formula_without_optional_tract_TSV_bytes=seal,
        optional_all_marker_haplotype_tract_upper_records=sites*query_haplotypes,
        optional_all_marker_haplotype_tract_upper_bytes=sites*query_haplotypes*96+4096,
        native_markers_are_bounded_by_actual_input_sites_not_assumed_filter_saving=True,
        compression_savings_assumed=False,actual_marker_retention_not_read=True)
