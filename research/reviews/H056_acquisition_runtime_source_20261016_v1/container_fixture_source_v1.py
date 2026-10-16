"""One deferred tiny container-only fixture. No execution or IO on import.
Root injects its guarded save and existing supervised tool command runner once.
No native/model fixture, data acquisition or scientific QC qualification.
"""
def qualify_v1(encoder_type, guarded_save, supervised_tool, bgzip, tabix, bcftools):
    raw = (b"##fileformat=VCFv4.2\n##contig=<ID=20,length=63025520>\n"
           b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
           b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tfixture\n"
           b"20\t20000001\tcontainer_only_A\tA\tC\t.\tPASS\t.\tGT\t0|1\n"
           b"20\t29999999\tcontainer_only_B\tG\tT\t.\tPASS\t.\tGT\t1|0\n")
    pieces = []
    encoder = encoder_type(pieces.append)
    for at in range(0,len(raw),7):
        encoder.append(raw[at:at+7])
    metadata = encoder.close()
    import gzip,hashlib
    compressed = b"".join(pieces)
    assert gzip.decompress(compressed) == raw
    assert metadata["logical_sha256"] == hashlib.sha256(raw).hexdigest()
    vcf_path = guarded_save("container_fixture.vcf.gz",compressed)
    outputs = {}
    for name,argv in (
        ("integrity",[bgzip,"-t",vcf_path]),
        ("index",[tabix,"-p","vcf",vcf_path]),
        ("samples",[bcftools,"query","-l",vcf_path]),
        ("count",[bcftools,"index","-n",vcf_path]),
        ("contigs",[tabix,"-l",vcf_path])):
        outputs[name] = supervised_tool(name,argv)
    assert outputs["samples"].strip() == b"fixture"
    assert outputs["count"].strip() == b"2"
    assert outputs["contigs"].strip() == b"20"
    return dict(schema="H056_tiny_BGZF_tabix_container_fixture_v1",passed=True,
                logical_bytes=len(raw),compressed_bytes=len(compressed),metadata=metadata,
                scientific_QC_or_model_qualification=False)
