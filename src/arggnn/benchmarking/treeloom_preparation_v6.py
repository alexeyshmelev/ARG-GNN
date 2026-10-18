"""V6 factory lifecycle: fresh task/native output; frozen v5 lineage and v4 numerics."""
from __future__ import annotations
from . import contracts as c
from . import treeloom_preparation_v5 as frozen

PARENT=frozen.PARENT
observed_lineage=frozen.observed_lineage


def _with_v6_source(descriptor):
    descriptor["source_refs"]=list({ref["path"]:ref for ref in
                                   [*descriptor["source_refs"],c.file_ref(__file__)]}.values())
    return descriptor


def qualification_factory(*,preparation_ref,generation_ref,output_directory,max_rss_bytes,
                          max_seconds=7200,max_output_bytes=16*2**20):
    # Materialization supplies task/native before task exists. The supervisor
    # subsequently creates task (plus tmp/cache), leaving native fresh for the
    # frozen qualifier's single out.mkdir() call.
    output=c.contained_path(output_directory)
    frozen._need(output.is_relative_to(c.ROOT/"experiments/results")
                 and output.name=="native" and not output.exists(),
                 "Fresh supplied task/native directory required; supervisor creates only its parent")
    descriptor=frozen.qualification_factory(preparation_ref=preparation_ref,
        generation_ref=generation_ref,output_directory=output_directory,
        max_rss_bytes=max_rss_bytes,max_seconds=max_seconds,max_output_bytes=max_output_bytes)
    descriptor["spec"]["output_directory"]=str(output.relative_to(c.ROOT))
    return _with_v6_source(descriptor)


def fit_factory(*,preparation_refs,generation_refs,output_directory,max_rss_bytes,qualification_ref):
    return _with_v6_source(frozen.fit_factory(preparation_refs=preparation_refs,
        generation_refs=generation_refs,output_directory=output_directory,
        max_rss_bytes=max_rss_bytes,qualification_ref=qualification_ref))


run_qualification=frozen.run_qualification
run_fit=frozen.run_fit
