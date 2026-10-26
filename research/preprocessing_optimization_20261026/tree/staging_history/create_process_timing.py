from pathlib import Path
out=Path('/workspace-SR003.nfs2/ARG-GNN/research/preprocessing_optimization_20261026/tree')
s=(out/'native_timing.py').read_text()
s=s.replace('torch.set_num_threads(1)\ntorch.set_num_interop_threads(1)\n','')
s=s.replace('import dataclasses\n','import dataclasses\nimport argparse\n')
s=s.replace('observations, parallel_observations, episodes, cut','observations, process_observations, episodes, cut')
s=s.replace('def main():\n', '''def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trees',type=int,default=4096)
    arguments=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
''')
s=s.replace('First 256 unchanged native holds','First requested unchanged native holds').replace('ts.trees(),256','ts.trees(),arguments.trees')
s=s.replace("    variants += [(f'compiled_threads_{workers}',parallel_observations.encode_block,dict(preparation_workers=workers)) for workers in (1,2,4,8)]", "    variants += [(f'compiled_processes_{workers}',process_observations.encode_block,dict(preparation_workers=workers,tree_chunk_size=128)) for workers in (2,4,8)]")
start=s.index('    isolated=tuple(reference_meta.get(')
end=s.index('    report=dict(status=',start)
s=s[:start]+s[end:]
s=s.replace("(OUT/'native_timing.json').write_text", "(OUT/f'process_timing_{arguments.trees}.json').write_text")
s=s.replace('no_truth_or_model_opened=True,no_reduction_inside_selected_native_holds=True)',
    'no_truth_or_model_opened=True,no_reduction_inside_selected_native_holds=True,includes_process_startup_and_serialization=True,maximum_pending_chunks=8,trees_per_chunk=128)')
(out/'process_timing.py').write_text(s)
