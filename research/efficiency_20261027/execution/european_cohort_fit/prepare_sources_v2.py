"""Narrow v1 worker adaptation to the reviewed deterministic resume driver."""
from pathlib import Path
HERE=Path(__file__).parent
text=(HERE/'fit_cli_v1.py').read_text()
text=text.replace('Fresh FIT only. No resume API is exposed; checkpoints omit RNG state.',
    'Optional exact continuation restores all RNG streams and the next parent, within the original90-update budget.')
text=text.replace('from .fit_gate_v1 import ROOT,RUNS,GPU_UUID,need,contained,reference,read,qualification_release',
    'from .fit_gate_v2 import ROOT,RUNS,GPU_UUID,need,contained,reference,read,qualification_release,resume_release')
text=text.replace('from .progress_v1 import snapshot','from .progress_v2 import snapshot')
text=text.replace('european_cohort_fit.fit_cli_v1','european_cohort_fit.fit_cli_v2')
text=text.replace('fit_cli_pins_v1.json','fit_cli_pins_v2.json')
text=text.replace('European_cohort8_FIT_CLI_sources_v1','European_cohort8_FIT_CLI_sources_v2')
text=text.replace("HERE/'fit_cli_v1.py'","HERE/'fit_cli_v2.py'")
text=text.replace("HERE/'fit_gate_v1.py'","HERE/'fit_gate_v2.py'")
text=text.replace('European_cohort8_FIT_CLI_review_v1','European_cohort8_FIT_CLI_review_v2')
text=text.replace("    return manifest_ref,review_ref", "    need(review.get('resume_restore_CPU_equivalence_passed') is True,'Independent restore-equivalence source review required')\n    return manifest_ref,review_ref")
text=text.replace("    parser.add_argument('--fit-release-sha256',required=True)",
    "    parser.add_argument('--fit-release-sha256',required=True)\n    parser.add_argument('--resume-driver-pins-sha256',required=True)\n    parser.add_argument('--resume-checkpoint-path')\n    parser.add_argument('--resume-checkpoint-sha256')")
text=text.replace("    output=contained(args.output);directory=output.parent", "    resume_gate=resume_release(args.resume_checkpoint_path,args.resume_checkpoint_sha256)\n    output=contained(args.output);directory=output.parent")
text=text.replace('import driver_v2 as driver','import driver_v3 as driver')
text=text.replace("    runtime=driver.read_small", "    need(driver.source_manifest_ref()['sha256']==args.resume_driver_pins_sha256,'Exact resume driver source manifest')\n    runtime=driver.read_small")
text=text.replace("        fit_release_ref=initial_gate['fit_release_ref'])", "        fit_release_ref=initial_gate['fit_release_ref'],\n        resume_checkpoint_ref=None if resume_gate is None else resume_gate['resume_checkpoint_ref'])")
text=text.replace("            need(gate==initial_gate,'Exact qualification/release evidence unchanged')", "            need(gate==initial_gate,'Exact qualification/release evidence unchanged')\n            need(resume_release(args.resume_checkpoint_path,args.resume_checkpoint_sha256)==resume_gate,\n                 'Exact prior FIT family and selected checkpoint remain unchanged')")
text=text.replace('physical_total_bytes=total,qualification_gate=gate)', 'physical_total_bytes=total,qualification_gate=gate,resume_gate=resume_gate)')
text=text.replace('worker_RSS_bytes=512*GIB,CPU_threads=8,GPU_lease=row)',
    "worker_RSS_bytes=512*GIB,CPU_threads=8,GPU_lease=row,resume_gate=resume_gate,\n            resume_driver_source_manifest_ref=driver.source_manifest_ref())")
text=text.replace('fresh_fit_only=True,resume_supported=False,source_manifest_ref=SOURCE_MANIFEST_REF,',
    "fresh_fit_only=False,resume_supported=True,source_manifest_ref=SOURCE_MANIFEST_REF,\n            resumed_from_ref=result['resumed_from_ref'],resumed_after_updates=result['resumed_after_updates'],\n            resume_checkpoint_ref=result['resume_checkpoint_ref'],resume_driver_source_manifest_ref=driver.source_manifest_ref(),")
text=text.replace('European_cohort8_FIT_CLI_completion_v1','European_cohort8_FIT_CLI_completion_v2')
(HERE/'fit_cli_v2.py').write_text(text)
print('Prepared resume-capable FIT CLI v2;no launch.')
