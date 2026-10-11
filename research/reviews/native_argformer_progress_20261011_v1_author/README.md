# Native ARGformer progress and ancestry scoring

VAL000 r003 fit and CPU retrieval have completed. VAL001 r003 is still fitting in the recorded snapshot. progress.json records source-bound receipts and current MLM stage; no MLM accuracy is interpreted as ancestry accuracy.

Once both VAL retrieval receipts are passed, run the existing scorer functions through this small owned wrapper:
    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python -B research/reviews/native_argformer_progress_20261011_v1_author/score_ready_VAL.py --execute

The wrapper waits for both seals before opening external VAL targets. It reuses score_parent, ExternalVALC4Truth and full_span_score unchanged; it never runs models or cosine retrieval. It creates only the new compact experiments/results/native_argformer_tsinfer_fit_v7_summary_v2 packet, preserving raw top1/top5 probabilities and exact common Present64/C4 support. Actual task differences and transductive fits are disclosed.
