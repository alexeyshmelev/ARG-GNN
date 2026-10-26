"""Publish concise figure status after source, reader and compilation checks."""
from pathlib import Path
import hashlib
import json
import re

root = Path.cwd()
out = root / 'paper/figure_redesign_20261026_v5'
receipt = json.loads((out/'integration.json').read_text())
assert receipt['all_eight_result_tables_unchanged'] and receipt['reader_acceptance']['accepted']
assert receipt['canonical_compile_success'] and receipt['overleaf_compile_success']
assert receipt['visual_inspection_complete']

p = root/'README.md'
s = p.read_text()
stop = re.search(r'<!-- owner-stop-current:start -->.*?<!-- owner-stop-current:end -->', s, re.S).group(0)
block = '''<!-- figure-redesign-current:start -->
**TreeLoom tensor diagrams, 26 October:** [prediction flow](paper/figures/TreeLoom_architecture.png) and [encoder expansion](paper/figures/TreeLoom_encoder.png) trace concrete input rows, learned layers, tensor assembly, both tree rounds and leaf memory. A fresh reader traced a named leaf across intervals and a named ancestor through the computation using only the images. The stricter review replaces revision 4's broad conceptual reading, which did not meet the owner's requirements. [Paper](paper/current_full_panel_working.pdf) · [Overleaf bundle](paper/current_full_panel_overleaf_working.zip) · [Review and source checks](paper/figure_redesign_20261026_v5/README.md). All eight table source blocks are unchanged. Experiments remain stopped.
<!-- figure-redesign-current:end -->'''
s, count = re.subn(r'<!-- figure-redesign-current:start -->.*?<!-- figure-redesign-current:end -->', lambda m:block, s, flags=re.S)
assert count == 1 and stop in s
p.write_text(s)

p = root/'paper/README.md'
s = p.read_text()
s = re.sub(r'The ten-page AAAI draft.*?occupies pages 7–10\.',
           'The AAAI draft presents **TreeLoom** for **local ancestry inference (LAI)**. The manuscript includes the main paper, references and scientific supplement.', s, flags=re.S)
start=s.index('Figures 1 and 2 show')
end=s.index('\n\nThe main tables',start)
s=s[:start]+'''Figures 1 and 2 trace the prediction computation and expand its tree encoder. Node and leaf rows, operator operands, tensor assembly, interval measurements, shared tree rounds and memory writeback are explicit. The saved example retains five leaves and three genomic intervals. [Prediction flow](figures/TreeLoom_architecture.png) · [Encoder](figures/TreeLoom_encoder.png) · [Generators, source checks and independent reading](figure_redesign_20261026_v5/README.md).'''+s[end:]
start=s.index('The manuscript and a freshly extracted Overleaf bundle')
end=s.index('\n\nThe manuscript remains',start)
pages=receipt['canonical_pages']
s=s[:start]+f'''The manuscript and a freshly extracted Overleaf bundle compile to {pages} pages. Both figure pages and the full document were visually inspected. All eight table source blocks are unchanged. [Build and review record](figure_redesign_20261026_v5/integration.json).'''+s[end:]
p.write_text(s)

(out/'README.md').write_text('''# TreeLoom tensor diagrams, revision 5

[Prediction flow](../figures/TreeLoom_architecture.png) · [Encoder expansion](../figures/TreeLoom_encoder.png) · [Paper](../current_full_panel_working.pdf) · [Overleaf bundle](../current_full_panel_overleaf_working.zip)

The two figures use explicit tensor rows, connected operands and repeated operations. The example follows the same leaves through three intervals and shows all ancestor rows through both graph rounds. The encoder has a marked parent and expansion. Memory copying, both recurrence directions, group reduction, subtraction and scatter are distinct. Every color has a semantic legend.

Revision 4's conceptual reading passed too weak a test. The owner still could not trace the computation. Revision 5 uses a fresh reader with no source, manuscript or conversational history, who must trace a named leaf and ancestor from the images. The root checks this reconstruction against the frozen implementation. The [review record](review.md) preserves issues and repairs. This is a communication check, not owner approval or evidence of scientific novelty.

The final fresh reader correctly reconstructed all tested key paths. [Full independent explanation](review_round3/reader_report.md) · [Acceptance and exact image hashes](reader_acceptance.json). The two figures occupy pages 3 and 4 of the eleven-page compiled draft.

[Source contract](source_audit.md) · [Source hashes](source_hashes.json) · [Integration receipt](integration.json)

Observed values come from the previously saved illustrative tree sequence. Hidden activations and probabilities are schematic. No new simulation, model training, profiling, checkpoint change or benchmark run occurred. All eight manuscript table source blocks are unchanged. Experiments remain stopped.

Rebuild the CPU-only drawings on anogena from the dedicated repository:

```bash
. scripts/project_env.sh
"$ARGGNN_PYTHON" paper/figure_redesign_20261026_v5/build_architecture.py
"$ARGGNN_PYTHON" paper/figure_redesign_20261026_v5/build_encoder.py
```

The integration script requires a reader acceptance receipt bound to the exact images. Canonical PDF, SVG and PNG figures are also stored in `paper/figures/`. The Overleaf bundle contains its main source, bibliography, AAAI style and both vector PDFs.
''')
print('Updated figure documentation. Owner stop block preserved exactly.')
