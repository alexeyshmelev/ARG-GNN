# AAAI-27 template and bibliography provenance

Checked 2026-10-10. The manuscript assets use the **2027 Author Kit,
TemplateVersion 2027.1**. No AAAI-26 or AAAI-25 fallback is used.

## Acquisition and integrity

Official sources:

- [AAAI-27 Author Kit](https://aaai.org/authorkit27/).
- [Official AuthorKit27.zip](https://aaai.org/wp-content/uploads/2026/05/AuthorKit27.zip).
- [Main Technical Track call](https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/).
- [Submission instructions](https://aaai.org/conference/aaai/aaai-27/submission-instructions/).

Direct HTTP requests to the AAAI kit pages and archive returned 403 from the
available clients. No access-control bypass was attempted. The primary agent
located the existing browser-downloaded official archive and copied it into
`.runtime/cache/author-kit/AuthorKit27.zip` on anogena. No local path, account
identifier or browser credentials are recorded here.

The cached archive is 5,495,535 bytes and has SHA256:

`e28c6ac9bc6eb3b4e2d849547d2cefb5162610ee39d0a12e0dc62d1126b44a7d`

This matches the independently recorded official-kit fingerprint in
[this public provenance record](https://github.com/liusulldel/aaai-hard-gate/blob/261bbfb77055d552f0daa0721192a3203fa16d8a/aaai27-hard-gate/references/official-gates.md).
The style and bibliography style were also independently cross-checked against
public mirrors at pinned commits in
[Hojjat-AAAI-2027](https://github.com/Hojat-Asgariandehkordi/Hojjat-AAAI-2027/tree/0bb6423a01e43bf91d58c64c392251ae8f64ab98),
[DSCLR](https://github.com/lumos1596/DSCLR/tree/add1c07fa18dc6653dd6ae9dadbae44019401f64/paper/AuthorKit27/AuthorKit27),
and [the voxel-search report](https://github.com/shpigelgi/final-project-voxel-bidirectional-search/tree/c49fc3acff9c09b29e9bee8f34bcbfed533eb52d/deliverables/report/overleaf).
All three copies agree with the official browser-downloaded archive's files.
Those mirrors were integrity cross-checks, not the acquisition source.

The extraction checked a 20 MB archive bound, a 100 MB total extraction bound,
member count, relative paths, traversal, platform path separators, symbolic links
and special file modes. Seventeen original files, totaling 5,824,174 bytes, were
extracted unchanged into `paper/vendor/AuthorKit27/`. No template asset was run
or modified. All extracted files are non-executable regular files.
`paper/vendor/author_kit_manifest.json` records every original member's byte size
and SHA256. No copyright transfer or other legal contract was accepted.

Key fingerprints:

| Asset | SHA256 |
| --- | --- |
| aaai2027.sty | `391bce82815bf698b8e382dd3ae7e30c75d7ab46df140cb295b1266016bc8623` |
| aaai2027.bst | `5db7765ba99de5c1e4686f9b3940a0add9c5e702f2164514462bec130ccb6e3c` |
| AnonymousSubmission2027.tex | `035ebdb17e57885a1fd43a188fd17777bdbf90f1fda1a1e000c49c7f52ce1f9d` |
| CameraReady2027.tex | `db509cb393dae5c113bf29633e17d31a5fefa610aad09056569034f9bcff5b46` |
| ReproducibilityChecklist.tex | `06a3459158089bf1c64b738986118f1d1566e816da4b710c6397561e33c3d5e6` |

## Required preamble and author instructions

Use the unchanged `aaai2027.sty` and `aaai2027.bst`. The original anonymous and
camera-ready sample sources remain available for checking details. The mandatory
anonymous preamble begins:

```latex
\documentclass[letterpaper]{article}
\usepackage[submission]{aaai2027}
\usepackage[hyphens]{url}
\usepackage{graphicx}
\urlstyle{rm}
\def\UrlFont{\rm}
\usepackage{natbib}
\usepackage{caption}
\frenchspacing
\pdfinfo{
/TemplateVersion (2027.1)
}
```

The 2027 style loads `newtxtext`, `helvet` and `courier` itself. Do not copy an
older template's explicit `times`, `helvet`, `courier` or page-dimension commands
into this preamble. The style also selects `aaai2027` bibliography formatting
itself. Use `\bibliography{references}` without adding a conflicting bibliography
style. `aaai2027.bst` must be discoverable under its original basename.

The sample permits the optional `algorithm`/`algorithmic` block when needed and
recommends `booktabs`. It permits section-number depth 0, 1 or 2. It does not
permit modifying the layout, spacing or fonts to fit additional text.

The current AAAI-27 Main Technical Track call, inspected by the primary agent in
the official browser, limits main non-reference content to **seven pages** and
the complete main PDF to **nine pages**, with pages 8–9 reserved for references.
The roughly twenty-page overall research-document target is not a main-track
PDF allowance. Any technical supplement must follow the current track-specific
instructions separately. The generic kit does not itself establish permission
to append a supplement to the review PDF.

Other concrete requirements in the supplied instructions:

- PDFLaTeX-compatible US-Letter, two-column layout, no page numbers, headers or
  footers, and PDF version 1.5 or higher.
- Embedded fonts in text and graphics, with no Type 3 fonts. Use the style's text
  fonts and keep Title Case capitalization for the manuscript title.
- Anonymous review mode, anonymous author block, empty affiliations, no revealing
  metadata or acknowledgments. No copyright-transfer form is needed at review.
- No layout or spacing overrides, negative spacing around structural elements,
  `geometry`, `hyperref`, `navigator`, `times`, `titlesec`, `pgfplots`, or other
  packages explicitly prohibited by the sample. The full list in the original
  instructions governs packages not listed here.
- Insert figures with `graphicx`, pre-generate plots outside the paper source,
  crop them externally, and avoid in-source `clip`/`trim` or figure-grouping
  `minipage`. Raster figures should be 300 dpi at placed size.
- Tables use 10-point Roman text or, where necessary, 9-point text. Do not resize
  the entire table with `resizebox` or similar whole-table scaling commands.
- Preserve caption typography and placement, keep every object within margins
  and the gutter, and check figures in grayscale. The instructions require
  WCAG 2.0 contrast greater than 4.5:1 for color use.
- Follow current track requirements for the separate reproducibility checklist.
  The kit includes the original checklist source and PDF for that purpose.

For a portable Overleaf project, root should copy the required original style
and bibliography style unchanged beside the manuscript, or configure both TeX
and BibTeX search paths to the vendor directory. Include the manuscript's own
references and used figures. The instructional examples, Word files, example
bibliography and archive are provenance material, not automatically part of the
final source submission. The generic camera-ready instructions specify a single
manuscript source, a separate bibliography, only used graphics and a source ZIP
not exceeding 10 MB. Recheck event-specific instructions before submission.

No compilation was performed in this asset-acquisition subtask. Root owns
manuscript generation and render/font/page-limit QA. Successful compilation alone
does not establish compliance.

## Verified bibliography

`paper/references.bib` has seven unique entries. GST is both a supplied reference
and a closest comparator, so it appears once. Full authors are included, names
use portable LaTeX accents, and braces were structurally checked. Machine-readable
metadata provenance is in `paper/vendor/bibliography_provenance.json`.

| Key | Verified reference | Authority |
| --- | --- | --- |
| argformer2026 | Bonet, Shanks, Comajoan Cara, Abante and Ioannidis. ARGformer: learning on ancestral recombination graphs with transformers. Bioinformatics 42, Supplement 2, btag438, 2026 | [Crossref deposit](https://api.crossref.org/works/10.1093%2Fbioinformatics%2Fbtag438), [DOI](https://doi.org/10.1093/bioinformatics/btag438) |
| ancientancestry2026 | Shanks, Bonet, Comajoan Cara and Ioannidis. Graph transformer for ancient ancestry inference. bioRxiv, version 1, posted 7 April 2026 | [Crossref deposit](https://api.crossref.org/works/10.64898%2F2026.04.05.714076), [bioRxiv version metadata](https://api.biorxiv.org/details/biorxiv/10.64898/2026.04.05.714076), [version 1](https://www.biorxiv.org/content/10.64898/2026.04.05.714076v1) |
| eurasians2024 | Irving-Pease et al. The selection landscape and genetic legacy of ancient Eurasians. Nature 625(7994), 312–320, 2024 | [Crossref deposit](https://api.crossref.org/works/10.1038%2Fs41586-023-06705-1), [DOI](https://doi.org/10.1038/s41586-023-06705-1) |
| gst2026 | Jose E. Escrig Molina, Baoquan Chen and Daniel Probst. Graph Set Transformer. arXiv:2606.05116, 2026 | [arXiv version 1](https://arxiv.org/abs/2606.05116v1) |
| graphssm2024 | Jintang Li, Ruofan Wu, Xinzhou Jin, Boqun Ma, Liang Chen and Zibin Zheng. State Space Models on Temporal Graphs: A First-Principles Study. NeurIPS 37, 127030–127058, 2024 | [Official proceedings](https://papers.nips.cc/paper_files/paper/2024/hash/e5ba3d6d93213db6b1d1931c6517fe1a-Abstract-Conference.html), [DOI](https://doi.org/10.52202/079017-4034) |
| continuum2024 | Edoardo Calvello, Nikola B. Kovachki, Matthew E. Levine and Andrew M. Stuart. Continuum Attention for Neural Operators. arXiv:2406.06486, first posted 10 June 2024, consulted version 4 revised 20 December 2025 | [arXiv metadata](https://arxiv.org/abs/2406.06486), [consulted full text](https://arxiv.org/html/2406.06486v4) |
| settransformer2019 | Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi and Yee Whye Teh. Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. ICML, PMLR 97, 3744–3753, 2019 | [Official PMLR metadata and BibTeX](https://proceedings.mlr.press/v97/lee19d.html) |

Metadata details that should not be silently altered:

- Nature's DOI contains 2023, but the verified publication date is 10 January
  2024. The bibliography therefore uses 2024.
- The bioRxiv registered DOI is `10.64898/2026.04.05.714076`. The supplied `v1`
  suffix identifies its content version. Version 1 was posted on 7 April 2026,
  rather than using the 5 April date embedded in the identifier. bioRxiv reports
  the preprint's license as CC BY and its corresponding author as Alexander G.
  Ioannidis.
- Bioinformatics and bioRxiv display the same full name, Marçal Comajoan Cara.
  Their Crossref deposits split the given/family fields differently. Both entries
  use the family grouping Comajoan Cara, consistent with the bioRxiv deposit.
- Set Transformer uses the publisher's Adam Kosiorek spelling and verified pages
  3744–3753. An arXiv author record adds the middle initial R, but the proceedings
  deposit was used for the published entry.
- Continuum Attention is the exact requested neural-operator paper, not Sparse
  and Continuous Attention Mechanisms or Infinite-former. Its citation uses the
  original 2024 preprint year and states the consulted 2025 version in the note.
- GST remains labeled a preprint. An unspecified arXiv comment saying
  “conference” is not evidence of acceptance at any particular venue.
