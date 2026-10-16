"""Build the self-contained AAAI rewrite using the existing remote TeX runtime."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import io
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BUILD = OUT / "build"
TEX = ROOT / ".runtime/tex/.TinyTeX/bin/x86_64-linux"
env = os.environ.copy()
env.update(
    TEXMFHOME=str(ROOT / ".runtime/texmf"),
    TEXMFVAR=str(ROOT / ".runtime/tex/var"),
    TEXMFCONFIG=str(ROOT / ".runtime/tex/config"),
    TEXMFCACHE=str(BUILD / "cache"),
    TMPDIR=str(BUILD),
    BIBINPUTS=str(OUT),
    BSTINPUTS=str(OUT),
)
stages = [
    ("final_pass1", [str(TEX / "pdflatex"), "-interaction=nonstopmode",
                     "-halt-on-error", "-output-directory=build", "manuscript.tex"]),
    ("final_bibtex", [str(TEX / "bibtex"), "build/manuscript"]),
    ("final_pass2", [str(TEX / "pdflatex"), "-interaction=nonstopmode",
                     "-halt-on-error", "-output-directory=build", "manuscript.tex"]),
    ("final_pass3", [str(TEX / "pdflatex"), "-interaction=nonstopmode",
                     "-halt-on-error", "-output-directory=build", "manuscript.tex"]),
]
actual = []
for name, command in stages:
    result = subprocess.run(command, cwd=OUT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    log = BUILD / (name + ".stdout")
    log.write_bytes(result.stdout)
    actual.append(dict(stage=name, argv=command, returncode=result.returncode,
        stdout=str(log.relative_to(ROOT))))
    print(name, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)

shutil.copyfile(BUILD / "manuscript.pdf", OUT / "manuscript.pdf")
log = (BUILD / "manuscript.log").read_text()
warnings = [line for line in log.splitlines()
            if "Warning" in line or "Overfull" in line]
pdfinfo = subprocess.run(["/usr/bin/pdfinfo", str(OUT / "manuscript.pdf")],
    stdout=subprocess.PIPE, check=True, text=True).stdout
page_count = int(re.search(r"Pages:\s+(\d+)", pdfinfo).group(1))
text = subprocess.run(["/usr/bin/pdftotext", "-layout",
    str(OUT / "manuscript.pdf"), "-"], stdout=subprocess.PIPE,
    check=True, text=True).stdout
render = BUILD / "final_rendered"
render.mkdir(exist_ok=True)
pages = []
for number in range(1, page_count + 1):
    result = subprocess.run(["/usr/bin/pdftoppm", "-r", "85", "-f", str(number),
        "-l", str(number), "-singlefile", "-png", str(OUT / "manuscript.pdf")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=30)
    (render / ("page_" + str(number).zfill(2) + ".png")).write_bytes(result.stdout)
    page = Image.open(io.BytesIO(result.stdout)).convert("RGB")
    page.thumbnail((612, 792))
    pages.append(page)
for start in range(0, page_count, 3):
    ims = pages[start:start + 3]
    contact = Image.new("RGB", (sum(im.width for im in ims),
        max(im.height for im in ims) + 25), "#e3e8ed")
    x = 0
    for offset, im in enumerate(ims):
        contact.paste(im, (x, 25))
        ImageDraw.Draw(contact).text((x + 8, 5),
            "Page " + str(start + offset + 1), fill="black")
        x += im.width
    contact.save(render / ("contact_" + str(start + 1) + "_" +
        str(start + len(ims)) + ".jpg"), quality=85)
receipt = dict(
    stages=actual, pages=page_count, warnings=warnings,
    source_sha256=hashlib.sha256((OUT / "manuscript.tex").read_bytes()).hexdigest(),
    figure_sha256=hashlib.sha256((OUT / "figures/whole_tree_architecture.tex").read_bytes()).hexdigest(),
    bibliography_sha256=hashlib.sha256((OUT / "references.bib").read_bytes()).hexdigest(),
    pdf_sha256=hashlib.sha256((OUT / "manuscript.pdf").read_bytes()).hexdigest(),
    render_directory=str(render.relative_to(ROOT)),
    page_text_presence=[dict(page=i+1, has_references="References" in t,
        has_appendix="Simulation Design and Biological" in t,
        beginning=t[:180], ending=t[-250:])
        for i, t in enumerate(text.split("\f")) if t.strip()],
    visual_review_pending=True,
    scientific_fidelity_audit_pending=True,
)
(BUILD / "compile_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(dict(pages=page_count, warnings=warnings,
    render_directory=str(render)), indent=2), flush=True)
