# Environment reconstruction
All commands run inside /workspace-SR003.nfs2/ARG-GNN after sourcing scripts/project_env.sh.
Create .runtime/env with the installed conda executable and the project condarc.
Use the explicit conda package URLs in experiments/conda-linux-64.lock.txt for the bootstrap environment.
Install torch==2.6.0 from the official cu124 PyTorch index, then the project requirements.
experiments/environment.lock.txt records resolved package versions. CUDA wheel versions need the official PyTorch index.
PYTHONNOUSERSITE=1 is required to prevent importing shared user-site packages.
This environment targets Linux x86_64 and is not a platform-independent lock.
NVTOP 3.3.2 is installed by conda. The provided project environment variables confine runtime files.

Tectonic 0.15.0 and Poppler are installed in the dedicated environment. The official AAAI-27 style requires PDFTeX, so a verified TinyTeX archive was additionally extracted inside .runtime/tex for compilation. Tectonic uses XeTeX and cannot satisfy that engine guard. No installer or global PATH registration was run. See docs/tex_provenance.json. The native desktop compiler cannot read the remote multi-file AAAI project without copying it onto the Mac, which the storage constraint forbids.
