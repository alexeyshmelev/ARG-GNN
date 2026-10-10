# Proposed containment only. Sourcing exports variables; it creates no paths.
# Activate the root-created candidate prefix, then re-source this file.
# Keep source copies and all build outputs separate for cpu and cu129.
export ARGMIX_NATIVE_REPO="/workspace-SR003.nfs2/ARG-GNN"
export ARGMIX_NATIVE_PROFILE="${ARGMIX_NATIVE_PROFILE:-cpu}"
case "$ARGMIX_NATIVE_PROFILE" in
  cpu|cu129) ;;
  *) printf "%s\n" "ARGMIX_NATIVE_PROFILE must be cpu or cu129" >&2; return 2 2>/dev/null || exit 2 ;;
esac
export ARGMIX_NATIVE_BASE="$ARGMIX_NATIVE_REPO/.runtime/official_argmix_abi210_v1"
export ARGMIX_NATIVE_WORK="$ARGMIX_NATIVE_BASE/$ARGMIX_NATIVE_PROFILE"
export ARGMIX_NATIVE_PREFIX="$ARGMIX_NATIVE_WORK/env"
export ARGMIX_NATIVE_SOURCE="$ARGMIX_NATIVE_WORK/source"
export ARGMIX_NATIVE_AUTHOR_SOURCE="$ARGMIX_NATIVE_REPO/.runtime/author_sources/argmix_c2481248eb0d"
export ARGMIX_NATIVE_LOG_DIR="$ARGMIX_NATIVE_WORK/logs"
export TMPDIR="$ARGMIX_NATIVE_WORK/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$ARGMIX_NATIVE_WORK/cache"
export XDG_CONFIG_HOME="$ARGMIX_NATIVE_WORK/config"
export XDG_DATA_HOME="$ARGMIX_NATIVE_WORK/data"
export XDG_STATE_HOME="$ARGMIX_NATIVE_WORK/state"
export CARGO_HOME="$XDG_CACHE_HOME/cargo"
export RUSTUP_HOME="$XDG_CACHE_HOME/rustup"
export CARGO_TARGET_DIR="$ARGMIX_NATIVE_WORK/targets/default"
export CONDA_PKGS_DIRS="$XDG_CACHE_HOME/conda-pkgs"
export CONDA_ENVS_PATH="$ARGMIX_NATIVE_WORK/conda-envs"
export CONDARC="$ARGMIX_NATIVE_WORK/config/condarc"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export PIP_CONFIG_FILE=/dev/null
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PYTHON_INSTALL_DIR="$ARGMIX_NATIVE_WORK/uv-python"
export PIXI_HOME="$XDG_CACHE_HOME/pixi-home"
export PIXI_CACHE_DIR="$XDG_CACHE_HOME/pixi"
export RATTLER_CACHE_DIR="$XDG_CACHE_HOME/rattler"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export WANDB_DIR="$ARGMIX_NATIVE_WORK/wandb"
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export WANDB_CONFIG_DIR="$ARGMIX_NATIVE_WORK/config/wandb"
export WANDB_DATA_DIR="$ARGMIX_NATIVE_WORK/wandb-data"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export PATH="$ARGMIX_NATIVE_PREFIX/bin:$PATH"
export PYTHONPATH="$ARGMIX_NATIVE_SOURCE/src"
export PYO3_PYTHON="$ARGMIX_NATIVE_PREFIX/bin/python"
export LIBTORCH_USE_PYTORCH=1
unset LIBTORCH_BYPASS_VERSION_CHECK LIBTORCH LIBTORCH_INCLUDE LIBTORCH_LIB
unset LIBTORCH_STATIC LIBTORCH_CXX11_ABI TORCH_CUDA_VERSION
export HDF5_DIR="$ARGMIX_NATIVE_PREFIX"
export LIBCLANG_PATH="$ARGMIX_NATIVE_PREFIX/lib"
export CC="$ARGMIX_NATIVE_PREFIX/bin/clang"
export CXX="$ARGMIX_NATIVE_PREFIX/bin/clang++"
export RUSTFLAGS="-Clinker-features=-lld"
export CARGO_BUILD_JOBS=2
export LD_LIBRARY_PATH="$ARGMIX_NATIVE_PREFIX/lib:$ARGMIX_NATIVE_PREFIX/lib/python3.12/site-packages/torch/lib"
export CUDA_VISIBLE_DEVICES=""
