# Keep every temp file, cache and download on D: (the C: drive is nearly full).
# Usage from bash: source /d/claude-code/baadhi/env.sh
B=/d/claude-code/baadhi
export TEMP="D:\claude-code\baadhi\.work\tmp" TMP="D:\claude-code\baadhi\.work\tmp" TMPDIR="$B/.work/tmp"
export UV_CACHE_DIR="$B/.work/uv-cache" UV_PYTHON_INSTALL_DIR="$B/.work/python" UV_LINK_MODE=copy
export PIP_CACHE_DIR="$B/.work/pip-cache" HF_HOME="$B/.work/hf" TORCH_HOME="$B/.work/torch"
export npm_config_cache="$B/.work/npm-cache" PYTHONUTF8=1
export PATH="$B/.venv/Scripts:$PATH"
