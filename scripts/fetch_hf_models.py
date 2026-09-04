"""Fetch the HuggingFace models Phase 2 needs, into models/hf/<name>/.

Run this on a machine/network that can actually reach huggingface.co --
huggingface.co, hf-mirror.com, storage.googleapis.com, and
cdn-lfs.huggingface.co are all unreachable from the sandboxed dev
environment this project is otherwise built in (see TASKS.md's Phase 2
notes), so this script is meant to be run manually from a normal Terminal
with normal internet access -- not from that sandbox.

Usage:
    pip install huggingface_hub einops
    python3 scripts/fetch_hf_models.py

Can be run from ANY directory -- it resolves models/hf relative to this
file's own location (repo_root/models/hf), not the current working
directory, specifically because running it from inside scripts/ (an easy
mistake -- it happened twice) used to silently create a second,
duplicate scripts/models/hf/ instead of erroring.

Only weight/config/tokenizer files are pulled (`ALLOW_PATTERNS` below) --
NOT the onnx/openvino/tf/rust duplicate weight formats every HF repo also
ships, which is what made the first version of this script pull ~10GB for
~2GB of actually-needed files. Safe to re-run: snapshot_download skips
files that already match on disk.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
    ("sentence-transformers/all-mpnet-base-v2", "all-mpnet-base-v2"),
    ("nomic-ai/nomic-embed-text-v1.5", "nomic-embed-text-v1.5"),
    ("BAAI/bge-reranker-base", "bge-reranker-base"),
]

# Weights + everything needed to load and run the model; excludes the
# onnx/, openvino/, *.h5 (tf), *.ot (rust) duplicate formats every repo
# ships alongside model.safetensors.
ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.model", "*.py"]

# nomic-embed-text-v1.5's config.json auto_map points at a SEPARATE repo,
# nomic-ai/nomic-bert-2048, for its custom architecture code (there is no
# native NomicBertModel in transformers) -- normally transformers fetches
# that repo from the Hub automatically at load time, which fails offline.
# We need only its two .py source files, not its 525MB of (unrelated,
# separately-pretrained) weights, so this is fetched and filtered down
# separately below rather than added to MODELS.
NOMIC_CODE_REPO = "nomic-ai/nomic-bert-2048"
NOMIC_TARGET = "nomic-embed-text-v1.5"


def _vendor_nomic_remote_code(dest_root: Path) -> None:
    """Copy nomic-bert-2048's modeling code into nomic-embed-text-v1.5's own
    directory and rewrite its auto_map to reference the LOCAL copy instead
    of the external repo, so it loads fully offline with no dependency on
    nomic-bert-2048 (or huggingface.co) ever again after this call."""
    import json
    import shutil
    import tempfile

    from huggingface_hub import snapshot_download

    target_dir = dest_root / NOMIC_TARGET
    if not target_dir.exists():
        return  # nomic wasn't fetched this run -- nothing to vendor into

    print(f"Fetching {NOMIC_CODE_REPO}'s modeling code for {NOMIC_TARGET}...")
    with tempfile.TemporaryDirectory() as tmp:
        code_dir = snapshot_download(repo_id=NOMIC_CODE_REPO, local_dir=tmp, allow_patterns=["*.py"])
        py_files = list(Path(code_dir).glob("*.py"))
        for py_file in py_files:
            shutil.copy(py_file, target_dir / py_file.name)
        print(f"  vendored {[f.name for f in py_files]} into {target_dir}")

    config_path = target_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    auto_map = config.get("auto_map", {})
    changed = False
    for key, value in auto_map.items():
        if "--" in value:  # "<repo>--<module>.<Class>" -> "<module>.<Class>" (local)
            auto_map[key] = value.split("--", 1)[1]
            changed = True
    if changed:
        config["auto_map"] = auto_map
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  patched {config_path}'s auto_map to reference the local copy")


def main() -> None:
    from huggingface_hub import snapshot_download

    dest_root = REPO_ROOT / "models" / "hf"
    dest_root.mkdir(parents=True, exist_ok=True)
    print(f"Writing into {dest_root}")

    for model_name, dirname in MODELS:
        local_dir = dest_root / dirname
        print(f"Fetching {model_name} -> {local_dir} ...")
        snapshot_download(repo_id=model_name, local_dir=str(local_dir), allow_patterns=ALLOW_PATTERNS)
        print(f"  done: {local_dir}")

    _vendor_nomic_remote_code(dest_root)

    print("\nAll models fetched (nomic-embed-text-v1.5 needs the `einops` package")
    print("installed to actually load -- `pip install einops` if you haven't).")


if __name__ == "__main__":
    main()
