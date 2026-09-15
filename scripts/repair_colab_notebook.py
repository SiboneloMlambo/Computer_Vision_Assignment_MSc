"""Apply the canonical, idempotent data-download cell to the Colab notebook."""
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "colab_reproduction.ipynb"

DATA_CELL = '''# Download DA-2K and DIODE validation data.
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

!pip install -q huggingface_hub
from huggingface_hub import hf_hub_download

# --- DA-2K ---
# Extract into a temporary directory first. The repository already contains
# data/da2k/README.md, so moving a folder directly to data/da2k would create
# an incorrect nested directory.
data_root = Path("data")
target = data_root / "da2k"
staging = data_root / "da2k_extracted"
target.mkdir(parents=True, exist_ok=True)

if staging.exists():
    shutil.rmtree(staging)
staging.mkdir(parents=True)

da2k_zip = hf_hub_download(
    repo_id="depth-anything/DA-2K",
    filename="DA-2K.zip",
    repo_type="dataset",
)
with zipfile.ZipFile(da2k_zip) as archive:
    archive.extractall(staging)

annotation_files = list(staging.rglob("annotations.json"))
assert len(annotation_files) == 1, (
    f"Expected one annotations.json, found {len(annotation_files)}: "
    f"{annotation_files}"
)
dataset_root = annotation_files[0].parent

# Merge the actual dataset root into the existing data/da2k directory.
for source in dataset_root.iterdir():
    destination = target / source.name
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)

shutil.rmtree(staging)
assert (target / "annotations.json").is_file()
print("DA-2K ready:", sorted(p.name for p in target.iterdir()))

# --- DIODE validation split ---
diode_root = data_root / "diode"
diode_root.mkdir(parents=True, exist_ok=True)
diode_archive = Path("diode_val.tar.gz")
!wget -q --show-progress -O diode_val.tar.gz "https://diode-dataset.s3.amazonaws.com/val.tar.gz"

assert diode_archive.is_file() and diode_archive.stat().st_size > 0, (
    "DIODE download failed or produced an empty archive"
)
with tarfile.open(diode_archive, "r:gz") as archive:
    archive.extractall(diode_root)
diode_archive.unlink()

assert (diode_root / "val").is_dir(), (
    "data/diode/val was not found after extraction"
)
print("DIODE val ready:", sorted(p.name for p in (diode_root / "val").iterdir()))'''

notebook = json.loads(NOTEBOOK.read_text())
notebook["cells"][7]["source"] = DATA_CELL
NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"Updated {NOTEBOOK}")
