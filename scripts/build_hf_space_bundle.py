from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "dist" / "hf_space"
ADAPTER_DIR = ROOT / "models" / "contract-clarity-lora"
ADAPTER_BUNDLE_DIR = BUNDLE_DIR / "models" / "contract-clarity-lora"

APP_FILES = [
    "app.py",
    "README.md",
    "requirements.txt",
]

APP_DIRS = [
    "src",
]

ADAPTER_FILES = [
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
]


def copy_file(relative_path: str) -> None:
    source = ROOT / relative_path
    target = BUNDLE_DIR / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_dir(relative_path: str) -> None:
    source = ROOT / relative_path
    target = BUNDLE_DIR / relative_path
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))


def main() -> None:
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True)

    for relative_path in APP_FILES:
        copy_file(relative_path)

    for relative_path in APP_DIRS:
        copy_dir(relative_path)

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            "Missing models/contract-clarity-lora. Put the trained adapter there "
            "before building the LoRA Space bundle."
        )

    ADAPTER_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ADAPTER_FILES:
        source = ADAPTER_DIR / filename
        if source.exists():
            shutil.copy2(source, ADAPTER_BUNDLE_DIR / filename)

    print(f"Built Hugging Face Space bundle: {BUNDLE_DIR}")
    print("Upload or push the contents of this folder to your Space repo.")


if __name__ == "__main__":
    main()
