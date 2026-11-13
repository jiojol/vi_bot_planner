"""Download and verify the published policy and pinned visual encoders."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "https://github.com/jiojol/vi_bot_planner/releases/download/v0.1.0"
CHUNK_SIZE = 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url, path, expected):
    if path.is_file() and sha256(path) == expected:
        print(f"Verified {path.relative_to(ROOT)}", flush=True)
        return

    print(f"Downloading {path.relative_to(ROOT)}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as staging:
        temporary = Path(staging) / path.name
        digest = hashlib.sha256()
        with urlopen(url, timeout=90) as response, temporary.open("wb") as output:
            for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch for {url}")
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-only", action="store_true")
    args = parser.parse_args()
    if not args.vision_only:
        config = json.loads((ROOT / "configs/navigator_config.json").read_text())
        fetch(f"{RELEASE}/model.pt", ROOT / config["checkpoint"], config["policy_sha256"])

    lock = json.loads((ROOT / "configs/vision_weights.lock.json").read_text())
    for model in lock:
        source = f"https://huggingface.co/{model['model_id']}/resolve/{model['revision']}"
        for name, expected in model["sha256"].items():
            fetch(f"{source}/{name}", ROOT / model["directory"] / name, expected)


if __name__ == "__main__":
    main()
