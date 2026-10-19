"""Export only the four requested MIA methods and their honest handoff status."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
METHODS = {"rag_mia": ("RAG_MIA", "RAGMIAAttack"), "neighbour_mia": ("Neighbour_MIA", "NeighbourMIAAttack"),
           "spv_mia": ("SPV_MIA", "SPVMIAAttack"), "dcmi": ("DCMI", "DCMIAttack")}


def export(output):
    files = {}

    def include(relative):
        files[relative] = (ROOT / relative).read_bytes()

    for relative in ("run_attack.py", "requirements.txt", ".gitattributes", ".gitignore", "attacks/base.py", "attacks/__init__.py",
                     "docs/server_runtime_versions.json", "scripts/export_four_paper_code.py",
                     "scripts/restore_external_sources.py", "scripts/checkout_paper_source.py"):
        include(relative)
    for path in (ROOT / "utils").glob("*.py"):
        include(path.relative_to(ROOT).as_posix())
    for method in METHODS:
        include(f"attacks/{method}.py")
        include(f"configs/{method}.yml")
    source_manifest = json.loads((ROOT / "external/source_manifest.json").read_text(encoding="utf-8"))
    source_manifest["projects"] = [project for project in source_manifest["projects"]
                                   if project["project"] in {item[0] for item in METHODS.values()}]
    for project in source_manifest["projects"]:
        for item in project["files"]:
            relative = f"external/{project['project']}/{item['path']}"
            include(relative)
            if hashlib.sha256(files[relative]).hexdigest() != item["sha256"]:
                raise ValueError(f"Archived source checksum mismatch: {relative}")
    files["external/source_manifest.json"] = (json.dumps(source_manifest, indent=2, ensure_ascii=False) + "\n").encode()
    for path in (ROOT / "paper_reproductions").rglob("*"):
        if path.is_file() and path.suffix in {".md", ".json", ".py"} and "__pycache__" not in path.parts:
            include(path.relative_to(ROOT).as_posix())
    for reference in json.loads(files["paper_reproductions/reference_entries.json"]):
        raw = files["paper_reproductions/" + reference["reference_file"]]
        if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError("Original author reference checksum mismatch")

    # The teacher package has a registry containing exactly these four methods.
    registry = "\n".join(f"from .{method} import {class_name}" for method, (_, class_name) in METHODS.items())
    registry += "\n\nATTACK_REGISTRY = {\n" + "\n".join(f"    {method!r}: {class_name}," for method, (_, class_name) in METHODS.items()) + "\n}\n\n"
    registry += "def get_attack(attack_name, config):\n    return ATTACK_REGISTRY[attack_name.lower()](config)\n"
    files["attacks/registry.py"] = registry.encode()
    files["README.md"] = ("# Four-paper MIA code handoff\n\n"
                          "Start with [the per-paper status and code guide](paper_reproductions/README.md).\n\n"
                          "The archive contains the author's code adaptations and pinned original entry references. "
                          "It does not claim that all four paper experiments have been reproduced. "
                          "Data, checkpoints and dependencies must be prepared separately.\n\n"
                          "```bash\npython -m pip install -r requirements.txt\n"
                          "python scripts/restore_external_sources.py\npython run_attack.py --help\n```\n").encode()
    for name, raw in files.items():
        if name.endswith(".py"):
            ast.parse(raw.decode("utf-8-sig"), filename=name)
    forbidden = [name for name in files if "/Janus/" in name or name.endswith((".pt", ".pth", ".bin", ".pkl", ".safetensors", ".jsonl"))]
    if forbidden:
        raise ValueError(f"Unexpected asset in code delivery: {forbidden}")
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(files.items())}
    files["DELIVERY_SHA256.json"] = (json.dumps(hashes, indent=2) + "\n").encode()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(files.items()):
            archive.writestr(name, raw)
    print(f"Exported {len(files)} code/config/documentation files to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    export(parser.parse_args().output)


if __name__ == "__main__":
    main()
