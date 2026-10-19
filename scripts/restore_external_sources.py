"""Verify the archived sources, or restore their original server directory layout.

With no arguments this only checks hashes and Python syntax. Supplying
--target-root copies missing files after checking every destination for conflicts.
Existing files with different contents are never overwritten.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil


REPO_ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore(target_root=None):
    archive = REPO_ROOT / "external"
    manifest = json.loads((archive / "source_manifest.json").read_text(encoding="utf-8"))
    entries = []
    python_count = 0
    for project in manifest["projects"]:
        name = project["project"]
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid project name: {name}")
        for item in project["files"]:
            relative = PurePosixPath(item["path"])
            if relative.is_absolute() or ".." in relative.parts or "\\" in item["path"]:
                raise ValueError(f"Invalid source path: {relative}")
            source = archive / name / relative
            if source.is_symlink() or digest(source) != item["sha256"]:
                raise ValueError(f"Source checksum mismatch: {source}")
            if source.suffix == ".py":
                ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
                python_count += 1
            entries.append((source, name, relative, item["sha256"]))
    print(f"Verified {len(entries)} archived files and {python_count} Python source files.")
    if target_root is None:
        return

    target_root = Path(target_root).resolve()
    pending = []
    conflicts = []
    for source, name, relative, expected in entries:
        destination = target_root / name / relative
        if not destination.resolve().is_relative_to(target_root):
            raise ValueError(f"Destination escapes target root: {destination}")
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or not destination.is_file() or digest(destination) != expected:
                conflicts.append(str(destination))
        else:
            pending.append((source, destination))
    if conflicts:
        raise ValueError("Existing files differ; nothing was copied:\n" + "\n".join(conflicts))
    for source, destination in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    print(f"Restored {len(pending)} missing files under {target_root}; existing matching files retained.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, help="Restore project directories here; omit to verify only")
    args = parser.parse_args()
    try:
        restore(args.target_root)
    except (ValueError, OSError, SyntaxError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
