"""Check out the four authors' repositories at the audited commits, without running them."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def checkout(method, destination):
    target = destination / method["method"]
    if target.exists():
        head = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if head != method["commit"]:
            raise ValueError(f"Existing checkout has another commit; preserved: {target}")
        print(f"Existing matching checkout retained without edits: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--no-checkout", method["upstream_url"], str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", method["commit"]], check=True)
    modified = []
    if method["method"] == "spv_mia":
        for relative in ("ft_llms/llms_finetune.py", "ft_llms/refer_data_generate.py"):
            path = target / relative
            before = path.read_bytes()
            text = before.decode("utf-8")
            cleaned = re.sub(r"^os\.environ\['HTTPS?_PROXY'\] = .*\r?\n", "", text, flags=re.MULTILINE)
            if cleaned != text:
                path.write_text(cleaned, encoding="utf-8", newline="\n")
                modified.append({"file": relative, "upstream_sha256": hashlib.sha256(before).hexdigest(),
                                 "change": "removed embedded proxy settings; use environment configuration"})
    info = {"method": method["method"], "upstream": method["upstream_url"], "commit": method["commit"],
            "local_changes": modified, "experiments_executed": False}
    (target / ".paper_checkout.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Checked out {method['method']} at {method['commit']}; no experiment started")


def main():
    manifest = json.loads((ROOT / "paper_reproductions/manifest.json").read_text(encoding="utf-8"))
    names = [method["method"] for method in manifest["methods"]]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["all", *names], default="all")
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    for method in manifest["methods"]:
        if args.method in {"all", method["method"]}:
            checkout(method, args.destination.resolve())


if __name__ == "__main__":
    main()
