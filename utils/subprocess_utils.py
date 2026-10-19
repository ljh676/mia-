"""Run external methods without accepting failures or stale output files."""

import os
from pathlib import Path
import subprocess


RESULT_SUFFIXES = {".json", ".pkl", ".pickle", ".npy", ".txt"}


def _state(path):
    try:
        stat = Path(path).stat()
        return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size
    except FileNotFoundError:
        return None


def _files(paths, directories):
    result = {str(Path(p).resolve()) for p in paths or [] if p}
    for directory in directories or []:
        root = Path(directory)
        if root.is_dir():
            result.update(str(p.resolve()) for p in root.rglob("*") if p.is_file() and p.suffix in RESULT_SUFFIXES)
    return result


def require_fresh_output(path, run_info):
    resolved = str(Path(path).resolve())
    if resolved not in run_info["fresh_outputs"]:
        raise RuntimeError(f"External method did not produce a fresh result: {resolved}; see {run_info['log_path']}")
    return resolved


def run_with_log(command, cwd, log_path, timeout=3600, output_paths=None, output_dirs=None, env=None):
    # Relative output paths are relative to the caller, as in existing configs.
    before = {p: _state(p) for p in _files(output_paths, output_dirs)}
    log_path = os.path.abspath(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"[COMMAND] {command!r}\n[CWD] {cwd}\n")
        try:
            completed = subprocess.run(command, cwd=cwd, env=env, text=True, encoding="utf-8", errors="replace",
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.write(f"[FAILED] {exc}\n")
            raise RuntimeError(f"External method could not complete; see {log_path}") from exc
        log.write("[STDOUT]\n" + completed.stdout + "\n[STDERR]\n" + completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"External method exited with code {completed.returncode}; see {log_path}")
    fresh = [p for p in _files(output_paths, output_dirs) if _state(p) is not None and _state(p) != before.get(p)]
    return {"returncode": 0, "stdout": completed.stdout, "stderr": completed.stderr,
            "command": " ".join(command), "log_path": log_path, "fresh_outputs": fresh}
