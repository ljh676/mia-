"""子进程运行工具。"""

import os
import subprocess
import traceback
from typing import Dict, List


def run_with_log(command: List[str], cwd: str, log_path: str, timeout: int = 3600) -> Dict[str, object]:
    """运行原论文命令，并把 stdout/stderr/traceback 保存到日志文件。"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    joined = " ".join(command)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"[COMMAND] {joined}\n")
        log.write(f"[CWD] {cwd}\n\n")
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            log.write("[STDOUT]\n")
            log.write(completed.stdout or "")
            log.write("\n[STDERR]\n")
            log.write(completed.stderr or "")
            return {
                "returncode": completed.returncode,
                "stdout": completed.stdout or "",
                "stderr": completed.stderr or "",
                "command": joined,
            }
        except Exception:
            tb = traceback.format_exc()
            log.write("[TRACEBACK]\n")
            log.write(tb)
            return {"returncode": -1, "stdout": "", "stderr": tb, "command": joined}
