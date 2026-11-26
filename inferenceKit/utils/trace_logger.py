import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class TraceLogger:
    """
    Logs events to a JSONL file.
    """

    def __init__(self, path: str, enabled: bool = True, truncate: bool = True):
        self.enabled = enabled
        self.path = Path(path)
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Only truncate if truncate=True
            if truncate:
                with self.path.open("w") as f:
                    f.write("")

    def log(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        # NB: keep it flat-ish so future parsing is easy
        with self.path.open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    @classmethod
    def from_env_or_default(
        cls,
        default_path: str,
        env_var: str = "DPTS_TRACE_PATH",
        enabled_env: str = "DPTS_TRACE_ENABLED",
        truncate: bool = True,
    ) -> "TraceLogger":
        """
        Convenience: you can set DPTS_TRACE_ENABLED=0 to disable,
        DPTS_TRACE_PATH=/some/file.jsonl to override path.
        """
        enabled = os.environ.get(enabled_env, "1") not in ("0", "false", "False")
        path = os.environ.get(env_var, default_path)
        return cls(path, enabled=enabled, truncate=truncate)

