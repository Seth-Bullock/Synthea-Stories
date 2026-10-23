"""
A lightweight RAM monitor for long runs (Linux only; reads /proc, no extra packages).

Every `interval` seconds it appends a line like

    2026-10-23 15:41:07  RAM used 9.8 of 12.7 GB (2.9 GB available) | ollama 6.1 GB, python3 0.6 GB, ...

to a log file, so that if a Colab session runs out of memory the log (saved on
Google Drive) shows which process was growing. It also prints a warning when
available memory falls below `warn_gb`.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from pathlib import Path


def _meminfo() -> dict[str, float]:
    """Values from /proc/meminfo, in GB."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":", 1)
            info[key] = int(value.split()[0]) / 1024 / 1024
    return info


def _process_rss() -> dict[str, float]:
    """Resident memory per process name (summed over processes with the same name), in GB."""
    totals = defaultdict(float)
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/status") as f:
                fields = dict(line.split(":", 1) for line in f if ":" in line)
        except OSError:
            continue                        # process ended while we were looking
        rss = fields.get("VmRSS")
        if rss:
            totals[fields["Name"].strip()] += int(rss.split()[0]) / 1024 / 1024
    return totals


def memory_snapshot(top: int = 5) -> tuple[str, float]:
    """A one-line summary of RAM use, and the GB still available."""
    info = _meminfo()
    total, available = info["MemTotal"], info["MemAvailable"]
    biggest = sorted(_process_rss().items(), key=lambda kv: -kv[1])[:top]
    line = (f"RAM used {total - available:.1f} of {total:.1f} GB ({available:.1f} GB available) | "
            + ", ".join(f"{name} {gb:.1f} GB" for name, gb in biggest))
    return line, available


class MemoryMonitor(threading.Thread):
    def __init__(self, log_path: str | Path, interval: int = 60, warn_gb: float = 1.5):
        super().__init__(daemon=True)
        self.log_path = Path(log_path)
        self.interval = interval
        self.warn_gb = warn_gb
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                line, available = memory_snapshot()
                with self.log_path.open("a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {line}\n")
                if available < self.warn_gb:
                    print(f"WARNING: memory is running low. {line}", flush=True)
            except Exception as err:        # never let monitoring break a run
                print(f"(memory monitor: {err})", flush=True)
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()
