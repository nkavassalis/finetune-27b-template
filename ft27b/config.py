"""Load project.toml. Stdlib only (tomllib, Python 3.11+)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("project.toml")


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    def __getitem__(self, section: str) -> dict[str, Any]:
        return self.raw.get(section, {})

    @property
    def base(self) -> str:
        return self["model"]["base"]

    @property
    def system(self) -> str:
        return self["prompt"]["system"]

    @property
    def enable_thinking(self) -> bool:
        return bool(self["model"].get("enable_thinking", False))


def find(start: Path | None = None) -> Path:
    """project.toml in the current directory or any parent."""
    here = (start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        if (d / DEFAULT_PATH).exists():
            return d / DEFAULT_PATH
    raise FileNotFoundError("no project.toml here or in any parent directory")


def load(path: str | Path | None = None) -> Config:
    p = Path(path) if path else find()
    raw = tomllib.loads(p.read_text())
    for section, key in (("model", "base"), ("prompt", "system")):
        if key not in raw.get(section, {}):
            raise ValueError(f"{p}: [{section}] {key} is required")
    return Config(raw=raw, path=p)
