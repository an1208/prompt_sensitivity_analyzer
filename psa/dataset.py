"""Prompt sets: groups of semantically-equivalent prompt variants."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PromptGroup:
    id: str
    variants: List[str]
    category: str = "general"
    reference: Optional[str] = None  # optional gold answer, used for a reference-similarity score

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_groups(path: str | Path) -> List[PromptGroup]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data["groups"] if isinstance(data, dict) else data
    groups, seen = [], set()
    for g in raw:
        gid = g["id"]
        if gid in seen:
            raise ValueError(f"duplicate group id {gid!r}")
        seen.add(gid)
        variants = [v.strip() for v in g["variants"] if v and v.strip()]
        if len(variants) < 2:
            raise ValueError(f"group {gid!r} needs at least 2 variants")
        groups.append(PromptGroup(gid, variants, g.get("category", "general"), g.get("reference")))
    return groups


def save_groups(groups: List[PromptGroup], path: str | Path) -> None:
    Path(path).write_text(json.dumps({"groups": [g.to_dict() for g in groups]}, indent=2, ensure_ascii=False), encoding="utf-8")
