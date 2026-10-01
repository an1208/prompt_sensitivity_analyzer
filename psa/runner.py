"""Generation loop (resumable) and analysis glue."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .backends import Backend
from .dataset import PromptGroup
from .embedding import Embedder
from .metrics import aggregate, analyze_group


def generate_all(
    backend: Backend,
    groups: Sequence[PromptGroup],
    gen_path: Path,
    *,
    temperature: float = 0.0,
    n_samples: int = 1,
    seed: int = 0,
    max_new_tokens: int = 128,
    resume: bool = False,
    log: Callable[[str], None] = print,
) -> List[Dict[str, Any]]:
    """Generate n_samples outputs for every prompt variant. Appends to gen_path as it goes.

    The same sample seeds (seed, seed+1, ...) are used for every variant so comparisons are paired.
    """
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    done: Set[Tuple[str, int, int]] = set()
    records: List[Dict[str, Any]] = []
    if gen_path.exists():
        if resume:
            for line in gen_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    records.append(r)
                    done.add((r["group_id"], r["variant"], r["sample"]))
            log(f"[resume] {len(done)} generations already done")
        else:
            gen_path.unlink()

    total = sum(len(g.variants) * n_samples for g in groups)
    count = len(done)
    with gen_path.open("a", encoding="utf-8") as fh:
        for g in groups:
            for vi, prompt in enumerate(g.variants):
                for si in range(n_samples):
                    if (g.id, vi, si) in done:
                        continue
                    out = backend.generate(prompt, temperature=temperature, seed=seed + si, max_new_tokens=max_new_tokens)
                    rec = {"group_id": g.id, "variant": vi, "sample": si, "prompt": prompt, "output": out, "seed": seed + si}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    records.append(rec)
                    count += 1
                    log(f"[{count}/{total}] {g.id} v{vi} s{si}: {out[:70]!r}")
    return records


def outputs_by_group(groups: Sequence[PromptGroup], records: Sequence[Dict[str, Any]]) -> Dict[str, List[List[str]]]:
    table: Dict[Tuple[str, int], Dict[int, str]] = {}
    for r in records:
        table.setdefault((r["group_id"], r["variant"]), {})[r["sample"]] = r["output"]
    result: Dict[str, List[List[str]]] = {}
    for g in groups:
        per_variant = []
        for vi in range(len(g.variants)):
            samples = table.get((g.id, vi), {})
            per_variant.append([samples[k] for k in sorted(samples)])
        result[g.id] = per_variant
    return result


def analyze_all(
    groups: Sequence[PromptGroup], records: Sequence[Dict[str, Any]], embedder: Embedder, seed: int = 0
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    outs = outputs_by_group(groups, records)
    per_group = []
    for g in groups:
        if any(len(v) == 0 for v in outs[g.id]):
            raise ValueError(f"group {g.id!r} has missing generations - re-run with --resume")
        m = analyze_group(g.variants, outs[g.id], embedder, g.reference)
        m.update(id=g.id, category=g.category, prompts=g.variants, outputs=outs[g.id])
        per_group.append(m)
    return per_group, aggregate(per_group, seed=seed)
