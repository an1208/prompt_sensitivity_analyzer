"""Create paraphrase variants for your own questions using any backend."""
from __future__ import annotations

import re
from typing import List

from .backends import Backend
from .dataset import PromptGroup

TEMPLATE = (
    "Rewrite the following text in {n} different ways. Keep the meaning exactly the same and do not add or "
    "remove information. Output one rewrite per line, with no numbering and no commentary.\n\nText: {text}"
)


def parse_lines(raw: str, original: str, n: int) -> List[str]:
    out, seen = [], {original.strip().lower()}
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip().strip('"')
        if len(line) < 3 or line.lower() in seen:
            continue
        seen.add(line.lower())
        out.append(line)
        if len(out) == n:
            break
    return out


def paraphrase_questions(backend: Backend, questions: List[str], n: int = 4, seed: int = 0, temperature: float = 0.7) -> List[PromptGroup]:
    groups = []
    for i, q in enumerate(questions, 1):
        raw = backend.generate(TEMPLATE.format(n=n, text=q), temperature=temperature, seed=seed, max_new_tokens=256)
        variants = parse_lines(raw, q, n)
        if not variants:
            print(f"[warn] no usable paraphrases for question {i}; skipped: {q[:60]!r}")
            continue
        groups.append(PromptGroup(id=f"q{i:02d}", variants=[q] + variants, category="generated"))
    return groups
