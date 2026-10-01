"""Sensitivity metrics.

For a group of V paraphrases of one question we collect S >= 1 output samples per paraphrase, embed all
outputs, and compute:

  between_sim   mean cosine similarity of outputs from DIFFERENT paraphrases (upper triangle of the VxV matrix)
  sensitivity   1 - between_sim          (0 = outputs identical in meaning, higher = more prompt-sensitive)
  within_sim    mean cosine similarity of repeated samples of the SAME paraphrase (needs S >= 2 and T > 0)
  excess        within_sim - between_sim  (instability attributable to wording, beyond sampling noise)
  lexical_sim   mean token-level Jaccard between outputs of different paraphrases
  exact_match   share of cross-paraphrase output pairs that are identical after normalisation
  length_cv     coefficient of variation of mean output length across paraphrases
  prompt_output_spearman   rank correlation between prompt-pair similarity and output-pair similarity
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .embedding import Embedder


def _tokset(text: str) -> frozenset:
    return frozenset(re.findall(r"[a-z0-9']+", text.lower()))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip(" .!?,;:")


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    v = np.asarray(values, float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    if len(v) == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def analyze_group(
    prompts: Sequence[str],
    outputs: Sequence[Sequence[str]],
    embedder: Embedder,
    reference: Optional[str] = None,
) -> Dict[str, Any]:
    V = len(prompts)
    if V < 2 or len(outputs) != V:
        raise ValueError("need >= 2 prompt variants and one output list per variant")

    texts: List[str] = []
    owner: List[int] = []
    for i, outs in enumerate(outputs):
        if not outs:
            raise ValueError(f"variant {i} has no outputs")
        for o in outs:
            texts.append(o)
            owner.append(i)
    owner_arr = np.array(owner)
    idx = [np.where(owner_arr == i)[0] for i in range(V)]

    E = embedder.embed(texts)
    P = embedder.embed(list(prompts))
    toks = [_tokset(t) for t in texts]
    norms = [_norm(t) for t in texts]

    S = np.ones((V, V))
    Lx = np.ones((V, V))
    Ex = np.ones((V, V))
    for i in range(V):
        for j in range(i + 1, V):
            S[i, j] = S[j, i] = float((E[idx[i]] @ E[idx[j]].T).mean())
            Lx[i, j] = Lx[j, i] = float(np.mean([jaccard(toks[a], toks[b]) for a in idx[i] for b in idx[j]]))
            Ex[i, j] = Ex[j, i] = float(np.mean([norms[a] == norms[b] for a in idx[i] for b in idx[j]]))

    iu = np.triu_indices(V, 1)
    between = float(S[iu].mean())

    within_vals = []
    for i in range(V):
        if len(idx[i]) >= 2:
            sub = E[idx[i]] @ E[idx[i]].T
            within_vals.append(float(sub[np.triu_indices(len(idx[i]), 1)].mean()))
    within = float(np.mean(within_vals)) if within_vals else None

    per_variant = [float((S[i].sum() - 1.0) / (V - 1)) for i in range(V)]
    mean_len = [float(np.mean([len(texts[a].split()) for a in idx[i]])) for i in range(V)]
    length_cv = float(np.std(mean_len) / np.mean(mean_len)) if np.mean(mean_len) > 0 else 0.0

    prompt_sim = P @ P.T
    pair_prompt = [float(v) for v in prompt_sim[iu]]
    pair_output = [float(v) for v in S[iu]]

    result: Dict[str, Any] = {
        "n_variants": V,
        "n_samples": int(np.mean([len(i) for i in idx])),
        "between_sim": between,
        "sensitivity": 1.0 - between,
        "min_pair_sim": float(S[iu].min()),
        "std_pair_sim": float(S[iu].std()),
        "within_sim": within,
        "excess_sensitivity": (within - between) if within is not None else None,
        "lexical_sim": float(Lx[iu].mean()),
        "exact_match_rate": float(Ex[iu].mean()),
        "length_cv": length_cv,
        "prompt_output_spearman": spearman(pair_prompt, pair_output),
        "per_variant_mean_sim": per_variant,
        "most_fragile_variant": int(np.argmin(per_variant)),
        "most_robust_variant": int(np.argmax(per_variant)),
        "sim_matrix": S.tolist(),
        "pair_prompt_sim": pair_prompt,
        "pair_output_sim": pair_output,
    }
    if reference:
        R = embedder.embed([reference])[0]
        ref = [float(np.mean(E[idx[i]] @ R)) for i in range(V)]
        result["reference_sim_per_variant"] = ref
        result["reference_sim_mean"] = float(np.mean(ref))
        result["reference_sim_min"] = float(np.min(ref))
        result["reference_sim_spread"] = float(np.max(ref) - np.min(ref))
    return result


def aggregate(group_metrics: Sequence[Dict[str, Any]], seed: int = 0) -> Dict[str, Any]:
    sens = [g["sensitivity"] for g in group_metrics]
    lo, hi = bootstrap_ci(sens, seed=seed)
    excess = [g["excess_sensitivity"] for g in group_metrics if g.get("excess_sensitivity") is not None]
    within = [g["within_sim"] for g in group_metrics if g.get("within_sim") is not None]
    all_p = [v for g in group_metrics for v in g["pair_prompt_sim"]]
    all_o = [v for g in group_metrics for v in g["pair_output_sim"]]
    ref_spread = [g["reference_sim_spread"] for g in group_metrics if "reference_sim_spread" in g]
    return {
        "n_groups": len(group_metrics),
        "mean_sensitivity": float(np.mean(sens)),
        "median_sensitivity": float(np.median(sens)),
        "sensitivity_ci95": [lo, hi],
        "max_sensitivity": float(np.max(sens)),
        "mean_between_sim": float(np.mean([g["between_sim"] for g in group_metrics])),
        "mean_within_sim": float(np.mean(within)) if within else None,
        "mean_excess_sensitivity": float(np.mean(excess)) if excess else None,
        "mean_lexical_sim": float(np.mean([g["lexical_sim"] for g in group_metrics])),
        "mean_exact_match_rate": float(np.mean([g["exact_match_rate"] for g in group_metrics])),
        "pooled_prompt_output_spearman": spearman(all_p, all_o),
        "mean_reference_spread": float(np.mean(ref_spread)) if ref_spread else None,
    }
