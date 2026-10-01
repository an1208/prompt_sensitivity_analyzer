"""Write results.json, per_group.csv, report.md and (optionally) PNG figures."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return "n/a" if math.isnan(x) else f"{x:.{nd}f}"
    return str(x)


def _table(headers: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _trunc(s: str, n: int = 80) -> str:
    s = s.replace("\n", " ⏎ ").replace("|", "\\|")
    return s if len(s) <= n else s[: n - 1] + "…"


def _figures(per_group: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    files = []

    ordered = sorted(per_group, key=lambda g: g["sensitivity"], reverse=True)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(ordered) + 1)))
    ax.barh([g["id"] for g in ordered][::-1], [g["sensitivity"] for g in ordered][::-1], color="#4C78A8")
    ax.set_xlabel("Sensitivity (1 - mean cross-paraphrase cosine similarity)")
    ax.set_title("Prompt sensitivity by group (higher = less robust)")
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity_by_group.png", dpi=150)
    plt.close(fig)
    files.append("sensitivity_by_group.png")

    n = len(per_group)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, g in zip(axes.ravel(), per_group):
        ax.axis("on")
        M = np.array(g["sim_matrix"])
        im = ax.imshow(M, vmin=max(0.0, float(M.min()) - 0.05), vmax=1.0, cmap="viridis")
        ax.set_title(g["id"], fontsize=8)
        ax.set_xticks(range(len(M)))
        ax.set_yticks(range(len(M)))
        ax.tick_params(labelsize=7)
    fig.suptitle("Output similarity between prompt variants (cosine)")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmaps.png", dpi=150)
    plt.close(fig)
    files.append("heatmaps.png")
    return files


def write_outputs(
    out_dir: Path, config: Dict[str, Any], per_group: List[Dict[str, Any]], overall: Dict[str, Any], figures: bool = True
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"config": config, "overall": overall, "groups": per_group}, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    csv_cols = ["id", "category", "n_variants", "n_samples", "sensitivity", "between_sim", "min_pair_sim", "within_sim",
                "excess_sensitivity", "lexical_sim", "exact_match_rate", "length_cv", "prompt_output_spearman",
                "most_fragile_variant", "reference_sim_mean", "reference_sim_spread"]
    with (out_dir / "per_group.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_cols, extrasaction="ignore")
        w.writeheader()
        for g in per_group:
            w.writerow({k: g.get(k) for k in csv_cols})

    figs = _figures(per_group, out_dir) if figures else []
    ordered = sorted(per_group, key=lambda g: g["sensitivity"], reverse=True)
    lo, hi = overall["sensitivity_ci95"]

    L: List[str] = ["# Prompt Sensitivity Report", "", "## Configuration", ""]
    L += [f"- **{k}**: {v}" for k, v in config.items()]
    L += ["", "## Overall", ""]
    L.append(
        _table(
            ["Metric", "Value"],
            [
                ["Groups analysed", overall["n_groups"]],
                ["Mean sensitivity (1 - cosine)", f"{_f(overall['mean_sensitivity'])}  (95% bootstrap CI {_f(lo)} - {_f(hi)})"],
                ["Median / max sensitivity", f"{_f(overall['median_sensitivity'])} / {_f(overall['max_sensitivity'])}"],
                ["Mean cross-paraphrase output similarity", _f(overall["mean_between_sim"])],
                ["Mean within-paraphrase similarity (sampling noise)", _f(overall["mean_within_sim"])],
                ["Mean excess sensitivity (within - between)", _f(overall["mean_excess_sensitivity"])],
                ["Mean lexical (Jaccard) similarity", _f(overall["mean_lexical_sim"])],
                ["Mean exact-match rate across paraphrases", _f(overall["mean_exact_match_rate"])],
                ["Prompt-sim vs output-sim Spearman (pooled)", _f(overall["pooled_prompt_output_spearman"])],
                ["Mean reference-similarity spread", _f(overall["mean_reference_spread"])],
            ],
        )
    )
    L += ["", "## Per-group results (most sensitive first)", ""]
    L.append(
        _table(
            ["Group", "Category", "Sensitivity", "Min pair sim", "Within sim", "Excess", "Lexical sim", "Exact match", "Fragile variant"],
            [
                [g["id"], g["category"], _f(g["sensitivity"]), _f(g["min_pair_sim"]), _f(g["within_sim"]),
                 _f(g["excess_sensitivity"]), _f(g["lexical_sim"]), _f(g["exact_match_rate"], 2), f"v{g['most_fragile_variant']}"]
                for g in ordered
            ],
        )
    )

    L += ["", "## Most fragile prompts", ""]
    for g in ordered[:3]:
        L += [f"### {g['id']} — sensitivity {_f(g['sensitivity'])}", ""]
        rows = []
        for vi, p in enumerate(g["prompts"]):
            rows.append([f"v{vi}", _f(g["per_variant_mean_sim"][vi]), _trunc(p, 70), _trunc(g["outputs"][vi][0], 70)])
        L.append(_table(["Var", "Mean sim to others", "Prompt", "Output (sample 0)"], rows))
        L.append("")

    if figs:
        L += ["## Figures", ""] + [f"![{f}]({f})" for f in figs] + [""]

    L += [
        "## How to read this",
        "",
        "* **Sensitivity** is `1 - mean cosine similarity` between output embeddings of *different* paraphrases. "
        "0 means outputs are semantically identical regardless of wording; larger means wording changes the answer.",
        "* **Within sim / Excess** separate wording effects from sampling randomness. Run with `--temperature 0.7 --n-samples 3` (or more) "
        "to populate them: if paraphrase outputs are no less similar than repeated samples of one prompt (excess ≈ 0), the model is robust.",
        "* **Prompt-sim vs output-sim Spearman** > 0 means prompts that are worded more differently produce more different outputs.",
        "* Embedding similarity measures *semantic closeness of outputs*, not correctness. Add a `reference` answer per group to also "
        "get `reference_sim_*` (does wording change how close the answer is to the gold answer?).",
        "* Absolute values depend on the embedder; compare models/prompts using the **same** embedder and dataset.",
        "",
    ]
    path = out_dir / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
