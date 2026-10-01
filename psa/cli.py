"""Command line: psa {run,analyze,paraphrase,demo}."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from . import __version__
from .backends import make_backend
from .dataset import PromptGroup, load_groups, save_groups
from .embedding import make_embedder
from .paraphrase import paraphrase_questions
from .report import write_outputs
from .runner import analyze_all, generate_all

DEFAULT_PROMPTS = Path(__file__).resolve().parent.parent / "data" / "prompt_sets.json"


def _do_analysis(out_dir: Path, groups: List[PromptGroup], records, args, config) -> None:
    embedder = make_embedder(args.embedder, args.embed_model)
    config["embedder"] = embedder.name
    per_group, overall = analyze_all(groups, records, embedder, seed=args.seed)
    path = write_outputs(out_dir, config, per_group, overall, figures=not args.no_figures)
    lo, hi = overall["sensitivity_ci95"]
    print(f"\nMean sensitivity: {overall['mean_sensitivity']:.3f}  (95% CI {lo:.3f}-{hi:.3f}) over {overall['n_groups']} groups")
    print(f"Report: {path}")


def _add_analysis_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--embedder", choices=["st", "hash"], default="st", help="st = sentence-transformers (default), hash = offline lexical fallback")
    p.add_argument("--embed-model", help="sentence-transformers model (default all-MiniLM-L6-v2)")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--seed", type=int, default=0)


def _cmd_run(a: argparse.Namespace) -> int:
    groups = load_groups(a.prompts)
    if a.limit:
        groups = groups[: a.limit]
    backend = make_backend(a.backend, a.model, a.host, **({"load_in_4bit": a.load_in_4bit, "device": a.device} if a.backend == "hf" else {}))
    out_dir = Path(a.out) if a.resume and a.out else Path(a.out or "runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_groups(groups, out_dir / "groups.json")
    config = {
        "backend": backend.name, "temperature": a.temperature, "n_samples": a.n_samples, "seed": a.seed,
        "max_new_tokens": a.max_new_tokens, "prompt_file": str(a.prompts), "groups": len(groups),
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    records = generate_all(
        backend, groups, out_dir / "generations.jsonl", temperature=a.temperature, n_samples=a.n_samples,
        seed=a.seed, max_new_tokens=a.max_new_tokens, resume=a.resume,
    )
    _do_analysis(out_dir, groups, records, a, config)
    print(f"Run folder: {out_dir}")
    return 0


def _cmd_analyze(a: argparse.Namespace) -> int:
    d = Path(a.run_dir)
    groups = load_groups(d / "groups.json")
    records = [json.loads(l) for l in (d / "generations.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    config = json.loads((d / "config.json").read_text(encoding="utf-8")) if (d / "config.json").exists() else {}
    _do_analysis(d, groups, records, a, config)
    return 0


def _cmd_paraphrase(a: argparse.Namespace) -> int:
    src = Path(a.questions)
    questions = [q.strip() for q in src.read_text(encoding="utf-8").splitlines() if q.strip()]
    backend = make_backend(a.backend, a.model, a.host)
    groups = paraphrase_questions(backend, questions, n=a.n, seed=a.seed)
    save_groups(groups, a.out)
    print(f"Wrote {len(groups)} groups to {a.out}. Review them - paraphrases from small models can drift in meaning!")
    return 0


def _cmd_demo(a: argparse.Namespace) -> int:
    a.backend, a.model, a.host = "mock", None, None
    a.prompts, a.limit = DEFAULT_PROMPTS, None
    a.temperature, a.n_samples, a.max_new_tokens = 0.7, 3, 64
    a.embedder, a.embed_model, a.load_in_4bit, a.device, a.resume = "hash", None, False, None, False
    print("DEMO: mock backend + hashing embedder. Numbers illustrate the pipeline only; they say nothing about real models.\n")
    return _cmd_run(a)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="psa", description="Prompt Sensitivity Analyser")
    p.add_argument("--version", action="version", version=f"psa {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="generate outputs for all prompt variants and analyse them")
    r.add_argument("--backend", choices=["hf", "ollama", "callable", "mock"], required=True)
    r.add_argument("--model", help="HF repo id / Ollama tag / 'module:function' or 'file.py:function' for callable")
    r.add_argument("--host", help="Ollama host")
    r.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--n-samples", type=int, default=1, help="samples per variant; >=2 with temperature>0 enables the sampling-noise baseline")
    r.add_argument("--max-new-tokens", type=int, default=128)
    r.add_argument("--limit", type=int, help="only the first N groups")
    r.add_argument("--out", help="output base dir (default runs/); with --resume: the exact run dir to continue")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--load-in-4bit", action="store_true", help="HF backend: 4-bit quantisation (needs bitsandbytes + GPU)")
    r.add_argument("--device", help="HF backend device (cpu / cuda)")
    _add_analysis_args(r)
    r.set_defaults(fn=_cmd_run)

    an = sub.add_parser("analyze", help="re-analyse an existing run folder (e.g. with a different embedder)")
    an.add_argument("run_dir")
    _add_analysis_args(an)
    an.set_defaults(fn=_cmd_analyze)

    pa = sub.add_parser("paraphrase", help="create a prompt-set JSON from a text file of questions (one per line)")
    pa.add_argument("questions")
    pa.add_argument("--backend", choices=["hf", "ollama", "callable", "mock"], default="ollama")
    pa.add_argument("--model")
    pa.add_argument("--host")
    pa.add_argument("-n", type=int, default=4, help="paraphrases per question")
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--out", default="my_prompts.json")
    pa.set_defaults(fn=_cmd_paraphrase)

    d = sub.add_parser("demo", help="offline end-to-end demo (mock model + hashing embedder)")
    d.add_argument("--out")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--no-figures", action="store_true")
    d.set_defaults(fn=_cmd_demo)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        code = args.fn(args)
    except (ValueError, ImportError, RuntimeError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
