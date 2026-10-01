import json

import pytest

from psa.backends import CallableBackend, MockBackend
from psa.cli import main
from psa.dataset import PromptGroup, load_groups
from psa.paraphrase import parse_lines
from psa.runner import generate_all


def test_bundled_dataset_valid():
    from psa.cli import DEFAULT_PROMPTS

    groups = load_groups(DEFAULT_PROMPTS)
    assert len(groups) == 10 and all(len(g.variants) == 5 for g in groups)


def test_cli_end_to_end_mock(tmp_path):
    with pytest.raises(SystemExit) as e:
        main(["run", "--backend", "mock", "--embedder", "hash", "--n-samples", "2", "--temperature", "0.7",
              "--out", str(tmp_path), "--limit", "4"])
    assert e.value.code == 0
    run = next(tmp_path.iterdir())
    for f in ["results.json", "report.md", "per_group.csv", "generations.jsonl", "groups.json", "config.json"]:
        assert (run / f).exists(), f
    res = json.loads((run / "results.json").read_text())
    assert res["overall"]["n_groups"] == 4
    assert res["overall"]["mean_within_sim"] is not None  # n_samples=2 & T>0
    assert len((run / "generations.jsonl").read_text().splitlines()) == 4 * 5 * 2

    # re-analysis of the same run works
    with pytest.raises(SystemExit) as e:
        main(["analyze", str(run), "--embedder", "hash", "--no-figures"])
    assert e.value.code == 0


def test_resume_skips_completed(tmp_path):
    groups = [PromptGroup("g", ["alpha beta", "alpha gamma"])]
    calls = []

    class Counting(MockBackend):
        def generate(self, prompt, **kw):
            calls.append(prompt)
            return super().generate(prompt, **kw)

    path = tmp_path / "gen.jsonl"
    generate_all(Counting(), groups, path, log=lambda s: None)
    assert len(calls) == 2
    generate_all(Counting(), groups, path, resume=True, log=lambda s: None)
    assert len(calls) == 2  # nothing regenerated
    generate_all(Counting(), groups, path, resume=False, log=lambda s: None)
    assert len(calls) == 4  # fresh run


def test_callable_backend_from_file(tmp_path):
    f = tmp_path / "sys.py"
    f.write_text("def answer(p):\n    return p.upper()\n")
    b = CallableBackend(f"{f}:answer")
    assert b.generate("hi") == "HI"
    with pytest.raises(ValueError):
        CallableBackend("nofunction")


def test_paraphrase_parsing():
    raw = "1. What's the capital of France?\n- Which city is France's capital?\n\n\"Name France's capital city.\"\nWhat is the capital of France?"
    got = parse_lines(raw, "What is the capital of France?", 5)
    assert got == ["What's the capital of France?", "Which city is France's capital?", "Name France's capital city."]


def test_mock_backend_determinism():
    b = MockBackend()
    assert b.generate("Explain the water cycle", temperature=0.7, seed=1) == b.generate("Explain the water cycle", temperature=0.7, seed=1)
