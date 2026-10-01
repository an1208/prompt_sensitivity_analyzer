import numpy as np
import pytest

from psa.embedding import HashingEmbedder
from psa.metrics import aggregate, analyze_group, bootstrap_ci, spearman

E = HashingEmbedder()


def test_identical_outputs_zero_sensitivity():
    r = analyze_group(["a b", "a c", "a d"], [["the answer is 42"]] * 3, E)
    assert r["sensitivity"] == pytest.approx(0.0, abs=1e-9)
    assert r["exact_match_rate"] == 1.0 and r["lexical_sim"] == 1.0


def test_disjoint_outputs_high_sensitivity():
    r = analyze_group(["p1", "p2", "p3"], [["alpha beta gamma"], ["delta epsilon zeta"], ["eta theta iota"]], E)
    assert r["sensitivity"] > 0.8
    assert r["exact_match_rate"] == 0.0


def test_fragile_variant_detected():
    outs = [["red green blue"], ["red green blue"], ["red green blue"], ["completely different words here"]]
    r = analyze_group(["a", "b", "c", "d"], outs, E)
    assert r["most_fragile_variant"] == 3
    assert r["most_robust_variant"] in (0, 1, 2)


def test_within_and_excess_with_multiple_samples():
    same = ["one two three", "one two three"]
    r = analyze_group(["a", "b"], [same, ["four five six", "four five six"]], E)
    assert r["within_sim"] == pytest.approx(1.0)
    assert r["excess_sensitivity"] == pytest.approx(r["within_sim"] - r["between_sim"])
    assert r["excess_sensitivity"] > 0.8


def test_no_within_with_single_sample():
    r = analyze_group(["a", "b"], [["x y"], ["x z"]], E)
    assert r["within_sim"] is None and r["excess_sensitivity"] is None


def test_reference_similarity():
    r = analyze_group(["a", "b"], [["paris is the capital"], ["banana bread recipe"]], E, reference="paris is the capital")
    assert r["reference_sim_per_variant"][0] > r["reference_sim_per_variant"][1]
    assert r["reference_sim_spread"] > 0.5


def test_input_validation():
    with pytest.raises(ValueError):
        analyze_group(["only one"], [["x"]], E)


def test_spearman_and_bootstrap():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert spearman([1, 2], [1, 2]) is None
    lo, hi = bootstrap_ci([0.1, 0.2, 0.3, 0.4], seed=1)
    assert 0.1 <= lo <= 0.25 <= hi <= 0.4
    assert bootstrap_ci([0.5]) == (0.5, 0.5)


def test_aggregate():
    g1 = analyze_group(["a", "b", "c"], [["x y"], ["x y"], ["x z"]], E)
    g2 = analyze_group(["a", "b", "c"], [["p q"], ["r s"], ["t u"]], E)
    agg = aggregate([g1, g2])
    assert agg["n_groups"] == 2
    assert g1["sensitivity"] < agg["mean_sensitivity"] < g2["sensitivity"]
    assert agg["mean_within_sim"] is None


def test_hashing_embedder_is_deterministic_and_unit_norm():
    a = E.embed(["hello world", "hello world", ""])
    assert np.allclose(a[0], a[1])
    assert np.linalg.norm(a[0]) == pytest.approx(1.0)
    assert np.linalg.norm(a[2]) == 0.0
