"""Paired statistics, run summaries and pilot CLI gates. Offline only."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dermarbiter.research.analysis import (
    correctness, mcnemar_exact, paired_bootstrap, summarize, to_markdown,
)
from dermarbiter.research.text_benchmark import Completion, Config, run_case
from tests.test_text_harness_v2 import dx, dx_case

ROOT = Path(__file__).resolve().parents[1]


class Scripted:
    name = "fake"

    def __init__(self, answer):
        self.answer = answer

    def complete(self, system, prompt, config, seed):
        return Completion(json.dumps(dx(self.answer)), "v1", {"totalTokenCount": 5}, "STOP")


def _run(cases, method, answer_for):
    recs = [run_case(c, Config("m", "v1", method), Scripted(answer_for(c))) for c in cases]
    return {"dir": method, "manifest": {"config": recs[0]["config"]}, "records": recs}


def test_bootstrap_and_mcnemar_basic_properties():
    a, b = [1, 0, 0, 1, 0, 0, 1, 0], [1, 1, 1, 1, 0, 1, 1, 0]
    boot = paired_bootstrap(a, b, n_boot=2000, seed=1)
    assert boot["delta"] == pytest.approx(3 / 8)
    assert boot["ci_low"] <= boot["delta"] <= boot["ci_high"]
    assert paired_bootstrap(a, b, n_boot=2000, seed=1) == boot  # seeded
    m = mcnemar_exact(a, b)
    assert (m["only_reference"], m["only_arm"]) == (0, 3)
    assert m["p_value"] == pytest.approx(0.25)
    assert mcnemar_exact(a, a)["p_value"] == 1.0
    with pytest.raises(ValueError):
        paired_bootstrap([1], [1, 0])


def test_summarize_pairs_arms_on_identical_cases():
    cases = [dx_case(f"c{i}") for i in range(6)]          # gold top-1 = Psoriasis
    direct = _run(cases, "direct", lambda c: "Eczema")
    better = _run(cases, "vote3", lambda c: "Psoriasis" if c["case_id"] < "c4" else "Eczema")
    direct["records"] = direct["records"][:5]              # one missing case
    assert correctness(cases, direct["records"]) == [0] * 6
    rows = summarize(cases, [better, direct], reference="direct", n_boot=500)
    assert [r["arm"] for r in rows] == ["direct", "vote3"]
    d, v = rows
    assert d["missing"] == 1 and d["accuracy"] == 0
    assert v["accuracy"] == pytest.approx(4 / 6) and v["calls_per_case"] == 3
    assert v["vs_direct_delta"] == pytest.approx(4 / 6)
    table = to_markdown(rows)
    assert "ref" in table and "not for publication" in table


def _cli(*args):
    # Strip API keys so these tests can never reach a real provider.
    env = {k: v for k, v in os.environ.items()
           if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY")}
    return subprocess.run([sys.executable, str(ROOT / "scripts/run_text_benchmark.py"), *args],
                          capture_output=True, text=True, cwd=ROOT, env=env)


def _write(tmp_path, cases):
    path = tmp_path / "d.jsonl"
    path.write_text("".join(json.dumps(c) + "\n" for c in cases))
    return path


def test_limit_selects_fixed_subset_and_dev_pilot_gate(tmp_path):
    cases = [{**dx_case(f"c{i}", split="dev"), "text_origin": "patient_reported_structured",
              "gold_origin": "dermatologist_panel_label", "data_status": "pilot_candidate"}
             for i in range(10)]
    path = _write(tmp_path, cases)
    plans = [json.loads(_cli("plan", str(path), "--split", "dev", "--limit", "3").stdout)
             for _ in range(2)]
    assert plans[0]["n_cases"] == 3 and plans[0]["selected_case_ids"] == plans[1]["selected_case_ids"]
    common = ["run", str(path), "--split", "dev", "--model", "m", "--expected-model-version", "m",
              "--max-calls", "99", "--output-dir", str(tmp_path / "out"), "--limit", "3"]
    refused = _cli(*common)
    assert refused.returncode != 0 and "--dev-pilot" in refused.stderr
    assert not (tmp_path / "out").exists()
    # --dev-pilot passes the data gate (then stops at the missing API key, before any output)
    passed_gate = _cli(*common, "--dev-pilot")
    assert "Data acceptance review" not in passed_gate.stderr


def test_dev_pilot_never_unlocks_test_split(tmp_path):
    cases = [{**dx_case(f"t{i}", split="test"), "text_origin": "patient_reported_structured",
              "gold_origin": "dermatologist_panel_label", "data_status": "pilot_candidate"}
             for i in range(3)]
    path = _write(tmp_path, cases)
    out = _cli("run", str(path), "--split", "test", "--model", "m", "--expected-model-version", "m",
               "--max-calls", "9", "--output-dir", str(tmp_path / "o"), "--dev-pilot")
    assert out.returncode != 0 and "Data acceptance review" in out.stderr


def test_variance_decomposition_shares_sum_and_detect_harness_effect():
    from dermarbiter.research.analysis import variance_decomposition
    cases = [dx_case(f"c{i}") for i in range(8)]
    # harness effect only: vote3 always right, direct always wrong, same for both models
    runs = []
    for model in ("m1", "m2"):
        for method, ans in (("direct", "Eczema"), ("vote3", "Psoriasis")):
            run = _run(cases, method, lambda c, a=ans: a)
            run["manifest"]["config"] = {**run["manifest"]["config"], "model": model}
            runs.append(run)
    out = variance_decomposition(cases, runs)
    parts = {k: v for k, v in out["share"].items() if k != "harness_total"}
    assert sum(parts.values()) == pytest.approx(1.0)
    assert out["share"]["harness"] == pytest.approx(1.0)
    assert out["share"]["model"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="Unbalanced"):
        variance_decomposition(cases, runs[:3])


def test_holm_and_prespecified_pairs():
    from dermarbiter.research.analysis import holm, prespecified_comparisons
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5]
    cases = [dx_case(f"c{i}") for i in range(8)]
    runs = [_run(cases, "isolated_refine", lambda c: "Eczema"),
            _run(cases, "debate", lambda c: "Psoriasis"),
            _run(cases, "vote6", lambda c: "Psoriasis")]
    rows = prespecified_comparisons(cases, runs, n_boot=200)
    assert [(r["base"], r["arm"]) for r in rows] == [("isolated_refine", "debate"), ("vote6", "debate")]
    first = rows[0]
    assert first["delta"] == pytest.approx(1.0) and first["only_arm"] == 8
    assert first["p_holm"] == pytest.approx(min(1.0, 2 * first["p_value"]))
    assert rows[1]["delta"] == 0 and rows[1]["p_holm"] == 1.0
