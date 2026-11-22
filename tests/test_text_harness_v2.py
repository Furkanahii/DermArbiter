"""Closed-set dx task, DermArbiter arms, OpenAI-compatible provider, dataset builders.

Offline only: fake providers, synthetic rows. No API calls or clinical claims.
"""
import json

import pytest

from dermarbiter.research import datasets as ds
from dermarbiter.research.text_benchmark import (
    DX_VERSION, METHOD_CALLS, PANEL, PERSONAS, PRIOR_HEADER, VERSION, Completion, Config,
    OpenAICompatibleProvider, borda_vote, evaluate, parse_differential, run_case,
    strict_json, validate_cases, visible_case,
)

CANDS = ["Eczema", "Psoriasis", "Tinea", "Urticaria"]


def dx_case(cid="d1", weights=None, split="test"):
    return {"schema_version": DX_VERSION, "case_id": cid, "source_id": cid,
            "source_url": "https://example.org/fixture", "license": "fixture",
            "split": split, "modality": "text", "text_origin": "fixture",
            "gold_origin": "fixture", "data_status": "fixture",
            "case_text": "FICTIONAL SELF REPORT " + cid, "candidates": CANDS,
            "gold_weights": weights or {"Psoriasis": 0.617, "Eczema": 0.383},
            "metadata": {"combined_race": "HIDDEN_METADATA"}}


def dx(*conds, confidence=0.6):
    return {"differential": list(conds), "confidence": confidence if conds else None,
            "rationale": "fixture"}


class Fake:
    """Returns scripted outputs; by default every call says Psoriasis, Eczema."""
    name = "fake"

    def __init__(self, outputs=None):
        self.outputs = iter(outputs) if outputs else None
        self.calls = []

    def complete(self, system, prompt, config, seed):
        self.calls.append((system, prompt))
        value = next(self.outputs) if self.outputs else dx("Psoriasis", "Eczema")
        return Completion(json.dumps(value), "v1", {"totalTokenCount": 10}, "STOP")


def cfg(method="direct"):
    return Config("model", "v1", method)


# ── Closed-set parsing and aggregation ────────────────────────────────────

def test_parse_maps_case_insensitively_to_canonical_candidates():
    out = parse_differential(json.dumps(dx("psoriasis", " ECZEMA ")), CANDS)
    assert out["differential"] == ["Psoriasis", "Eczema"] and out["answer"] == "Psoriasis"


@pytest.mark.parametrize("value", [
    dx("Melanoma"),                                  # not a candidate
    dx("Eczema", "Tinea", "Psoriasis", "Urticaria"),  # more than three
    dx("Eczema", "eczema"),                          # duplicate
    {"differential": [], "confidence": 0.5, "rationale": "x"},  # abstain with confidence
    {"differential": ["Eczema"], "confidence": 0.5},            # missing key
])
def test_parse_rejects_invalid_differentials(value):
    with pytest.raises(ValueError):
        parse_differential(json.dumps(value), CANDS)


def test_borda_ranks_and_ties_abstain():
    v = borda_vote([dx("Eczema", "Tinea"), dx("Eczema"), dx("Tinea", "Eczema")])
    assert v["answer"] == "Eczema" and v["differential"][:2] == ["Eczema", "Tinea"]
    assert v["confidence"] is None and v["agreement"] == pytest.approx(2 / 3)
    assert borda_vote([dx("Eczema"), dx("Tinea")])["answer"] is None
    assert borda_vote([dx(), dx()])["answer"] is None


# ── Every arm runs on the dx task with the declared call budget ──────────

@pytest.mark.parametrize("method,n", METHOD_CALLS.items())
def test_dx_all_arms_call_budget_and_input_isolation(method, n):
    provider = Fake()
    row = run_case(dx_case(), cfg(method), provider)
    assert row["status"] == "ok" and row["task"] == "dx"
    assert row["calls_attempted"] == n and len(provider.calls) == n
    assert row["final"]["answer"] == "Psoriasis"
    for system, prompt in provider.calls:
        assert "FICTIONAL SELF REPORT" in prompt
        assert "HIDDEN_METADATA" not in prompt and "0.617" not in prompt
        assert "example.org" not in prompt
    assert set(visible_case(dx_case())) == {"case", "candidate_conditions"}


def test_role_arms_use_personas_and_generic_arms_do_not():
    provider = Fake()
    run_case(dx_case(), cfg("vote3"), provider)
    assert all("PANEL ROLE" not in s for s, _ in provider.calls)
    provider = Fake()
    run_case(dx_case(), cfg("role_vote"), provider)
    assert [s.split("PANEL ROLE: ")[1].split(".")[0] for s, _ in provider.calls] == \
        ["SPECIALIST", "GENERALIST", "SKEPTIC"]


def test_dermarbiter_moderator_decides_and_sees_both_panels():
    outputs = ([dx("Eczema")] * 3 +          # initial role panel
               [dx("Eczema")] * 3 +          # revisions
               [dx("Tinea", "Eczema")])      # moderator overrides
    provider = Fake(outputs)
    row = run_case(dx_case(), cfg("dermarbiter"), provider)
    assert row["initial"]["answer"] == "Eczema"
    assert row["panel_vote"]["answer"] == "Eczema"
    assert row["final"]["answer"] == "Tinea"
    mod_system, mod_prompt = provider.calls[-1]
    assert PERSONAS["moderator"] in mod_system
    prior = strict_json(mod_prompt.split(PRIOR_HEADER)[1])
    assert set(prior) == {"initial_panel", "revised_panel"}
    assert set(prior["initial_panel"]) == set(PANEL)


def test_role_debate_revisions_see_named_peers_not_self():
    provider = Fake()
    run_case(dx_case(), cfg("role_debate"), provider)
    for i, (_, prompt) in enumerate(provider.calls[3:]):
        prior = strict_json(prompt.split(PRIOR_HEADER)[1])
        assert set(prior["peers"]) == set(PANEL) - {PANEL[i]}
    for _, prompt in provider.calls[:3]:
        assert PRIOR_HEADER not in prompt


# ── Scoring ───────────────────────────────────────────────────────────────

def test_dx_scoring_ties_top3_soft_and_all_case_denominator():
    tied = dx_case("tied", {"Eczema": 0.5, "Tinea": 0.5})
    single = dx_case("single")                     # Psoriasis 0.617 / Eczema 0.383
    missing = dx_case("missing")
    r1 = run_case(tied, cfg(), Fake([dx("Tinea")]))            # tie member -> correct
    r2 = run_case(single, cfg(), Fake([dx("Eczema", "Psoriasis")]))  # wrong top-1, right top-3
    out = evaluate([tied, single, missing], [r1, r2])
    assert out["n_planned"] == 3 and out["n_missing"] == 1
    assert out["accuracy_all_cases"] == pytest.approx(1 / 3)
    assert out["top3_accuracy_all_cases"] == pytest.approx(2 / 3)
    assert out["top1_soft_all_cases"] == pytest.approx((0.5 + 0.383) / 3)


def test_dx_validation_rejects_unreachable_reference_bad_weights_and_mixed_schema():
    for weights in ({"Melanoma": 1.0}, {"Eczema": 0.8, "Tinea": 0.8}, {"Eczema": -1}):
        with pytest.raises(ValueError):
            validate_cases([dx_case(weights=weights)])
    mcq = {**dx_case("m"), "schema_version": VERSION, "question": "q",
           "options": {"A": "a", "B": "b"}, "answer": "A"}
    with pytest.raises(ValueError, match="Mixed"):
        validate_cases([dx_case(), mcq])


# ── OpenAI-compatible provider ────────────────────────────────────────────

def test_openai_compatible_contract(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self):
            return json.dumps({"id": "r1", "model": "served-v1",
                               "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
                               "choices": [{"finish_reason": "stop",
                                            "message": {"content": json.dumps(dx("Tinea"))}}]}).encode()

    def fake_urlopen(request, timeout):
        seen["url"], seen["auth"] = request.full_url, request.get_header("Authorization")
        seen["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr("dermarbiter.research.text_benchmark.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider("http://localhost:8000/v1/", "SECRET")
    row = run_case(dx_case(), Config("qwen", "served-v1"), provider)
    assert row["status"] == "ok" and row["total_tokens"] == 12
    assert row["provider"] == "openai_compatible"
    assert seen["url"] == "http://localhost:8000/v1/chat/completions"
    assert "SECRET" not in seen["url"] and seen["auth"] == "Bearer SECRET"
    rf = seen["body"]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    items = rf["json_schema"]["schema"]["properties"]["differential"]["items"]
    assert items["enum"] == CANDS          # decoding constrained to the candidate list
    assert row["response_schema_sha256"]
    assert seen["body"]["max_tokens"] == 1024 and "tools" not in seen["body"]
    assert "SECRET" not in json.dumps(row)


def test_openai_compatible_truncation_and_bad_url():
    with pytest.raises(ValueError):
        OpenAICompatibleProvider("localhost:8000?key=SECRET")
    p = OpenAICompatibleProvider("http://x/v1")
    assert p._FINISH["length"] == "MAX_TOKENS"


# ── Builders ──────────────────────────────────────────────────────────────

def _mcq_row(i, q=None, cop=1, choice="single", **kw):
    return {"id": f"id{i}", "question": q or f"Question number {i} about skin?",
            "opa": f"a{i}", "opb": f"b{i}", "opc": f"c{i}", "opd": f"d{i}",
            "cop": cop, "choice_type": choice, "subject_name": "Skin", "topic_name": None, **kw}


def test_medmcqa_builder_filters_shuffles_and_splits_deterministically():
    rows = [_mcq_row(i) for i in range(40)] + [
        _mcq_row(100, choice="multi"),
        _mcq_row(101, q="Identify the lesion shown in the image"),
        _mcq_row(102, cop=-1),
        _mcq_row(103, q="Same question?", cop=0), _mcq_row(104, q="Same question?", cop=2),
        {**_mcq_row(105), "subject_name": "Medicine"},
    ]
    cases, report = ds.build_medmcqa_skin(rows, revision="r", source_sha256="h")
    assert len(cases) == 40
    assert report["excluded"] == {"multi_answer": 1, "image_dependent": 1, "no_answer_key": 1,
                                  "conflicting_duplicate_keys": 2, "exact_duplicates_merged": 0}
    for c in cases:  # shuffled key still points at the published correct text
        i = c["case_id"].split("id")[1]
        assert c["options"][c["answer"]] == f"b{i}"
    again, _ = ds.build_medmcqa_skin(rows, revision="r", source_sha256="h")
    assert [c["split"] for c in cases] == [c["split"] for c in again]
    assert [c["answer"] for c in cases] == [c["answer"] for c in again]
    assert len({c["answer"] for c in cases}) > 1
    assert {"dev", "test"} == {c["split"] for c in cases}


def test_frozen_requires_reviewer():
    with pytest.raises(ValueError, match="accepted-by"):
        ds.build_medmcqa_skin([_mcq_row(1)], revision="r", source_sha256="h", data_status="frozen")
    cases, _ = ds.build_medmcqa_skin([_mcq_row(1)], revision="r", source_sha256="h",
                                     data_status="frozen", accepted_by="reviewer-1")
    assert cases[0]["data_status"] == "frozen" and cases[0]["accepted_by"] == "reviewer-1"


def _scin_case(cid, **fields):
    base = {"case_id": cid, "age_group": "AGE_30_TO_39", "sex_at_birth": "FEMALE",
            "fitzpatrick_skin_type": "FST3", "related_category": "RASH",
            "condition_duration": "ONE_TO_FOUR_WEEKS", "body_parts_arm": "YES",
            "textures_rough_or_flaky": "YES", "condition_symptoms_itching": "YES",
            "combined_race": "SECRET_RACE"}
    return {**base, **fields}


def test_scin_builder_vocabulary_eligibility_and_text():
    cases_rows = [_scin_case(str(i), body_parts_leg="YES" if i % 2 else "") for i in range(6)]
    cases_rows.append(_scin_case("sparse", condition_duration="", body_parts_arm="",
                                 textures_rough_or_flaky="", condition_symptoms_itching=""))
    labels = [{"case_id": str(i), "weighted_skin_condition_label":
               "{'Eczema': 0.34, 'Psoriasis': 0.33, 'Rare thing': 0.34}" if i == 0 else
               "{'Psoriasis': 0.7, 'Tinea': 0.3}" if i < 4 else "{'Rare thing': 1.0}"}
              for i in range(6)]
    labels += [{"case_id": "sparse", "weighted_skin_condition_label": "{'Psoriasis': 1.0}"},
               {"case_id": "nolabel", "weighted_skin_condition_label": ""}]
    out, report = ds.build_scin_text(cases_rows, labels, cases_sha256="a", labels_sha256="b", top_k=2)
    assert report["vocabulary"] == ["Psoriasis", "Rare thing"]
    ids = {c["case_id"] for c in out}
    assert "scin:sparse" not in ids                       # too few reported fields
    assert report["excluded"]["too_few_reported_fields"] == 1
    assert report["excluded"]["no_dermatologist_label"] == 1
    c0 = next(c for c in out if c["case_id"] == "scin:0")
    assert sum(c0["gold_weights"].values()) == pytest.approx(1.0)
    text = c0["case_text"]
    assert "aged 30-39" in text and "Fitzpatrick skin type III" in text and "itching" in text
    assert "SECRET_RACE" not in json.dumps(visible_case(c0))
    assert c0["metadata"]["combined_race"] == "SECRET_RACE"


def test_trivial_baselines_choose_constant_on_dev():
    cases = [dx_case(f"c{i}", split="dev" if i < 4 else "test",
                     weights={"Eczema": 1.0} if i % 2 == 0 else {"Tinea": 1.0}) for i in range(8)]
    out = ds.trivial_baselines(cases)
    assert out["constant_answer_chosen_on_dev"] in ("Eczema", "Tinea")
    assert out["test"]["constant_top1"] == 0.5
    assert out["test"]["uniform_random_top1"] == pytest.approx(0.25)


def test_response_schema_mcq_and_plain_json_fallback(monkeypatch):
    from dermarbiter.research.text_benchmark import VERSION, response_schema
    mcq = {**dx_case("m"), "schema_version": VERSION, "question": "q",
           "options": {"B": "b", "A": "a"}, "answer": "A"}
    sch = response_schema(mcq)
    assert sch["properties"]["answer"]["enum"] == ["A", "B", None]
    assert sch["properties"]["confidence"]["maximum"] == 1
    seen = {}

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self):
            return json.dumps({"model": "v1", "choices": [{"finish_reason": "stop",
                               "message": {"content": "{}"}}]}).encode()

    def fake(request, timeout):
        seen["body"] = json.loads(request.data)
        return R()
    monkeypatch.setattr("dermarbiter.research.text_benchmark.urlopen", fake)
    OpenAICompatibleProvider("http://x/v1").complete("s", "p", Config("m", "v1"), 0)
    assert seen["body"]["response_format"] == {"type": "json_object"}   # no schema given
    OpenAICompatibleProvider("http://x/v1", json_mode=False).complete("s", "p", Config("m", "v1"), 0, schema=sch)
    assert "response_format" not in seen["body"]
