"""Offline research-integrity tests. No API calls or clinical claims."""
import json
from urllib.error import HTTPError

import pytest

from dermarbiter.research.text_benchmark import (
    Completion, Config, GeminiProvider, METHOD_CALLS, VERSION, evaluate,
    parse_answer, run_case, strict_json, validate_cases, visible_case, vote,
)


def case(cid="one"):
    return {"schema_version": VERSION, "case_id": cid, "source_id": cid,
            "source_url": "https://example.org/fixture", "license": "fixture",
            "split": "test", "modality": "text", "text_origin": "fixture", "data_status": "fixture",
            "gold_origin": "fixture", "question": "FICTIONAL TEST QUESTION " + cid,
            "options": {"A": "option one", "B": "option two"}, "answer": "A",
            "explanation": "HIDDEN_ANSWER_EXPLANATION"}


def answer(label="A", confidence=0.8):
    return {"answer": label, "confidence": confidence, "rationale": "fixture explanation"}


class FakeProvider:
    def __init__(self, outputs=None, version="fixed-v1", finish="STOP", usage=None):
        self.outputs = iter(outputs) if outputs else None
        self.calls = []
        self.version, self.finish = version, finish
        self.usage = {"totalTokenCount": 20} if usage is None else usage

    def complete(self, system, prompt, config, seed):
        self.calls.append((system, prompt, seed))
        value = next(self.outputs) if self.outputs else answer()
        if isinstance(value, Exception):
            raise value
        return Completion(value if isinstance(value, str) else json.dumps(value),
                          self.version, self.usage, self.finish)


def config(method="direct"):
    return Config("model-id", "fixed-v1", method)


@pytest.mark.parametrize("method,n", METHOD_CALLS.items())
def test_method_calls_input_isolation_and_usage(method, n):
    provider = FakeProvider()
    row = run_case(case(), config(method), provider)
    assert row["status"] == "ok"
    assert row["calls_attempted"] == n
    assert row["total_tokens"] == 20*n
    for _, prompt, _ in provider.calls:
        assert "FICTIONAL TEST QUESTION" in prompt
        assert "HIDDEN_ANSWER_EXPLANATION" not in prompt
        assert "example.org" not in prompt
    assert set(visible_case(case())) == {"question", "options"}


def test_independent_votes_do_not_see_peers():
    provider = FakeProvider()
    run_case(case(), config("vote6"), provider)
    assert all("PRIOR OUTPUTS" not in prompt for _, prompt, _ in provider.calls)


def test_isolated_refinement_has_same_topology_but_no_peer_information():
    provider = FakeProvider()
    row = run_case(case(), config("isolated_refine"), provider)
    assert row["calls_attempted"] == 6
    for _, prompt, _ in provider.calls[3:]:
        prior = strict_json(prompt.split("PRIOR OUTPUTS (fallible, not new patient evidence):\n")[1])
        assert set(prior) == {"own"}


def test_seeds_pair_initial_sampling_without_claiming_determinism():
    first, second = FakeProvider(), FakeProvider()
    run_case(case(), config("debate"), first)
    run_case(case(), config("isolated_refine"), second)
    assert first.calls[:3] == second.calls[:3]


def test_debate_scores_revisions_and_freezes_initial_panel():
    provider = FakeProvider([answer("B")]*3 + [answer("A")]*3)
    row = run_case(case(), config("debate"), provider)
    assert row["initial"]["answer"] == "B"
    assert row["final"]["answer"] == "A"
    for _, prompt, _ in provider.calls[3:]:
        prior = strict_json(prompt.split("PRIOR OUTPUTS (fallible, not new patient evidence):\n")[1])
        assert prior["own"]["answer"] == "B"
        assert all(p["answer"] == "B" for p in prior["peers"])
    assert evaluate([case()], [row])["beneficial_revisions"] == 1


def test_harmful_revision_and_missing_prediction_remain_visible():
    row = run_case(case(), config("self_refine"), FakeProvider([answer(), answer("B"), answer("B")]))
    score = evaluate([case(), case("two")], [row])
    assert score["accuracy_all_cases"] == 0
    assert score["n_missing"] == 1
    assert score["harmful_revisions"] == 1


def test_correct_one_of_two_is_half_not_one():
    row = run_case(case(), config(), FakeProvider())
    assert evaluate([case(), case("two")], [row])["accuracy_all_cases"] == .5


@pytest.mark.parametrize("outputs", [["not json"], [RuntimeError("SECRET_API_KEY")]])
def test_failed_call_is_retained_without_retry_or_secret(outputs):
    provider = FakeProvider(outputs)
    row = run_case(case(), config("debate"), provider)
    assert row["status"] == "error" and row["calls_attempted"] == 1
    assert len(provider.calls) == 1
    assert "SECRET_API_KEY" not in json.dumps(row)
    assert evaluate([case()], [row])["n_error"] == 1


@pytest.mark.parametrize("provider", [FakeProvider(version="other"), FakeProvider(finish="MAX_TOKENS")])
def test_version_drift_or_truncation_invalidates_run(provider):
    row = run_case(case(), config(), provider)
    assert row["status"] == "error"
    assert row["total_tokens"] == 20


def test_missing_usage_is_unknown_not_zero():
    row = run_case(case(), config(), FakeProvider(usage={}))
    assert row["total_tokens"] is None and not row["token_usage_complete"]


def test_vote_agreement_is_not_confidence_and_tie_abstains():
    assert vote([answer(), answer("B")])["answer"] is None
    v = vote([answer(), answer(), answer("B")])
    assert v["confidence"] is None and v["agreement"] == 2/3


@pytest.mark.parametrize("raw", [
    '{"answer":"A","answer":"B","confidence":0.5,"rationale":"x"}',
    '{"answer":"A","confidence":NaN,"rationale":"x"}',
    '{"answer":"C","confidence":0.5,"rationale":"x"}',
    '{"answer":"A","confidence":true,"rationale":"x"}',
    '{"answer":"A","confidence":null,"rationale":"x"}',
    '```json\n{}\n```',
])
def test_invalid_model_json_is_not_repaired_silently(raw):
    with pytest.raises(ValueError):
        parse_answer(raw, case()["options"])


@pytest.mark.parametrize("change", [{"image_url": "image.jpg"}, {"answer": "Z"},
                                   {"options": {}}, {"text_origin": "image_caption"}])
def test_invalid_data_rejected(change):
    item = case()
    item.update(change)
    with pytest.raises(ValueError):
        validate_cases([item])


def test_split_leakage_duplicate_and_empty_rejected():
    first, second = case(), case("two")
    second.update(split="dev", source_id=first["source_id"])
    for items in ([], [first, first], [first, second]):
        with pytest.raises(ValueError):
            validate_cases(items)


def test_evaluator_rejects_unknown_duplicate_changed_input_and_mixed_arms():
    row = run_case(case(), config(), FakeProvider())
    for rows in ([row, row], [run_case(case("unknown"), config(), FakeProvider())],
                 [row, run_case(case("two"), config("structured"), FakeProvider())]):
        with pytest.raises(ValueError):
            evaluate([case(), case("two")], rows)
    altered = case()
    altered["question"] += " modified"
    with pytest.raises(ValueError):
        evaluate([altered], [row])


def test_gemini_request_and_response_contract(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({"modelVersion": "fixed-v1", "usageMetadata": {"totalTokenCount": 24},
                "candidates": [{"finishReason": "STOP", "content": {"parts": [
                    {"thought": True, "text": "internal"}, {"text": json.dumps(answer())}]}}]}).encode()
    def fake_urlopen(request, timeout):
        assert "SECRET" not in request.full_url
        assert request.get_header("X-goog-api-key") == "SECRET"
        body = json.loads(request.data)
        assert body["generationConfig"]["maxOutputTokens"] == 1024
        assert "tools" not in body
        return Response()
    monkeypatch.setattr("dermarbiter.research.text_benchmark.urlopen", fake_urlopen)
    result = run_case(case(), config(), GeminiProvider("SECRET"))
    assert result["status"] == "ok" and result["total_tokens"] == 24
    assert "internal" not in result["trace"][0]["text"]


def test_provider_errors_do_not_leak_secret(monkeypatch):
    def fail(*args, **kwargs):
        raise HTTPError("SECRET", 429, "SECRET", {}, None)
    monkeypatch.setattr("dermarbiter.research.text_benchmark.urlopen", fail)
    with pytest.raises(RuntimeError, match="provider_http_429") as exc:
        GeminiProvider("SECRET").complete("s", "p", config(), 0)
    assert "SECRET" not in str(exc.value)
