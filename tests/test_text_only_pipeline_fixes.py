"""
Regression tests for the text-only pipeline fixes (22 Nov 2026).

1. The case text (query + patient history) must reach every agent prompt.
   Before the fix, Phase 2 built prompts from evidence cards only, so with
   tools disabled the agents never saw the case.
2. Gold/reference fields must never reach a prompt.
3. The debate must be able to change the *scored* diagnosis. Before the fix,
   Phase 5 ranked the pre-debate briefs, so the debate only produced text.
"""
from __future__ import annotations

import json

import pytest

from dermarbiter.agents.generalist import GeneralistAgent
from dermarbiter.agents.skeptic import SkepticAgent
from dermarbiter.agents.specialist import SpecialistAgent
from dermarbiter.core.blackboard import AgentBrief, BlackboardState
from dermarbiter.core.config import AgentConfig
from dermarbiter.core.debate_protocol import (
    independent_read,
    revise_briefs_after_debate,
    synthesis,
    targeted_debate,
)

HISTORY_TOKEN = "UNIQUE_CASE_HISTORY_TOKEN_7f3a"
GOLD_TOKEN = "GOLD_DIAGNOSIS_SHOULD_NEVER_LEAK"


class ScriptedRouter:
    """Stands in for ModelRouter; records prompts and answers by phase."""

    def __init__(self, initial: dict[str, list[str]], revised: dict[str, list[str]] | None):
        self.initial = initial
        self.revised = revised
        self.prompts: list[tuple[str, str]] = []

    def call(self, agent_role, messages, temperature=None, max_tokens=None, json_mode=False):
        prompt = messages[-1]["content"]
        self.prompts.append((agent_role, prompt))
        if "Reassess your position" in prompt:
            if self.revised is None:
                return "not json at all"
            dx = self.revised[agent_role]
            return json.dumps({"top3_differential": dx, "confidence": 0.8,
                               "reasoning": "updated after debate", "disagreement_flags": []})
        if "DEBATE TOPIC" in prompt:
            return f"[{agent_role}] the history argues for psoriasis"
        dx = self.initial[agent_role]
        return json.dumps({"top3_differential": dx, "confidence": 0.7,
                           "reasoning": "initial read", "cited_cards": [],
                           "disagreement_flags": []})


def _agents(router):
    def cfg(role):
        return AgentConfig(role=role, model_backend="google_api", model_name="test-model",
                           temperature=0.0, system_prompt_path="", has_tool_access=False)
    return {
        "specialist": SpecialistAgent(config=cfg("specialist"), model_router=router),
        "generalist": GeneralistAgent(config=cfg("generalist"), model_router=router),
        "skeptic": SkepticAgent(config=cfg("skeptic"), model_router=router),
    }


def _state() -> BlackboardState:
    return BlackboardState.from_case(
        {
            "case_id": "T-1",
            "query": "Itchy scaly plaques on both elbows.",
            "image_path": "/tmp/should_be_dropped.jpg",
            "clinical_history": f"6 months, family history of psoriasis {HISTORY_TOKEN}",
            "age": 34,
            "ground_truth": {"diagnosis_label": GOLD_TOKEN},
            "patient_context": {"sex": "female", "diagnosis_label": GOLD_TOKEN},
        },
        text_only=True,
    )


# ── Case presentation ─────────────────────────────────────────────────────

def test_from_case_folds_history_and_drops_image():
    state = _state()
    assert state.image_path is None
    assert state.patient_context["clinical_history"].endswith(HISTORY_TOKEN)
    assert state.patient_context["age"] == 34


def test_case_presentation_excludes_gold_fields():
    text = _state().get_case_presentation()
    assert HISTORY_TOKEN in text
    assert "Itchy scaly plaques" in text
    assert "female" in text
    assert GOLD_TOKEN not in text


def test_from_case_keeps_image_by_default():
    state = BlackboardState.from_case({"case_id": "x", "image_path": "a.jpg"})
    assert state.image_path == "a.jpg"


# ── Fix 1: case text reaches every prompt ─────────────────────────────────

def test_history_reaches_every_brief_and_argument_prompt():
    router = ScriptedRouter(
        initial={"specialist": ["eczema"], "generalist": ["psoriasis"], "skeptic": ["tinea"]},
        revised=None,
    )
    agents = _agents(router)
    state = _state()
    independent_read(state, agents)
    targeted_debate(state, agents, max_rounds=1, revise_after_debate=False)

    brief_prompts = [p for r, p in router.prompts if "DEBATE TOPIC" not in p]
    debate_prompts = [p for r, p in router.prompts if "DEBATE TOPIC" in p]
    assert len(brief_prompts) == 3 and debate_prompts
    for prompt in brief_prompts + debate_prompts:
        assert HISTORY_TOKEN in prompt
        assert GOLD_TOKEN not in prompt


# ── Fix 2: debate can change the scored diagnosis ─────────────────────────

def test_post_debate_revision_changes_final_diagnosis():
    router = ScriptedRouter(
        initial={"specialist": ["eczema"], "generalist": ["psoriasis"], "skeptic": ["eczema"]},
        revised={"specialist": ["psoriasis"], "generalist": ["psoriasis"], "skeptic": ["psoriasis"]},
    )
    agents = _agents(router)
    state = _state()
    independent_read(state, agents)
    targeted_debate(state, agents, max_rounds=1)
    synthesis(state, agents)

    assert state.initial_briefs["specialist"].top3_differential == ["eczema"]
    assert state.briefs["specialist"].top3_differential == ["psoriasis"]
    assert state.final_diagnosis[0] == "psoriasis"
    revision_prompts = [p for _, p in router.prompts if "Reassess your position" in p]
    assert len(revision_prompts) == 3
    assert all(HISTORY_TOKEN in p and "the history argues" in p for p in revision_prompts)


def test_without_revision_final_diagnosis_reflects_initial_briefs():
    router = ScriptedRouter(
        initial={"specialist": ["eczema"], "generalist": ["psoriasis"], "skeptic": ["eczema"]},
        revised={"specialist": ["psoriasis"], "generalist": ["psoriasis"], "skeptic": ["psoriasis"]},
    )
    agents = _agents(router)
    state = _state()
    independent_read(state, agents)
    targeted_debate(state, agents, max_rounds=1, revise_after_debate=False)
    synthesis(state, agents)
    assert state.final_diagnosis[0] == "eczema"
    assert state.initial_briefs == {}


def test_failed_revision_keeps_brief_and_flags_it():
    router = ScriptedRouter(
        initial={"specialist": ["eczema"], "generalist": ["psoriasis"], "skeptic": ["tinea"]},
        revised=None,  # unparseable revision output
    )
    agents = _agents(router)
    state = _state()
    independent_read(state, agents)
    targeted_debate(state, agents, max_rounds=1)
    for role in ("specialist", "generalist", "skeptic"):
        assert state.briefs[role].top3_differential == state.initial_briefs[role].top3_differential
        assert "revision_failed" in state.briefs[role].disagreement_flags


def test_revision_skipped_on_early_exit():
    router = ScriptedRouter(
        initial={"specialist": ["eczema"], "generalist": ["eczema"], "skeptic": ["eczema"]},
        revised={"specialist": ["x"], "generalist": ["x"], "skeptic": ["x"]},
    )
    agents = _agents(router)
    state = _state()
    independent_read(state, agents)
    state.early_exit = True
    targeted_debate(state, agents, max_rounds=1)
    assert state.initial_briefs == {}
    assert not any("Reassess your position" in p for _, p in router.prompts)


def test_agents_without_revise_brief_are_left_alone():
    state = _state()
    state.add_brief(AgentBrief(agent_role="specialist", top3_differential=["a"], confidence=0.5))
    from dermarbiter.core.blackboard import DebateTurn
    state.add_debate_turn(DebateTurn(round_num=1, speaker="specialist", argument="x", token_count=1))

    class Legacy:  # no revise_brief
        pass

    revise_briefs_after_debate(state, {"specialist": Legacy()})
    assert state.briefs["specialist"].top3_differential == ["a"]
    assert state.initial_briefs == {}
