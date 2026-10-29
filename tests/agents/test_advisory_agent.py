from datetime import date

import pytest

from crop_insurance_agent.agents.advisory_agent import (
    advisory_agent,
    build_facts,
    template_advisory,
    unsupported_numbers,
    write_advisory,
)
from crop_insurance_agent.schemas.advisory import AdvisoryDraft
from tests.factories import (
    StubLLM,
    make_claim_decision,
    make_policy_assessment,
    make_weather,
)


@pytest.fixture
def facts():
    return build_facts(make_weather(), make_policy_assessment(), make_claim_decision(), None)


def test_number_check_allows_rounding_and_indian_grouping(facts):
    ok = (
        "Approved: Rs 48,132 (40.1% of your Rs 1,20,000 sum insured). You got 293 mm of rain "
        "against the usual 585.6 mm, 50% below normal, from 1 June to 30 September 2023."
    )
    assert unsupported_numbers(ok, facts) == []
    assert unsupported_numbers("You will get Rs 50,000 within 14 days.", facts) == [
        "50,000",
        "14",
    ]


def test_llm_text_is_used_when_numbers_check_out(facts):
    draft = AdvisoryDraft(
        headline="Your claim was approved for Rs 48,132.",
        explanation="Rain was 292.8 mm, well below the usual 585.6 mm.",
    )
    llm = StubLLM({AdvisoryDraft: draft})
    note = write_advisory(facts, llm)
    assert note.generated_by == "llm" and note.headline == draft.headline
    ((_, (system, human)),) = llm.calls
    assert "English" in system[1] and '"payout_rs": 48132.0' in human[1]


def test_fabricated_number_falls_back_to_template(facts):
    draft = AdvisoryDraft(headline="Approved for Rs 60,500.", explanation="Paid in 7 days.")
    note = write_advisory(facts, StubLLM({AdvisoryDraft: draft}))
    assert note.generated_by == "template"
    assert note.headline == "Your claim was approved for a payout of Rs 48,132."
    assert "60,500" in note.notes[0]
    assert unsupported_numbers(f"{note.headline} {note.explanation}", facts) == []


def test_template_mentions_assumptions_rejection_and_errors():
    with_assumption = build_facts(
        make_weather(),
        make_policy_assessment(),
        make_claim_decision(assumptions=["DEFAULT ASSUMPTION ..."]),
        None,
    )
    assert "default rule" in template_advisory(with_assumption).explanation

    rejected = build_facts(
        make_weather(),
        make_policy_assessment(),
        make_claim_decision(False, comment="Duplicate"),
        None,
    )
    t = template_advisory(rejected)
    assert "not approved" in t.headline and "Duplicate" in t.explanation

    failed = build_facts(
        None, None, None, {"agent": "weather_agent", "message": "no weather files found"}
    )
    t = template_advisory(failed)
    assert "could not be assessed" in t.headline and "no weather files" in t.explanation


def test_facts_carry_dates_and_omit_missing_parts():
    facts = build_facts(make_weather(), None, None, None)
    assert set(facts) == {"weather"}
    assert facts["weather"]["season_start"] == date(2023, 6, 1)


def test_unsupported_language_is_refused():
    with pytest.raises(ValueError, match="not supported"):
        advisory_agent({"language": "ta"}, llm=StubLLM({}))
