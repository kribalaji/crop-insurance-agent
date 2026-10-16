"""advisory_agent: explain the outcome to the farmer in plain language.

Facts are assembled in code from the other agents' outputs; the LLM only words
them. Every number in the LLM's text must match a fact (allowing rounding to 0-2
decimals). If one does not, the note falls back to a deterministic template
built from the same facts and says so. The LangGraph node reads `weather`,
`policy`, `claim`, `error` and `language` from the state and writes `advisory`.
"""

import json
import re
from collections.abc import Iterator, Mapping
from datetime import date
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from crop_insurance_agent.config import Settings, get_settings
from crop_insurance_agent.llm import get_llm
from crop_insurance_agent.schemas.advisory import (
    LANGUAGE_NAMES,
    AdvisoryDraft,
    AdvisoryNote,
    Language,
)
from crop_insurance_agent.schemas.claim import MAX_TOTAL_PAYOUT_PCT, ClaimDecision
from crop_insurance_agent.schemas.policy import PolicyAssessment
from crop_insurance_agent.schemas.weather import WeatherAssessment

SYSTEM_PROMPT = """\
You explain crop insurance claim outcomes to farmers in plain, friendly {language}.
Use only the JSON facts provided. Rules:
- Every number you write must appear in the facts. Do not calculate new numbers. You may
  round to a whole number or one decimal place. Write amounts as "Rs" followed by digits.
- No jargon. Call growing degree units "heat the crop received"; call the rainfall normal
  "the usual rainfall for this season".
- Say plainly whether the claim was approved, rejected or could not be assessed.
- If `assumptions` is not empty, say the amount includes a default assumption that a
  reviewer checked, because the policy did not state the exact rule.
- If there is a reviewer comment, include its substance.
- If there is an error, explain in simple terms why the claim could not be assessed and
  that the farmer can contact their insurer or bank branch.
- headline: one sentence. explanation: three to six short sentences.
"""

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Small counting words written as digits ("2 covers", "step 1") are always allowed.
_ALWAYS_ALLOWED = {float(i) for i in range(11)}


def build_facts(
    weather: WeatherAssessment | None,
    policy: PolicyAssessment | None,
    claim: ClaimDecision | None,
    error: Mapping[str, str] | None,
) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    if weather is not None:
        facts["weather"] = {
            "crop": weather.crop,
            "season_start": weather.season_start,
            "season_end": weather.season_end,
            "data_source": weather.source.upper(),
            "rainfall_mm": weather.rainfall.actual_mm,
            "usual_rainfall_mm": weather.rainfall.normal_mm,
            "rainfall_deficit_pct": weather.rainfall.deficit_pct,
            "rainfall_category": weather.rainfall.category,
            "heat_vs_usual_pct": weather.gdu.deviation_pct,
        }
    if policy is not None:
        facts["policy"] = {
            "policy_id": policy.policy_id,
            "crop_covered": policy.crop_covered,
            "triggered": policy.triggered,
            "covers": [
                {
                    "name": e.trigger.name,
                    "observed": e.observed,
                    "trigger_threshold": e.trigger.threshold,
                    "condition_met": bool(e.met and e.quote_verified),
                }
                for e in policy.evaluations
            ],
        }
    if claim is not None:
        est = claim.estimate
        facts["claim"] = {
            "status": claim.status,
            "payout_rs": claim.payout_amount,
            "estimated_payout_rs": est.payout_amount,
            "payout_pct_of_sum_insured": est.total_payout_pct,
            "area_ha": est.area_ha,
            "sum_insured_rs": est.sum_insured_total,
            "covers_paid": [{"name": c.name, "payout_pct": c.payout_pct} for c in est.covers],
            "assumptions": est.assumptions,
            "needs_manual_assessment": est.needs_manual_assessment,
            "reviewer_comment": claim.approval.comment,
        }
    if error is not None:
        facts["error"] = {"stage": error["agent"], "reason": error["message"]}
    return facts


def _numbers(obj: Any) -> Iterator[float]:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, int | float):
        yield float(obj)
    elif isinstance(obj, date):
        yield from (obj.year, obj.month, obj.day)
    elif isinstance(obj, str):
        for m in _NUMBER.findall(obj):
            yield float(m.replace(",", ""))
    elif isinstance(obj, Mapping):
        for v in obj.values():
            yield from _numbers(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            yield from _numbers(v)


def unsupported_numbers(text: str, facts: Mapping[str, Any]) -> list[str]:
    """Numbers in `text` that no fact supports, even after rounding to 0-2 decimals."""
    allowed = set(_ALWAYS_ALLOWED) | {MAX_TOTAL_PAYOUT_PCT}
    for f in _numbers(facts):
        allowed |= {round(abs(f), k) for k in (0, 1, 2)}
    return [m for m in _NUMBER.findall(text) if round(float(m.replace(",", "")), 2) not in allowed]


def _rs(x: float) -> str:
    return f"Rs {x:,.0f}"


def template_advisory(facts: Mapping[str, Any]) -> AdvisoryDraft:
    """Deterministic wording used when the LLM text cannot be trusted."""
    w, p, c, e = (facts.get(k) for k in ("weather", "policy", "claim", "error"))
    lines = []
    if e:
        headline = "Your claim could not be assessed yet."
        lines.append(f"The {e['stage'].replace('_', ' ')} step failed: {e['reason']}")
        lines.append("Please contact your insurer or bank branch for help.")
        return AdvisoryDraft(headline=headline, explanation=" ".join(lines))

    if c and c["status"] == "approved" and c["payout_rs"] > 0:
        headline = f"Your claim was approved for a payout of {_rs(c['payout_rs'])}."
    elif c and c["status"] == "approved":
        headline = "Your claim was reviewed and no payout is due."
    else:
        headline = "Your claim was reviewed and not approved."

    if w:
        rain = f"Your {w['crop']} received {w['rainfall_mm']:g} mm of rain this season"
        rain += f", against the usual {w['usual_rainfall_mm']:g} mm."
        lines.append(rain)
    if p:
        met = [cv["name"] for cv in p["covers"] if cv["condition_met"]]
        if not p["crop_covered"]:
            lines.append("Your policy does not cover this crop.")
        elif met:
            lines.append(f"These policy conditions were met: {', '.join(met)}.")
        else:
            lines.append("None of the policy's weather conditions were met.")
    if c:
        if c["covers_paid"] and c["status"] == "approved":
            lines.append(f"The payout is {c['payout_pct_of_sum_insured']:g}% of your sum insured.")
        if c["assumptions"]:
            lines.append(
                "Part of the amount uses a default rule because the policy did not state one; "
                "a reviewer checked it."
            )
        if c["reviewer_comment"]:
            lines.append(f"Reviewer's note: {c['reviewer_comment']}")
    return AdvisoryDraft(headline=headline, explanation=" ".join(lines))


def _json_default(o: Any) -> str:
    return o.isoformat() if isinstance(o, date) else str(o)


def write_advisory(
    facts: dict[str, Any], llm: BaseChatModel, language: Language = "en"
) -> AdvisoryNote:
    result = llm.with_structured_output(AdvisoryDraft).invoke(
        [
            ("system", SYSTEM_PROMPT.format(language=LANGUAGE_NAMES[language])),
            ("human", json.dumps(facts, indent=2, default=_json_default)),
        ]
    )
    draft = AdvisoryDraft.model_validate(result)
    bad = unsupported_numbers(f"{draft.headline} {draft.explanation}", facts)
    if not bad:
        return AdvisoryNote(
            language=language, generated_by="llm", facts=facts, **draft.model_dump()
        )
    fallback = template_advisory(facts)
    return AdvisoryNote(
        language=language,
        generated_by="template",
        facts=facts,
        notes=[f"LLM text used numbers not in the facts ({', '.join(bad)}); template used."],
        **fallback.model_dump(),
    )


def _maybe(model: type, value: Any) -> Any:
    return None if value is None else model.model_validate(value)


def advisory_agent(
    state: Mapping[str, Any],
    *,
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
) -> dict[str, Any]:
    """LangGraph node: agent outputs (or `error`) in, `advisory` (AdvisoryNote) out."""
    language = state.get("language", "en")
    if language not in LANGUAGE_NAMES:
        raise ValueError(f"advisory language {language!r} is not supported yet")
    facts = build_facts(
        _maybe(WeatherAssessment, state.get("weather")),
        _maybe(PolicyAssessment, state.get("policy")),
        _maybe(ClaimDecision, state.get("claim")),
        state.get("error"),
    )
    llm = llm or get_llm(settings=settings or get_settings())
    return {"advisory": write_advisory(facts, llm, language)}
