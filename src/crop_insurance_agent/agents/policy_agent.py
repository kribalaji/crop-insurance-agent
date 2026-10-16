"""policy_agent: does the weather assessment trigger the policy's covers?

The LLM only reads retrieved policy text and extracts trigger conditions as
`PolicyTerms`. Everything that decides the outcome is deterministic: each
extracted quote must appear verbatim in the retrieved text, and thresholds are
compared against weather_agent's numbers in code. The LangGraph node reads
`policy_request` and `weather` from the graph state and writes `policy`.
"""

import operator
import re
from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from crop_insurance_agent.config import Settings, get_settings
from crop_insurance_agent.llm import get_llm
from crop_insurance_agent.rag.store import Chunk, PolicyStore
from crop_insurance_agent.schemas.policy import (
    Citation,
    PolicyAssessment,
    PolicyRequest,
    PolicyTerms,
    TriggerEvaluation,
    TriggerMetric,
)
from crop_insurance_agent.schemas.weather import WeatherAssessment

SYSTEM_PROMPT = """\
You extract weather trigger conditions from crop insurance policy excerpts.
Use only the excerpts provided; never rely on outside knowledge.

Supported metrics:
- rainfall_actual_mm: cumulative rainfall over the risk period, in mm.
- rainfall_deficit_pct: shortfall of seasonal rainfall below the long-period normal, as a
  positive percentage (40 means rainfall was 40% below normal).
- gdu_deviation_pct: percentage by which cumulative growing degree days exceed (positive)
  or fall short of (negative) the normal.

Rules:
- Extract a trigger only if it can be written exactly as <metric> <comparator> <threshold>
  with one of the supported metrics. Skip every other cover (prevented sowing, yield-based
  claims, hail, flood, pest and so on).
- comparator is the direction in which the condition is met: "less than 600 mm" is "<",
  "40 percent or more" is ">=", "more than 15 percent" is ">".
- exit_threshold: value of the same metric at which the maximum payout is reached, if stated.
- max_payout_pct: maximum payout as a percentage of the sum insured, if stated.
- source_quote: copy the sentence that states the trigger condition word for word.
- page: the page number shown for that excerpt.
- crops: the crops the policy covers, as written.
- risk_period_start / risk_period_end: dates of the risk period, if stated.
- sum_insured_per_ha: rupees per hectare for the claimed crop, if stated.
- Use null for anything not stated. Do not guess.
"""

# Queries used to pull the relevant parts of a policy; {crop} is filled in.
RETRIEVAL_QUERIES = (
    "{crop} deficit rainfall cover trigger exit maximum payout",
    "{crop} high temperature growing degree days cover trigger",
    "{crop} crops covered season risk period sum insured per hectare",
)

_CROP_ALIASES = {"paddy": "rice", "corn": "maize", "jowar": "sorghum"}

_COMPARE: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def _norm_crop(name: str) -> str:
    name = re.sub(r"\(.*?\)", "", name).strip().lower()
    return _CROP_ALIASES.get(name, name)


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower().rstrip(".")


def observed_value(metric: TriggerMetric, weather: WeatherAssessment) -> float | None:
    match metric:
        case "rainfall_actual_mm":
            return weather.rainfall.actual_mm
        case "rainfall_deficit_pct":
            return weather.rainfall.deficit_pct
        case "gdu_deviation_pct":
            return weather.gdu.deviation_pct


def retrieve(store: PolicyStore, policy_id: str, crop: str, k: int = 4) -> list[Chunk]:
    seen: dict[tuple[int, str], Chunk] = {}
    for q in RETRIEVAL_QUERIES:
        for c in store.search(policy_id, q.format(crop=crop), k=k):
            seen.setdefault((c.page, c.text), c)
    return sorted(seen.values(), key=lambda c: c.page)


def extract_terms(
    llm: BaseChatModel, policy_id: str, crop: str, chunks: list[Chunk]
) -> PolicyTerms:
    excerpts = "\n---\n".join(f"[page {c.page}]\n{c.text}" for c in chunks)
    result = llm.with_structured_output(PolicyTerms).invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", f"Policy: {policy_id}\nClaimed crop: {crop}\n\nExcerpts:\n{excerpts}"),
        ]
    )
    return PolicyTerms.model_validate(result)


def evaluate(
    policy_id: str, terms: PolicyTerms, weather: WeatherAssessment, chunks: list[Chunk]
) -> PolicyAssessment:
    context = _norm_text(" ".join(c.text for c in chunks))
    notes = []

    evaluations = []
    for trig in terms.triggers:
        observed = observed_value(trig.metric, weather)
        met = None if observed is None else _COMPARE[trig.comparator](observed, trig.threshold)
        verified = _norm_text(trig.source_quote) in context
        evaluations.append(
            TriggerEvaluation(trigger=trig, observed=observed, met=met, quote_verified=verified)
        )
        if not verified:
            notes.append(f"Ignored {trig.name!r}: its quoted condition is not in the policy text.")
        if observed is None:
            notes.append(f"{trig.name!r} could not be evaluated: {trig.metric} is undefined.")

    covered_crops = {_norm_crop(c) for c in terms.crops}
    crop_covered = _norm_crop(weather.crop) in covered_crops
    if not terms.crops:
        notes.append("The policy text retrieved does not list the crops covered.")
    elif not crop_covered:
        notes.append(f"{weather.crop!r} is not among the covered crops {terms.crops}.")
    period = (terms.risk_period_start, terms.risk_period_end)
    if None not in period and period != (weather.season_start, weather.season_end):
        notes.append(
            f"Weather was assessed for {weather.season_start}..{weather.season_end} but the "
            f"policy risk period is {period[0]}..{period[1]}."
        )
    if not terms.triggers:
        notes.append("No weather trigger expressible in the supported metrics was found.")

    triggered = crop_covered and any(e.met and e.quote_verified for e in evaluations)
    return PolicyAssessment(
        policy_id=policy_id,
        crop=weather.crop,
        crop_covered=crop_covered,
        sum_insured_per_ha=terms.sum_insured_per_ha,
        evaluations=evaluations,
        triggered=triggered,
        citations=[Citation(policy_id=c.policy_id, page=c.page, text=c.text) for c in chunks],
        notes=notes,
    )


def assess_policy(
    request: PolicyRequest,
    weather: WeatherAssessment,
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    store: PolicyStore | None = None,
) -> PolicyAssessment:
    settings = settings or get_settings()
    store = store or PolicyStore(settings.chroma_path)
    store.sync(settings.policies_dir)
    chunks = retrieve(store, request.policy_id, weather.crop)
    terms = extract_terms(
        llm or get_llm(settings=settings), request.policy_id, weather.crop, chunks
    )
    return evaluate(request.policy_id, terms, weather, chunks)


def policy_agent(
    state: Mapping[str, Any],
    *,
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    store: PolicyStore | None = None,
) -> dict[str, Any]:
    """LangGraph node: `policy_request` + `weather` in, `policy` (PolicyAssessment) out."""
    request = PolicyRequest.model_validate(state["policy_request"])
    weather = WeatherAssessment.model_validate(state["weather"])
    return {"policy": assess_policy(request, weather, settings, llm=llm, store=store)}
