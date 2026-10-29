from pydantic import BaseModel

from crop_insurance_agent.graph.checkpoint import _schema_types, checkpoint_serde
from crop_insurance_agent.schemas.claim import ClaimDecision, ClaimEstimate
from crop_insurance_agent.schemas.weather import WeatherAssessment
from tests.factories import make_weather


class NotAllowed(BaseModel):
    x: int = 1


def test_allowlist_covers_the_state_schemas():
    names = {t.__name__ for t in _schema_types()}
    assert {"WeatherAssessment", "PolicyAssessment", "ClaimDecision", "AdvisoryNote"} <= names
    assert {ClaimEstimate, ClaimDecision, WeatherAssessment} <= set(_schema_types())


def test_schema_objects_round_trip():
    serde = checkpoint_serde()
    weather = make_weather()
    restored = serde.loads_typed(serde.dumps_typed(weather))
    assert isinstance(restored, WeatherAssessment) and restored == weather


def test_other_classes_are_not_rebuilt():
    serde = checkpoint_serde()
    restored = serde.loads_typed(serde.dumps_typed(NotAllowed()))
    assert not isinstance(restored, NotAllowed)
