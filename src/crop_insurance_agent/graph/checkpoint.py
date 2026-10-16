"""Persistent SQLite checkpointer for claims paused at the approval step.

Deserialisation is restricted to this project's pydantic schemas (plus
LangGraph's built-in safe types), so a tampered checkpoint database cannot make
the loader instantiate arbitrary classes.
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from crop_insurance_agent.schemas import advisory, claim, policy, weather


def _schema_types() -> list[type]:
    return [
        obj
        for module in (weather, policy, claim, advisory)
        for obj in vars(module).values()
        if isinstance(obj, type)
        and issubclass(obj, BaseModel)
        and obj.__module__ == module.__name__
    ]


def checkpoint_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=_schema_types())


def sqlite_checkpointer(path: Path) -> SqliteSaver:
    """Open (creating if needed) the checkpoint database; safe across threads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn, serde=checkpoint_serde())
