"""Tests for flat JSON schemas and terminal tool parameters."""

from __future__ import annotations

import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from khorek.agents.aspect import report_issues_tool, report_revalidation_tool
from khorek.agents.fixer import report_fix_result_tool
from khorek.agents.reviewer import report_review_tool
from khorek.agents.schema import flat_schema
from khorek.agents.verifier import report_verification_tool
from khorek.llm import Tool


class SampleState(StrEnum):
    """Sample enum for checking preserved schema constraints."""

    ready = "ready"
    blocked = "blocked"


class NestedSample(BaseModel):
    """Nested model with enum and numeric constraints."""

    model_config = ConfigDict(extra="forbid")

    retries: int = Field(ge=1, le=4)
    state: SampleState


class SampleReport(BaseModel):
    """Top-level model for flat-schema coverage."""

    model_config = ConfigDict(extra="forbid")

    nested: NestedSample


def _has_schema_reference(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            key in {"$ref", "$defs"} or _has_schema_reference(child) for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_has_schema_reference(item) for item in value)
    return False


def _assert_tool_parameters_are_flat(tool: Tool) -> None:
    assert not _has_schema_reference(tool.parameters)


def test_flat_schema_inlines_refs_and_preserves_constraints() -> None:
    schema = flat_schema(SampleReport)

    assert not _has_schema_reference(schema)
    assert schema["required"] == ["nested"]
    nested = schema["properties"]["nested"]
    assert nested["required"] == ["retries", "state"]
    assert nested["additionalProperties"] is False
    assert nested["properties"]["retries"]["minimum"] == 1
    assert nested["properties"]["retries"]["maximum"] == 4
    assert nested["properties"]["state"]["enum"] == ["ready", "blocked"]
    assert json.loads(json.dumps(schema)) == schema


def test_report_issues_parameters_are_ref_free() -> None:
    _assert_tool_parameters_are_flat(report_issues_tool())


def test_report_revalidation_parameters_are_ref_free() -> None:
    _assert_tool_parameters_are_flat(report_revalidation_tool())


def test_report_fix_result_parameters_are_ref_free() -> None:
    _assert_tool_parameters_are_flat(report_fix_result_tool())


def test_report_review_parameters_are_ref_free() -> None:
    _assert_tool_parameters_are_flat(report_review_tool())


def test_report_verification_parameters_are_ref_free() -> None:
    _assert_tool_parameters_are_flat(report_verification_tool())
