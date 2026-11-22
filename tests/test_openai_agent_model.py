"""Test optional OpenAI agent reasoning without network or credentials.

These tests inject a small LangChain-compatible fake chat model into the real
adapter. They verify prompts, structured validation, citations, token/call
budgets, and error handling without importing or constructing ``ChatOpenAI``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest
from langchain_openai.chat_models.base import OpenAIInvalidRequestError
from openai import BadRequestError
from pydantic import ValidationError

from public_sector_opportunity_research.agents.contracts import (
    EvidenceRecord,
    EvidenceReference,
    ResearchAnalysis,
    ResearchEvidenceBatch,
    SynthesisAnalysis,
)
from public_sector_opportunity_research.agents.mock_llm import (
    AgentBudgetExceeded,
    AgentBudgetTracker,
)
from public_sector_opportunity_research.agents.openai_model import (
    OpenAIStructuredAgentModel,
    StructuredModelOutputError,
    StructuredModelProviderError,
    _openai_output_schema,
    _sanitized_provider_error,
)
from public_sector_opportunity_research.agents.strategy_researcher import (
    make_strategy_researcher_node,
)
from public_sector_opportunity_research.cli import build_parser


class FakeRawMessage:
    """Mimic the token metadata exposed by a LangChain AI message."""

    def __init__(self, input_tokens: int = 25, output_tokens: int = 12) -> None:
        self.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        self.response_metadata = {}


class FakeRunnable:
    """Return one queued structured result or raise a queued error."""

    def __init__(self, owner: "FakeChatModel") -> None:
        self.owner = owner

    def invoke(self, messages):
        self.owner.invocations.append(messages)
        response = self.owner.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeChatModel:
    """Implement the LangChain methods used by the provider adapter."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.schemas: list[type] = []
        self.invocations: list[list[dict[str, str]]] = []

    def with_structured_output(self, schema, *, method, include_raw, strict):
        assert method == "json_schema"
        assert include_raw is True
        assert strict is True
        self.schemas.append(schema)
        return FakeRunnable(self)


def _invalid_request_error() -> OpenAIInvalidRequestError:
    """Build the exact LangChain/OpenAI 400 exception seen by the adapter."""
    body = {
        "error": {
            "message": (
                "Unsupported schema keyword 'maxLength'; "
                "prompt=PRIVATE RESEARCH QUESTION; "
                "api_key=sk-test-secret-value"
            ),
            "type": "invalid_request_error",
            "param": "text.format.schema",
            "code": "invalid_json_schema",
        }
    }
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/responses",
        headers={"Authorization": "Bearer sk-request-secret-value"},
        content=b"PRIVATE REQUEST BODY",
    )
    response = httpx.Response(400, request=request, json=body)
    return OpenAIInvalidRequestError(
        "unsafe fallback text containing sk-fallback-secret-value",
        response=response,
        body=body,
    )


def _wrapped_invalid_request_error() -> OpenAIInvalidRequestError:
    """Wrap a detailed SDK error in the generic LangChain error seen live."""
    detailed_body = {
        "error": {
            "message": "Parameter 'text.format.schema' has an invalid definition.",
            "type": "invalid_request_error",
            "param": "text.format.schema",
            "code": "invalid_json_schema",
        }
    }
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/responses",
        headers={"Authorization": "Bearer sk-chain-secret-value"},
        content=b"PRIVATE DOCUMENT EXCERPT",
    )
    detailed_response = httpx.Response(400, request=request, json=detailed_body)
    original = BadRequestError(
        "Parameter 'text.format.schema' has an invalid definition.",
        response=detailed_response,
        body=detailed_body,
    )
    generic_response = httpx.Response(
        400,
        request=request,
        json={"error": {"message": "Error code: 400"}},
    )
    wrapper = OpenAIInvalidRequestError(
        "Error code: 400",
        response=generic_response,
        body=None,
    )
    wrapper.__cause__ = original
    return wrapper


class OneEvidenceService:
    """Return one strategy record for node-level citation enforcement."""

    def research(self, **kwargs):
        return ResearchEvidenceBatch(
            evidence=[_full_evidence()],
            retrieval_trace_ids=["trace-strategy"],
            requested_categories=list(kwargs["document_types"]),
            missing_categories=[],
        )


def _reference(
    *,
    chunk_id: str = "chunk-strategy",
    document_id: str = "document-strategy",
    document_type: str = "strategy",
    excerpt: str = "The agency requires one accessible resident portal.",
) -> EvidenceReference:
    """Create compact fictional evidence used by provider-unit tests."""
    return EvidenceReference(
        chunk_id=chunk_id,
        document_id=document_id,
        agency_id="synthetic-ndrs",
        document_title=f"Synthetic {document_type}",
        document_type=document_type,
        publication_date=date(2027, 1, 1),
        effective_date=None,
        section_heading="Priority",
        excerpt=excerpt,
        reranker_rank=1,
        retrieval_trace_id=f"trace-{document_type}",
    )


def _full_evidence() -> EvidenceRecord:
    """Create the retrieval-boundary version of the compact test evidence."""
    return EvidenceRecord(
        chunk_id="chunk-strategy",
        document_id="document-strategy",
        agency_id="synthetic-ndrs",
        agency_name="Synthetic Agency",
        document_title="Synthetic strategy",
        document_type="strategy",
        publication_date=date(2027, 1, 1),
        capture_date=date(2027, 1, 2),
        section_heading="Priority",
        text="The agency requires one accessible resident portal.",
        vector_rank=1,
        vector_similarity_score=0.8,
        reranker_rank=1,
        reranker_score=2.0,
        retrieval_trace_id="trace-strategy",
        freshness_status="not_checked",
    )


def _research_response(*, unknown_citation: bool = False) -> dict[str, object]:
    """Return one valid or deliberately invalid cited research response."""
    chunk_id = "chunk-unknown" if unknown_citation else "chunk-strategy"
    return {
        "parsed": {
            "summary": "One documented need was found.",
            "findings": [
                {
                    "finding_id": "finding-need",
                    "finding_type": "operational_need",
                    "statement": "The agency requires an accessible resident portal.",
                    "chunk_ids": [chunk_id],
                    "document_ids": ["document-strategy"],
                    "confidence": "high",
                }
            ],
        },
        "raw": FakeRawMessage(),
        "parsing_error": None,
    }


def _synthesis_response() -> dict[str, object]:
    """Return a candidate whose provider score will be recalculated."""
    return {
        "parsed": {
            "summary": "One supported candidate was identified.",
            "opportunities": [
                {
                    "opportunity_id": "opp-portal",
                    "title": "Accessible resident portal",
                    "need_summary": "A documented portal need has planning funding.",
                    "status": "potential",
                    "rank": 99,
                    "score": {
                        "mandate": 0,
                        "repeated_evidence": 0,
                        "funding": 0,
                        "commercial_readiness": 0,
                        "total": 0,
                    },
                    "commercially_actionable": False,
                    "supporting_finding_ids": ["finding-need", "finding-budget"],
                    "supporting_chunk_ids": ["chunk-strategy", "chunk-budget"],
                    "supporting_document_ids": [
                        "document-strategy",
                        "document-budget",
                    ],
                    "documented_facts": [
                        "The agency requires an accessible portal.",
                        "Planning funding is documented.",
                    ],
                    "inferences": ["The need merits procurement-status research."],
                    "commercial_implications": (
                        "Treat this as a research lead, not a known open procurement."
                    ),
                    "uncertainties": [
                        "Missing award evidence does not prove that no award exists."
                    ],
                }
            ],
        },
        "raw": FakeRawMessage(input_tokens=80, output_tokens=40),
        "parsing_error": None,
    }


def test_openai_adapter_uses_compact_isolated_prompts_and_provider_usage():
    """A successful call should contain only objective and compact evidence."""
    fake = FakeChatModel([_research_response()])
    model = OpenAIStructuredAgentModel(
        model_name="test-structured-model",
        budget=AgentBudgetTracker(),
        chat_model=fake,
    )

    invocation = model.analyze_strategy(
        objective="Assess accessible service modernization.",
        evidence=[_reference()],
        iteration=1,
    )

    assert invocation.output.findings[0].chunk_ids == ["chunk-strategy"]
    assert invocation.call.provider == "openai"
    assert invocation.call.usage_source == "provider"
    assert invocation.call.input_tokens == 25
    assert invocation.call.output_tokens == 12
    prompt_text = str(fake.invocations[0])
    assert "Assess accessible service modernization" in prompt_text
    assert "chunk-strategy" in prompt_text
    assert "The agency requires one accessible resident portal" in prompt_text
    assert "conversation_history" not in prompt_text


def test_openai_adapter_abstains_when_provider_finds_no_evidence():
    """Empty evidence can produce a valid empty finding list without invention."""
    fake = FakeChatModel(
        [
            {
                "parsed": {
                    "summary": "No supplied evidence supports a procurement finding.",
                    "findings": [],
                },
                "raw": FakeRawMessage(),
                "parsing_error": None,
            }
        ]
    )
    model = OpenAIStructuredAgentModel(
        model_name="test-structured-model",
        budget=AgentBudgetTracker(),
        chat_model=fake,
    )

    result = model.analyze_procurement(
        objective="Find confirmed awards.", evidence=[], iteration=1
    )
    assert result.output.findings == []


def test_procurement_prompt_receives_complete_award_and_requests_separate_facts():
    """Regression: vendor, executed status, value, scope, and dates reach the model."""
    source_text = Path("sample_data/06_confirmed_contract_award.txt").read_text(
        encoding="utf-8"
    )
    response = {
        "parsed": {
            "summary": "The confirmed award supports four material fact groups.",
            "findings": [
                {
                    "finding_id": f"finding-{finding_type}",
                    "finding_type": finding_type,
                    "statement": statement,
                    "chunk_ids": ["chunk-award"],
                    "document_ids": ["document-award"],
                    "confidence": "high",
                }
                for finding_type, statement in (
                    ("award_status", "The contract was executed by both parties."),
                    ("contract_scope", "The executed scope covers the portal and APIs."),
                    ("vendor", "Aurora Civic Systems is the named vendor."),
                    ("procurement_date", "The execution date is 2028-02-15."),
                )
            ],
        },
        "raw": FakeRawMessage(),
        "parsing_error": None,
    }
    fake = FakeChatModel([response])
    model = OpenAIStructuredAgentModel(
        model_name="test-structured-model",
        budget=AgentBudgetTracker(),
        chat_model=fake,
    )
    reference = _reference(
        chunk_id="chunk-award",
        document_id="document-award",
        document_type="confirmed-award",
        excerpt=source_text,
    )

    invocation = model.analyze_procurement(
        objective="Assess modernization procurement status.",
        evidence=[reference],
        iteration=1,
    )

    assert {item.finding_type for item in invocation.output.findings} == {
        "award_status",
        "contract_scope",
        "vendor",
        "procurement_date",
    }
    prompt_text = str(fake.invocations[0])
    assert "Aurora Civic" in prompt_text
    assert "$11,850,000" in prompt_text
    assert "Contract status: Executed by both fictional parties" in prompt_text
    assert "Service commencement date: 2028-02-22" in prompt_text
    assert "separate ResearchFinding records" in prompt_text
    assert "omit that finding or describe it as unknown" in prompt_text


def test_openai_synthesis_reapplies_existing_deterministic_scoring_rules():
    """Provider-proposed scores/ranks cannot replace application rules."""
    fake = FakeChatModel([_synthesis_response()])
    model = OpenAIStructuredAgentModel(
        model_name="test-structured-model",
        budget=AgentBudgetTracker(),
        chat_model=fake,
    )
    evidence = [
        _reference(),
        _reference(
            chunk_id="chunk-budget",
            document_id="document-budget",
            document_type="budget",
            excerpt="The fictional plan identifies funding for portal work.",
        ),
    ]
    findings = [
        {
            "finding_id": "finding-need",
            "finding_type": "operational_need",
            "statement": "The agency requires an accessible resident portal.",
            "chunk_ids": ["chunk-strategy"],
            "document_ids": ["document-strategy"],
            "confidence": "high",
        },
        {
            "finding_id": "finding-budget",
            "finding_type": "budget_evidence",
            "statement": "Planning funding supports portal work.",
            "chunk_ids": ["chunk-budget"],
            "document_ids": ["document-budget"],
            "confidence": "high",
        },
    ]

    from public_sector_opportunity_research.agents.contracts import ResearchFinding

    invocation = model.synthesize(
        objective="Identify grounded opportunities.",
        findings=[ResearchFinding.model_validate(item) for item in findings],
        evidence=evidence,
        iteration=1,
    )
    candidate = invocation.output.opportunities[0]
    prompt_text = str(fake.invocations[0])
    assert "confirmed award-status" in prompt_text
    assert "must cite both confirmed executed status" in prompt_text
    assert candidate.rank == 1
    assert candidate.score.model_dump() == {
        "mandate": 3,
        "repeated_evidence": 2,
        "funding": 3,
        "commercial_readiness": 1,
        "total": 9,
    }
    assert candidate.commercially_actionable is True


def test_openai_adapter_rejects_malformed_structured_output():
    """Missing required fields should become a narrow output error."""
    fake = FakeChatModel(
        [{"parsed": {"findings": []}, "raw": FakeRawMessage(), "parsing_error": None}]
    )
    model = OpenAIStructuredAgentModel(
        model_name="test-structured-model",
        budget=AgentBudgetTracker(),
        chat_model=fake,
    )
    with pytest.raises(StructuredModelOutputError):
        model.analyze_strategy(objective="test", evidence=[], iteration=1)


def test_researcher_node_rejects_provider_citation_outside_evidence():
    """Schema-valid but unknown provider citations must not enter graph state."""
    fake = FakeChatModel([_research_response(unknown_citation=True)])
    model = OpenAIStructuredAgentModel(
        model_name="test-structured-model",
        budget=AgentBudgetTracker(),
        chat_model=fake,
    )
    node = make_strategy_researcher_node(OneEvidenceService(), model)
    state = {
        "agency_id": "synthetic-ndrs",
        "objective": "Assess modernization.",
        "iteration": 1,
        "tasks": [
            {
                "task_id": "iteration-1-strategy",
                "role": "strategy",
                "question": "Find needs.",
                "document_types": ["strategy"],
                "iteration": 1,
            }
        ],
    }
    with pytest.raises(ValueError, match="Unknown strategy chunk citation"):
        node(state)


def test_post_response_budget_overrun_records_actual_provider_usage():
    """A completed paid response must remain in accounting when it exceeds budget."""
    response = _research_response()
    response["raw"] = FakeRawMessage(input_tokens=60000, output_tokens=123)
    fake = FakeChatModel([response])
    model = OpenAIStructuredAgentModel(
        model_name="test-structured-model",
        budget=AgentBudgetTracker(max_input_tokens=50000),
        chat_model=fake,
    )
    node = make_strategy_researcher_node(OneEvidenceService(), model)
    update = node(
        {
            "agency_id": "synthetic-ndrs",
            "objective": "Assess modernization.",
            "iteration": 1,
            "tasks": [
                {
                    "task_id": "iteration-1-strategy",
                    "role": "strategy",
                    "question": "Find needs.",
                    "document_types": ["strategy"],
                    "iteration": 1,
                }
            ],
        }
    )

    assert len(fake.invocations) == 1
    assert update["budget_exceeded"][0]["projected_tokens"] == 60000
    assert update["budget_exceeded"][0]["limit"] == 50000
    assert update["model_calls"] == [
        {
            "call_id": "1:strategy_researcher",
            "stage": "strategy_researcher",
            "provider": "openai",
            "model_name": "test-structured-model",
            "input_tokens": 60000,
            "output_tokens": 123,
            "usage_source": "provider",
                "status": "succeeded",
                "retry_count": 0,
                "error_category": None,
        }
    ]
    assert "Provider response accounted" in update["execution_trace"][0]["detail"]


def test_openai_adapter_checks_call_budget_before_second_provider_call():
    """A paid call must not start after the workflow call limit is exhausted."""
    fake = FakeChatModel([_research_response(), _research_response()])
    model = OpenAIStructuredAgentModel(
        model_name="test-structured-model",
        budget=AgentBudgetTracker(max_calls=1),
        chat_model=fake,
    )
    model.analyze_strategy(objective="test", evidence=[_reference()], iteration=1)
    with pytest.raises(AgentBudgetExceeded, match="call budget exceeded"):
        model.analyze_procurement(
            objective="test", evidence=[_reference()], iteration=1
        )
    assert len(fake.invocations) == 1


def test_openai_adapter_wraps_provider_errors_without_secret_values():
    """Transport/provider failures should become a stable application error."""
    fake = FakeChatModel([RuntimeError("simulated provider outage")])
    model = OpenAIStructuredAgentModel(
        model_name="test-structured-model",
        budget=AgentBudgetTracker(),
        chat_model=fake,
    )
    with pytest.raises(StructuredModelProviderError, match="RuntimeError"):
        model.analyze_strategy(objective="test", evidence=[], iteration=1)


def test_provider_schemas_remove_unsupported_string_lengths_only():
    """Provider schemas should avoid the strict-schema 400 regression.

    Application Pydantic models retain their length checks. Only the JSON sent
    to OpenAI has unsupported ``minLength``/``maxLength`` keys removed.
    """
    for model_type in (ResearchAnalysis, SynthesisAnalysis):
        provider_schema = _openai_output_schema(model_type)
        serialized = str(provider_schema["schema"])
        assert "minLength" not in serialized
        assert "maxLength" not in serialized
        assert "additionalProperties" in serialized
        assert "minItems" in serialized

    with pytest.raises(ValidationError):
        ResearchAnalysis.model_validate({"summary": "", "findings": []})


@pytest.mark.parametrize(
    ("stage", "invoke"),
    [
        (
            "strategy_researcher",
            lambda model: model.analyze_strategy(
                objective="private strategy question", evidence=[], iteration=1
            ),
        ),
        (
            "procurement_researcher",
            lambda model: model.analyze_procurement(
                objective="private procurement question", evidence=[], iteration=1
            ),
        ),
        (
            "synthesis_agent",
            lambda model: model.synthesize(
                objective="private synthesis question",
                findings=[],
                evidence=[],
                iteration=1,
            ),
        ),
    ],
)
def test_openai_400_diagnostics_are_sanitized_for_every_agent(stage, invoke):
    """All live agent calls should expose useful 400 fields without content."""
    fake = FakeChatModel([_invalid_request_error()])
    model = OpenAIStructuredAgentModel(
        model_name="gpt-6-luna",
        budget=AgentBudgetTracker(),
        chat_model=fake,
    )

    with pytest.raises(StructuredModelProviderError) as caught:
        invoke(model)

    error = caught.value
    assert error.stage == stage
    assert error.status_code == 400
    assert error.error_code == "invalid_json_schema"
    assert error.error_type == "invalid_request_error"
    assert error.parameter == "text.format.schema"
    assert "Unsupported schema keyword 'maxLength'" in str(error)
    assert "prompt=[REDACTED]" in str(error)
    assert "PRIVATE RESEARCH QUESTION" not in str(error)
    assert "sk-test-secret-value" not in str(error)
    assert "sk-request-secret-value" not in str(error)
    assert "sk-fallback-secret-value" not in str(error)
    assert "PRIVATE REQUEST BODY" not in str(error)


def test_diagnostics_follow_langchain_cause_to_original_openai_error():
    """A generic LangChain wrapper must not hide the original safe API fields."""
    error = _sanitized_provider_error(
        _wrapped_invalid_request_error(),
        stage="procurement_researcher",
    )

    assert error.status_code == 400
    assert error.error_code == "invalid_json_schema"
    assert error.error_type == "invalid_request_error"
    assert error.parameter == "text.format.schema"
    assert error.safe_message == (
        "Parameter 'text.format.schema' has an invalid definition."
    )
    assert "sk-chain-secret-value" not in str(error)
    assert "PRIVATE DOCUMENT EXCERPT" not in str(error)


def test_diagnostics_select_error_fields_from_response_json_fallback():
    """Useful fields may survive only in response JSON on some SDK versions."""
    body = {
        "error": {
            "message": "The selected model rejected one request parameter.",
            "type": "invalid_request_error",
            "param": "reasoning.effort",
            "code": "unsupported_value",
            "private_extra": "MUST NOT BE EXPOSED",
        },
        "unrelated": "COMPLETE RESPONSE MUST NOT BE EXPOSED",
    }
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/responses",
        headers={"Authorization": "Bearer sk-response-secret-value"},
        content=b"PRIVATE PROMPT AND EXCERPT",
    )
    response = httpx.Response(400, request=request, json=body)
    wrapper = OpenAIInvalidRequestError(
        "Error code: 400",
        response=response,
        body=None,
    )

    error = _sanitized_provider_error(wrapper, stage="standalone")

    assert error.error_code == "unsupported_value"
    assert error.error_type == "invalid_request_error"
    assert error.parameter == "reasoning.effort"
    assert error.safe_message == "The selected model rejected one request parameter."
    serialized = str(error)
    assert "MUST NOT BE EXPOSED" not in serialized
    assert "COMPLETE RESPONSE" not in serialized
    assert "sk-response-secret-value" not in serialized
    assert "PRIVATE PROMPT AND EXCERPT" not in serialized


def test_graph_cli_requires_explicit_openai_provider_selection():
    """Mock remains default and OpenAI/model overrides are explicit flags."""
    parser = build_parser()
    default_args = parser.parse_args(
        ["run-agent-graph", "--agency-id", "synthetic-ndrs", "--objective", "test"]
    )
    live_args = parser.parse_args(
        [
            "run-agent-graph",
            "--agency-id",
            "synthetic-ndrs",
            "--objective",
            "test",
            "--model-provider",
            "openai",
            "--agent-model",
            "test-structured-model",
        ]
    )
    # ``None`` lets Settings resolve AGENT_MODEL_PROVIDER before the built-in
    # mock fallback. The effective default remains mock.
    assert default_args.model_provider is None
    assert live_args.model_provider == "openai"
    assert live_args.agent_model == "test-structured-model"
