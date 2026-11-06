"""Test grounding and deterministic assessment rules without network calls."""

from __future__ import annotations

import sys
from datetime import date
from types import SimpleNamespace

import pytest

from public_sector_opportunity_research.assessment import (
    DocumentedAward,
    GroundedStatement,
    GroundingError,
    OpenAIAssessmentError,
    OpenAIAssessmentGenerator,
    OpportunityAssessmentDraft,
    OpportunityAssessmentService,
    finalize_assessment,
)
from public_sector_opportunity_research.ingestion import validate_metadata_dates
from public_sector_opportunity_research.models import (
    Chunk,
    DocumentMetadata,
    EmbeddingSyncResult,
    RetrievalResponse,
    SemanticSearchResult,
)
from public_sector_opportunity_research.storage import SQLiteStore


class FakeRetriever:
    """Return a prebuilt retrieval response to isolate assessment behavior."""

    def __init__(self, response):
        self.response = response

    def retrieve(self, *args, **kwargs):
        return self.response


class FakeGenerator:
    """Return a schema-validated draft and record whether it was called."""

    model_name = "fake-structured-model"

    def __init__(self, draft):
        self.draft = draft
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        return self.draft, "resp_test"


def _result(
    document_id,
    chunk_id,
    document_type,
    publication_date,
    effective_date=None,
):
    return SemanticSearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        text=f"Evidence text for {document_id}",
        agency_id="agency-a",
        agency_name="Synthetic Agency",
        document_title=f"{document_id} title",
        document_type=document_type,
        source_url=f"https://example.invalid/{document_id}",
        publication_date=publication_date,
        effective_date=effective_date,
        last_updated_date=None,
        capture_date="2028-01-30",
        section_heading="Evidence",
        page_number=1,
        vector_distance=0.1,
        vector_similarity_score=0.9,
        vector_rank=1,
        reranker_score=3.0,
        reranker_rank=1,
    )


def _statement(text, document_ids, chunk_ids, kind="documented_fact"):
    return GroundedStatement(
        text=text,
        statement_type=kind,
        source_document_ids=document_ids,
        chunk_ids=chunk_ids,
    )


def _draft(
    overlap="substantial",
    source_ids=None,
    chunk_ids=None,
    procurement_status="confirmed_award",
):
    source_ids = source_ids or ["doc_strategy", "doc_award"]
    chunk_ids = chunk_ids or ["chk_strategy", "chk_award"]
    return OpportunityAssessmentDraft(
        agency="Synthetic Agency",
        opportunity_title="Citizen-service modernization",
        agency_need_and_supporting_evidence=[
            _statement("The strategy documents a modernization need.", ["doc_strategy"], ["chk_strategy"])
        ],
        opportunity_status="potential",
        documented_procurement_status=procurement_status,
        award_scope_overlap=overlap,
        commercial_implications=[
            _statement(
                "The award may limit a new platform sale.",
                source_ids,
                chunk_ids,
                "model_inference",
            )
        ],
        explanation_of_assessment=[
            _statement("The evidence was compared chronologically.", source_ids, chunk_ids)
        ],
        supporting_source_ids=source_ids,
        supporting_chunk_ids=chunk_ids,
        relevant_publication_and_contract_dates=[],
        uncertainties_and_information_gaps=["Future option exercise is unknown."],
    )


def _award(
    publication=date(2028, 2, 20),
    effective=date(2028, 2, 15),
    execution=date(2028, 2, 15),
    status="confirmed_award",
):
    return DocumentedAward(
        document_id="doc_award",
        title="Synthetic award",
        procurement_status=status,
        contract_id="NDRS-2028-017-C1",
        selected_vendor="Synthetic Vendor",
        source_url="https://example.invalid/doc_award",
        publication_date=publication,
        effective_date=effective,
        execution_date=execution,
        last_updated_date=publication,
    )


def _evidence():
    return [
        _result("doc_strategy", "chk_strategy", "strategy", "2026-02-12"),
        _result(
            "doc_award",
            "chk_award",
            "confirmed-award",
            "2028-02-20",
            "2028-02-15",
        ),
    ]


def test_valid_structured_output_and_full_award_status():
    assessment = finalize_assessment(_draft("substantial"), _evidence(), [_award()])

    assert assessment.opportunity_status == "already_awarded"
    assert assessment.documented_procurement_status == "confirmed_award"
    assert assessment.documented_awards[0].document_id == "doc_award"
    assert {item.date for item in assessment.relevant_publication_and_contract_dates} == {
        date(2026, 2, 12),
        date(2028, 2, 20),
        date(2028, 2, 15),
    }
    # Revalidation proves the final serialized object still conforms to schema.
    assert assessment.model_validate(assessment.model_dump()) == assessment


def test_partial_overlap_becomes_partially_addressed():
    assessment = finalize_assessment(_draft("partial"), _evidence(), [_award()])
    assert assessment.opportunity_status == "partially_addressed"


def test_unknown_citation_is_rejected():
    bad = _draft(
        source_ids=["doc_strategy", "doc_invented"],
        chunk_ids=["chk_strategy"],
    )
    with pytest.raises(GroundingError, match="doc_invented"):
        finalize_assessment(bad, _evidence(), [_award()])


def test_no_documented_award_is_potential_not_proof_of_absence():
    evidence = [_evidence()[0]]
    draft = _draft(
        overlap="unknown",
        source_ids=["doc_strategy"],
        chunk_ids=["chk_strategy"],
        procurement_status="none",
    )
    assessment = finalize_assessment(draft, evidence, [])

    assert assessment.opportunity_status == "potential"
    assert any("not proof" in gap for gap in assessment.uncertainties_and_information_gaps)


def test_intent_to_award_is_not_treated_as_executed_contract():
    evidence = [
        _evidence()[0],
        _result(
            "doc_award",
            "chk_award",
            "intent-to-award",
            "2028-01-22",
            "2028-02-15",
        ),
    ]
    intent = _award(
        publication=date(2028, 1, 22),
        effective=date(2028, 2, 15),
        execution=None,
        status="intent_to_award",
    )
    assessment = finalize_assessment(
        _draft("substantial", procurement_status="intent_to_award"),
        evidence,
        [intent],
    )

    assert assessment.opportunity_status == "potential"
    assert assessment.documented_procurement_status == "intent_to_award"
    assert "not establish" in assessment.explanation_of_assessment[0].text


def test_award_predating_need_does_not_close_later_opportunity():
    evidence = [
        _result("doc_strategy", "chk_strategy", "strategy", "2028-01-01"),
        _result("doc_award", "chk_award", "award", "2027-01-01", "2027-02-01"),
    ]
    assessment = finalize_assessment(
        _draft("substantial"),
        evidence,
        [_award(date(2027, 2, 2), date(2027, 2, 1), date(2027, 2, 1))],
    )
    assert assessment.opportunity_status == "potential"
    assert any("predates" in gap for gap in assessment.uncertainties_and_information_gaps)


def test_missing_or_conflicting_contract_dates_are_insufficient_and_consistent():
    conflicting = _award(
        publication=date(2028, 2, 20),
        effective=date(2028, 2, 15),
        execution=None,
    )
    draft = _draft("substantial")
    # Simulate the earlier inconsistent model narrative. Finalization must not
    # retain it after deterministic status rules disagree.
    draft.explanation_of_assessment[0].text = "The opportunity is already awarded."

    assessment = finalize_assessment(draft, _evidence(), [conflicting])

    assert assessment.opportunity_status == "insufficient_evidence"
    assert assessment.documented_procurement_status == "conflicting_evidence"
    assert assessment.award_scope_overlap == "substantial"
    assert all(
        "already awarded" not in statement.text.lower()
        for statement in assessment.explanation_of_assessment
    )
    assert "insufficient" in assessment.commercial_implications[0].text.lower() or "do not" in assessment.commercial_implications[0].text.lower()


def test_empty_retrieval_returns_insufficient_without_calling_api(store):
    response = RetrievalResponse(
        "trace_empty",
        [],
        {"initial_candidates": []},
        EmbeddingSyncResult(0, 0, False),
    )
    generator = FakeGenerator(_draft())
    run = OpportunityAssessmentService(store, FakeRetriever(response), generator).assess(
        "question", "missing-agency"
    )

    assert run.assessment.opportunity_status == "insufficient_evidence"
    assert run.retrieved_evidence == []
    assert generator.calls == 0


def test_openai_api_errors_are_wrapped(monkeypatch):
    class FailingResponses:
        def parse(self, **kwargs):
            raise ConnectionError("simulated outage")

    class FailingClient:
        def __init__(self, **kwargs):
            self.responses = FailingResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-logged")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FailingClient))

    with pytest.raises(OpenAIAssessmentError, match="ConnectionError"):
        OpenAIAssessmentGenerator("test-model").generate(
            "question", "Synthetic Agency", _evidence(), [_award()]
        )


def test_explicit_scenario_copy_preserves_documents_chunks_and_awards(store, tmp_path):
    """Scenario databases copy exact IDs and rebuild vectors later."""
    for document_id, document_type, published in (
        ("doc_strategy", "strategy", date(2026, 2, 12)),
        ("doc_award", "confirmed-award", date(2028, 2, 20)),
    ):
        metadata = DocumentMetadata(
            agency_id="agency-a",
            agency_name="Synthetic Agency",
            title=f"{document_id} title",
            document_type=document_type,
            publication_date=published,
            effective_date=(
                date(2028, 2, 15) if document_type == "confirmed-award" else None
            ),
            capture_date=published,
            extra_metadata=(
                {
                    "procurement_status": "confirmed_award",
                    "contract_id": "NDRS-2028-017-C1",
                    "selected_vendor": "Synthetic Vendor",
                    "execution_date": "2028-02-15",
                }
                if document_type == "confirmed-award"
                else {}
            ),
        )
        store.save_document(
            document_id,
            metadata,
            f"hash-{document_id}",
            None,
            [
                Chunk(
                    chunk_id=f"chk_{document_id}",
                    document_id=document_id,
                    chunk_index=0,
                    text=f"text for {document_id}",
                    token_count=3,
                )
            ],
        )

    scenario_a = SQLiteStore(tmp_path / "scenario_a.sqlite3")
    scenario_b = SQLiteStore(tmp_path / "scenario_b.sqlite3")
    store.copy_documents_to(scenario_a, ["doc_strategy"])
    store.copy_documents_to(scenario_b, ["doc_strategy", "doc_award"])

    assert [row["document_id"] for row in scenario_a.list_documents()] == [
        "doc_strategy"
    ]
    assert {row["document_id"] for row in scenario_b.list_documents()} == {
        "doc_strategy",
        "doc_award",
    }
    assert scenario_a.list_documented_awards("agency-a") == []
    assert scenario_b.list_documented_awards("agency-a")[0]["document_id"] == "doc_award"
    assert scenario_b.list_chunks("doc_award")[0]["chunk_id"] == "chk_doc_award"


def test_confirmed_award_metadata_validation_rejects_conflicting_dates():
    metadata = DocumentMetadata(
        agency_id="agency-a",
        agency_name="Synthetic Agency",
        title="Confirmed award",
        document_type="confirmed-award",
        publication_date=date(2028, 2, 20),
        effective_date=date(2028, 2, 15),
        capture_date=date(2028, 2, 20),
        extra_metadata={
            "future_dated_fictional_evidence": True,
            "procurement_status": "confirmed_award",
            "contract_id": "contract-1",
            "selected_vendor": "Synthetic Vendor",
            "execution_date": "2028-02-16",
        },
    )

    with pytest.raises(ValueError, match="must match execution_date"):
        validate_metadata_dates(metadata)


def test_unlabeled_future_source_date_is_rejected():
    metadata = DocumentMetadata(
        agency_id="agency-a",
        agency_name="Synthetic Agency",
        title="Future source",
        document_type="strategy",
        publication_date=date(2099, 1, 1),
        capture_date=date(2099, 1, 1),
    )

    with pytest.raises(ValueError, match="future_dated_fictional_evidence"):
        validate_metadata_dates(metadata)
