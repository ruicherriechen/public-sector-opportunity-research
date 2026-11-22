"""Verify graph orchestration without network, OpenAI, or model downloads."""

from __future__ import annotations

import sqlite3
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from threading import Lock
from time import sleep

from public_sector_opportunity_research.agents.contracts import (
    EvidenceRecord,
    ResearchEvidenceBatch,
    ResearchGraphResult,
)
from public_sector_opportunity_research.agents.graph import (
    build_research_graph,
    run_offline_research_graph,
)
from public_sector_opportunity_research.agents.evidence import (
    compact_evidence,
    compact_procurement_evidence,
    compact_strategy_evidence,
    estimate_tokens,
)
from public_sector_opportunity_research.agents.independent_evaluator import (
    independent_evaluator_node,
)
from public_sector_opportunity_research.agents.mock_llm import (
    AgentBudgetExceeded,
    AgentBudgetTracker,
    SyntheticMockAgentModel,
)
from public_sector_opportunity_research.agents.procurement_researcher import (
    build_procurement_retrieval_query,
)
from public_sector_opportunity_research.agents.rag_service import RAGResearchService
from public_sector_opportunity_research.cli import build_parser
from public_sector_opportunity_research.config import Settings
from public_sector_opportunity_research.models import (
    EmbeddingSyncResult,
    RetrievalResponse,
    SemanticSearchResult,
)


class FakeLocalEvidenceService:
    """Return deterministic evidence while recording parallel branch calls."""

    def __init__(self, missing_procurement: bool = False) -> None:
        self.missing_procurement = missing_procurement
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def research(self, **kwargs):
        categories = list(kwargs["document_types"])
        agency_id = kwargs["agency_id"]
        self.calls.append((kwargs["query"], tuple(categories)))
        missing = categories if self.missing_procurement and "rfp" in categories else []
        evidence = [] if missing else [_evidence(agency_id, item) for item in categories]
        return ResearchEvidenceBatch(
            evidence=evidence,
            retrieval_trace_ids=[f"trace-{len(self.calls)}-{item}" for item in categories],
            requested_categories=categories,
            missing_categories=missing,
        )


class FakeSemanticRetriever:
    """Mimic the existing retriever at the shared-service boundary."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        document_type = kwargs["document_type"]
        results = [] if document_type == "budget" else [_semantic_result(document_type)]
        return RetrievalResponse(
            trace_id=f"trace-{document_type}",
            results=results,
            trace={"initial_candidates": []},
            embedding_sync=EmbeddingSyncResult(0, len(results), False),
        )


class ProcurementRelevanceRetriever:
    """Return one configurable award and empty earlier procurement stages."""

    def __init__(
        self,
        *,
        text: str,
        reranker_score: float,
        document_type: str = "confirmed-award",
    ) -> None:
        self.text = text
        self.reranker_score = reranker_score
        self.document_type = document_type

    def retrieve(self, **kwargs):
        document_type = kwargs["document_type"]
        results = []
        if document_type == self.document_type:
            base = _semantic_result(document_type)
            results = [base.__class__(**{
                **base.__dict__,
                "document_title": "USAspending Award TEST",
                "text": self.text,
                "reranker_score": self.reranker_score,
                "last_updated_date": "2026-01-01",
            })]
        return RetrievalResponse(
            trace_id=f"trace-{document_type}", results=results,
            trace={"initial_candidates": []},
            embedding_sync=EmbeddingSyncResult(0, len(results), False),
        )


class HHSQuerySensitiveAwardRetriever:
    """Reproduce the ORI award's long-query versus concise-query score shift."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        document_type = kwargs["document_type"]
        results = []
        if document_type == "confirmed-award":
            base = _semantic_result(document_type)
            ori_text = (
                "Official USAspending.gov prime award record. Award ID: "
                "7571MN26F80168. Recipient/vendor: DLT SOLUTIONS, LLC. "
                "Description/scope: IT modernization services including legacy "
                "system modernization, cloud migration, cybersecurity, data "
                "migration, and operations and maintenance."
            )
            peripheral_text = (
                "Official USAspending.gov prime award record. Award ID: "
                "PERIPHERAL-1. Description/scope: laptop docks, keyboards, and mice."
            )
            concise = len(str(kwargs["query"]).split()) <= 20
            results = [
                base.__class__(
                    **{
                        **base.__dict__,
                        "chunk_id": "chunk-hhs-ori",
                        "document_id": "document-hhs-ori",
                        "document_title": "USAspending Award 7571MN26F80168",
                        "text": ori_text,
                        "reranker_score": 1.58 if concise else -5.48,
                        "reranker_rank": 1,
                    }
                ),
                base.__class__(
                    **{
                        **base.__dict__,
                        "chunk_id": "chunk-peripheral",
                        "document_id": "document-peripheral",
                        "document_title": "USAspending Award PERIPHERAL-1",
                        "text": peripheral_text,
                        "reranker_score": -6.4,
                        "reranker_rank": 2,
                    }
                ),
            ]
        return RetrievalResponse(
            trace_id=f"trace-{document_type}",
            results=results,
            trace={"initial_candidates": []},
            embedding_sync=EmbeddingSyncResult(0, len(results), False),
        )


class OverlapDetectingRetriever(FakeSemanticRetriever):
    """Record whether two calls overlap inside the shared retrieval resource."""

    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.maximum_active = 0
        self.guard = Lock()

    def retrieve(self, **kwargs):
        with self.guard:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        sleep(0.01)
        try:
            return super().retrieve(**kwargs)
        finally:
            with self.guard:
                self.active -= 1


def _evidence(agency_id: str, document_type: str) -> EvidenceRecord:
    """Create one fully attributed passage for graph tests."""
    texts = {
        "strategy": (
            "The mission priority is a single unified resident portal with status "
            "tracking, governed APIs, translated content, and WCAG accessibility."
        ),
        "budget": (
            "Requested Resources: portal implementation and API enablement. "
            "Total | $14,000,000 | Synthetic planning amount."
        ),
        "program-plan": (
            "Payment accuracy and program integrity require configurable rules, "
            "entity resolution, case-review workflow, evidence tracking, and audit history."
        ),
        "rfp": (
            "RFP NDRS-2028-017 seeks a Resident Access and Integration Platform. "
            "Scope includes a resident portal, identity, intake, notices, status, and APIs."
        ),
        "intent-to-award": (
            "Notice of Intent to Award NDRS-2028-017 names selected vendor Aurora "
            "Civic Systems. This is an intent to award, not an executed contract."
        ),
        "confirmed-award": (
            "Executed contract NDRS-2028-017-C1 has an initial value of $11,850,000 "
            "and a $17,200,000 maximum. It covers the resident portal, identity, "
            "intake, document exchange, multilingual communications, status, governed "
            "APIs, accessibility, security, training, and phased rollout. Contract "
            "status executed; award and execution date 2028-02-15; Aurora Civic "
            "Systems is the vendor."
        ),
    }
    return EvidenceRecord(
        chunk_id=f"chunk-{document_type}",
        document_id=f"document-{document_type}",
        agency_id=agency_id,
        agency_name="Synthetic Agency",
        document_title=f"Synthetic {document_type}",
        document_type=document_type,
        publication_date=date(2026, 1, 1),
        capture_date=date(2026, 1, 2),
        effective_date=(
            date(2028, 2, 15) if document_type == "confirmed-award" else None
        ),
        text=texts.get(document_type, "Synthetic evidence used only by a unit test."),
        vector_rank=1,
        vector_similarity_score=0.8,
        reranker_rank=1,
        reranker_score=2.0,
        retrieval_trace_id=f"trace-{document_type}",
        freshness_status="not_checked",
    )


def _semantic_result(document_type: str) -> SemanticSearchResult:
    """Create one existing-model result for adapter integration tests."""
    return SemanticSearchResult(
        chunk_id=f"chunk-{document_type}",
        document_id=f"document-{document_type}",
        text="Synthetic result from the fake semantic retriever.",
        agency_id="synthetic-ndrs",
        agency_name="Synthetic Agency",
        document_title=f"Synthetic {document_type}",
        document_type=document_type,
        source_url=None,
        publication_date="2026-01-01",
        effective_date=None,
        last_updated_date=None,
        capture_date="2026-01-02",
        section_heading="Purpose",
        page_number=1,
        vector_distance=0.2,
        vector_similarity_score=0.8,
        vector_rank=1,
        reranker_score=2.0,
        reranker_rank=1,
    )


def _initial_state(max_iterations: int = 2):
    """Return the minimal JSON-compatible graph input used by unit tests."""
    return {
        "agency_id": "synthetic-ndrs",
        "objective": "Assess citizen-service modernization evidence.",
        "as_of_date": None,
        "freshness_cutoff_date": None,
        "max_iterations": max_iterations,
        "iteration": 0,
        "thread_id": "unit-test-thread",
        "tasks": [],
        "researcher_outputs": [],
        "execution_trace": [],
    }


def test_parallel_outputs_merge_before_synthesis_and_complete():
    """Both fan-out branches must survive state merging and reach synthesis."""
    service = FakeLocalEvidenceService()
    final_state = build_research_graph(service).invoke(_initial_state())
    result = ResearchGraphResult.model_validate(final_state["final_result"])

    assert result.status == "complete"
    assert result.model_provider == "mock"
    assert result.iterations_completed == 1
    assert {item.role for item in result.researcher_outputs} == {
        "strategy",
        "procurement",
    }
    assert set(result.synthesis.evidence_by_role) == {"strategy", "procurement"}
    assert result.evaluation.citation_integrity is True
    assert result.evaluation.evidence_sufficiency is True
    assert {item.status for item in result.synthesis.opportunities} == {
        "already_awarded",
        "potential",
    }
    assert result.budget.calls_used == 3
    assert result.budget.within_budget is True
    assert all(len(item.excerpt) <= 800 for item in result.synthesis.evidence)
    strategy = next(item for item in result.researcher_outputs if item.role == "strategy")
    procurement = next(
        item for item in result.researcher_outputs if item.role == "procurement"
    )
    assert {item.finding_type for item in strategy.findings} >= {
        "mission_priority",
        "operational_need",
        "budget_evidence",
    }
    assert {item.finding_type for item in procurement.findings} >= {
        "rfp",
        "award_status",
        "contract_scope",
        "vendor",
        "procurement_date",
    }
    assert result.synthesis.opportunities[0].status == "already_awarded"
    assert result.synthesis.opportunities[0].commercially_actionable is False
    assert result.synthesis.opportunities[1].status == "potential"
    assert result.synthesis.opportunities[0].score.total >= (
        result.synthesis.opportunities[1].score.total
    )

    # The confirmed source must yield separate status, scope, vendor, and date
    # facts so the deterministic evaluator can verify an awarded classification.
    confirmed_findings = [
        item
        for item in procurement.findings
        if "document-confirmed-award" in item.document_ids
    ]
    assert {item.finding_type for item in confirmed_findings} == {
        "award_status",
        "contract_scope",
        "vendor",
        "procurement_date",
    }
    assert any("$11,850,000" in item.statement for item in confirmed_findings)


def test_real_agency_mock_abstains_instead_of_reusing_synthetic_claims():
    """DOL-like terms must not trigger the scripted fictional resident scenario."""
    model = SyntheticMockAgentModel()
    real_record = _evidence("us-dol", "strategy").model_copy(
        update={
            "agency_name": "Department of Labor",
            "document_title": "DOL Enterprise Data Strategy",
            "text": (
                "A single enterprise approach uses governed APIs and improves "
                "data access across Department of Labor programs."
            ),
        }
    )
    reference = compact_evidence([real_record], focus_terms=("api", "data"))[0]
    result = model.analyze_strategy(
        objective="Identify modernization needs.",
        evidence=[reference],
        iteration=1,
    )
    assert result.output.findings == []
    assert "does not interpret unfamiliar real-agency" in result.output.summary


def test_evaluator_routes_lexically_unsupported_real_claim_to_human_review():
    """Existing citation IDs cannot validate a claim absent from the excerpt."""
    service = FakeLocalEvidenceService()
    final_state = build_research_graph(service).invoke(_initial_state())
    result = ResearchGraphResult.model_validate(final_state["final_result"])
    synthesis = result.synthesis.model_dump(mode="json")
    synthesis["implementation"] = "openai"
    synthesis["evidence"][0]["document_title"] = "DOL official report"
    synthesis["findings"][0]["statement"] = (
        "Quantum satellite licensing is the agency's funded operational priority."
    )
    update = independent_evaluator_node(
        {
            "iteration": 1,
            "max_iterations": 1,
            "synthesis": synthesis,
        }
    )
    assert update["route"] == "human_review"
    assert update["evaluation"]["grounding_review_required"]
    assert update["evaluation"]["evidence_sufficiency"] is False


def test_evaluator_rejects_cross_agency_evidence_even_with_valid_ids():
    """Citation syntax cannot bypass the agency evidence partition."""
    service = FakeLocalEvidenceService()
    final_state = build_research_graph(service).invoke(_initial_state())
    result = ResearchGraphResult.model_validate(final_state["final_result"])
    synthesis = result.synthesis.model_dump(mode="json")
    synthesis["evidence"][0]["agency_id"] = "other-agency"

    update = independent_evaluator_node(
        {
            "agency_id": result.agency_id,
            "iteration": 1,
            "max_iterations": 1,
            "synthesis": synthesis,
        }
    )

    assert update["route"] == "human_review"
    assert update["evaluation"]["citation_integrity"] is False


def test_suspicious_but_usable_evidence_is_warned_without_forcing_failure():
    """An isolated source warning remains visible without invalidating facts."""
    service = FakeLocalEvidenceService()
    final_state = build_research_graph(service).invoke(_initial_state())
    result = ResearchGraphResult.model_validate(final_state["final_result"])
    synthesis = result.synthesis.model_dump(mode="json")
    synthesis["evidence"][0]["suspicious_indicators"] = ["instruction_override"]

    update = independent_evaluator_node(
        {
            "agency_id": result.agency_id,
            "iteration": 1,
            "max_iterations": 1,
            "synthesis": synthesis,
        }
    )

    assert update["route"] == "complete"
    assert "instruction_override" in " ".join(
        update["evaluation"]["security_warnings"]
    )


def test_procurement_context_keeps_complete_short_award_and_strategy_unchanged():
    """The regression source must retain vendor, value, scope, status, and dates."""
    source_text = Path("sample_data/06_confirmed_contract_award.txt").read_text(
        encoding="utf-8"
    )
    record = _evidence("synthetic-ndrs", "confirmed-award").model_copy(
        update={"text": source_text}
    )

    procurement = compact_procurement_evidence([record], max_context_tokens=2000)

    assert procurement[0].excerpt == source_text.strip()
    assert "Aurora Civic\nSystems" in procurement[0].excerpt
    assert "$11,850,000" in procurement[0].excerpt
    assert "maximum value to $17,200,000" in procurement[0].excerpt
    assert "Contract status: Executed by both fictional parties" in procurement[0].excerpt
    assert "governed APIs to three benefits systems" in procurement[0].excerpt
    assert "Execution date: 2028-02-15" in procurement[0].excerpt

    # Strategy still uses its original focused 800-character implementation.
    strategy = compact_evidence([record], focus_terms=("executed",))
    assert len(strategy[0].excerpt) <= 800
    assert strategy[0].excerpt != source_text.strip()


def test_procurement_context_deduplicates_and_honors_aggregate_budget():
    """Repeated passages should not consume context twice or exceed the cap."""
    long_text = (
        "Confirmed Award. Contract status executed. Vendor Aurora Civic Systems. "
        "Award date 2028-02-15. Contract value $11,850,000. Executed scope portal APIs. "
        + "procurement history detail " * 1200
    )
    record = _evidence("synthetic-ndrs", "confirmed-award").model_copy(
        update={"text": long_text}
    )
    duplicate = record.model_copy(update={"chunk_id": "duplicate-chunk"})

    references = compact_procurement_evidence(
        [record, record, duplicate], max_context_tokens=2000
    )

    assert len(references) == 1
    assert estimate_tokens(references[0].excerpt) <= 2000
    assert "Vendor Aurora Civic Systems" in references[0].excerpt
    assert "Executed scope portal APIs" in references[0].excerpt


def test_missing_categories_loop_then_stop_at_human_review_limit():
    """Evidence gaps loop only to the bound and never become false findings."""
    service = FakeLocalEvidenceService(missing_procurement=True)
    final_state = build_research_graph(service).invoke(_initial_state(max_iterations=2))
    result = ResearchGraphResult.model_validate(final_state["final_result"])

    assert result.status == "human_review"
    assert result.iterations_completed == 2
    assert len(result.researcher_outputs) == 4
    assert result.evaluation.decision == "human_review"
    assert result.evaluation.unresolved_categories == [
        "confirmed-award",
        "intent-to-award",
        "rfp",
    ]
    # The fixed local corpus cannot change inside one graph invocation, so the
    # second bounded iteration reuses both branch results instead of repeating
    # identical retrieval and model calls.
    assert len(service.calls) == 2
    assert result.budget.calls_used == 3


def test_sqlite_checkpoint_persists_graph_state(tmp_path):
    """The offline runner should write checkpoints to a separate SQLite file."""
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    result = run_offline_research_graph(
        service=FakeLocalEvidenceService(),
        agency_id="synthetic-ndrs",
        objective="Assess modernization.",
        checkpoint_path=checkpoint_path,
        max_iterations=1,
        thread_id="persistent-test-thread",
    )

    assert result.status == "complete"
    with sqlite3.connect(checkpoint_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] > 0
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("persistent-test-thread",),
            ).fetchone()[0]
            > 0
        )


def test_shared_rag_service_preserves_traces_filters_and_freshness():
    """The adapter should reuse exact-type retrieval and retain provenance."""
    retriever = FakeSemanticRetriever()
    service = RAGResearchService(retriever)
    batch = service.research(
        query="What strategy and budget evidence exists?",
        agency_id="synthetic-ndrs",
        document_types=["strategy", "budget"],
        as_of_date="2026-12-31",
        freshness_cutoff_date="2025-01-01",
    )

    assert [call["document_type"] for call in retriever.calls] == [
        "strategy",
        "budget",
    ]
    assert all(call["agency_id"] == "synthetic-ndrs" for call in retriever.calls)
    assert batch.retrieval_trace_ids == ["trace-strategy", "trace-budget"]
    assert batch.missing_categories == ["budget"]
    assert batch.evidence[0].retrieval_trace_id == "trace-strategy"
    assert batch.evidence[0].freshness_status == "meets_cutoff"

    # The LangChain extension point returns the same validated service output.
    tool_output = service.as_langchain_tool().invoke(
        {
            "query": "What strategy evidence exists?",
            "agency_id": "synthetic-ndrs",
            "document_types": ["strategy"],
        }
    )
    assert tool_output["evidence"][0]["chunk_id"] == "chunk-strategy"


def test_unrelated_negative_award_does_not_satisfy_procurement_coverage():
    """A peripheral order must remain stored evidence but not prove overlap."""
    service = RAGResearchService(ProcurementRelevanceRetriever(
        text="DELL BPA ORDER - IT PERIPHERALS", reranker_score=-6.4
    ))
    batch = service.research(
        query="enterprise application modernization", agency_id="us-dol",
        document_types=["rfp", "intent-to-award", "confirmed-award"],
    )

    assert batch.evidence[0].procurement_relevance == "unrelated"
    assert set(batch.missing_categories) == {"rfp", "intent-to-award", "confirmed-award"}
    assert any("positive scope relevance" in item for item in batch.coverage_gaps)


def test_explicit_state_award_with_scope_overlap_survives_negative_raw_score():
    """Form-like state awards need not have a positive, uncalibrated MiniLM score."""

    service = RAGResearchService(ProcurementRelevanceRetriever(
        text=(
            "Primary Solicitation Results. One viable solution was selected for "
            "final contract award. Contract Start Date: 2026-02-02. Total Contract "
            "Cost: $101 million. The selected enterprise application modernization "
            "vendor will replace legacy systems."
        ),
        reranker_score=-2.4,
    ))
    batch = service.research(
        query="contract award vendor scope legacy system modernization enterprise application modernization",
        agency_id="state-agency",
        document_types=["rfp", "intent-to-award", "confirmed-award"],
    )

    assert batch.evidence[0].procurement_relevance == "relevant"
    assert "non-positive" in (batch.evidence[0].procurement_relevance_reason or "")
    assert batch.missing_categories == []
    assert set(batch.coverage_gaps) == {
        "rfp: unavailable preceding procurement stage",
        "intent-to-award: unavailable preceding procurement stage",
    }


def test_state_notice_of_award_is_relevant_without_claiming_execution():
    """An official vendor-selection notice can be relevant while still intent-stage."""

    service = RAGResearchService(ProcurementRelevanceRetriever(
        text=(
            "Revised Notice of Award. Vendor Selected for Award: Example PBC. "
            "UI Claimant Portal and Application. Initial term and renewal costs "
            "are listed, but contract execution status is unknown."
        ),
        reranker_score=-4.1,
        document_type="intent-to-award",
    ))
    batch = service.research(
        query=(
            "contract award solicitation vendor scope unemployment insurance "
            "claimant portal"
        ),
        agency_id="state-workforce-agency",
        document_types=["intent-to-award"],
    )

    assert batch.evidence[0].procurement_relevance == "relevant"
    assert batch.evidence[0].document_type == "intent-to-award"
    assert "non-positive" in (batch.evidence[0].procurement_relevance_reason or "")


def test_generic_buying_plan_words_do_not_override_negative_scope_score():
    """Procurement boilerplate must not become objective overlap by itself."""

    service = RAGResearchService(ProcurementRelevanceRetriever(
        text=(
            "Request for proposals buying plan. The agency may issue a solicitation "
            "and select a vendor following evaluation."
        ),
        reranker_score=-1.2,
        document_type="rfp",
    ))
    batch = service.research(
        query=(
            "contract award solicitation vendor scope unemployment insurance "
            "claimant portal"
        ),
        agency_id="state-workforce-agency",
        document_types=["rfp"],
    )

    assert batch.evidence[0].procurement_relevance == "unrelated"
    assert batch.missing_categories == ["rfp"]


def test_relevant_confirmed_award_makes_earlier_stages_nonblocking_gaps():
    """A supported executed award can establish status without an RFP copy."""
    service = RAGResearchService(ProcurementRelevanceRetriever(
        text="Executed enterprise application modernization and integration platform",
        reranker_score=4.2,
    ))
    batch = service.research(
        query="enterprise application modernization", agency_id="us-dol",
        document_types=["rfp", "intent-to-award", "confirmed-award"],
    )

    assert batch.evidence[0].procurement_relevance == "relevant"
    assert batch.missing_categories == []
    assert set(batch.coverage_gaps) == {
        "rfp: unavailable preceding procurement stage",
        "intent-to-award: unavailable preceding procurement stage",
    }
    assert batch.evidence[0].last_updated_date == date(2026, 1, 1)
    assert batch.evidence[0].capture_date == date(2026, 1, 2)


def test_long_objective_uses_concise_procurement_retrieval_query():
    """Instructional caveats must not dilute short award-record reranking."""
    objective = (
        "Identify evidence-grounded enterprise software and AI modernization "
        "opportunities. Evaluate mission priorities, budget or mandate evidence, "
        "relevant procurement history, existing vendor commitments, and potential "
        "commercial whitespace without treating missing procurement evidence as "
        "proof of availability."
    )

    query = build_procurement_retrieval_query(objective=objective)

    assert len(query.split()) <= 20
    assert "legacy system modernization" in query
    assert "enterprise application modernization" in query
    assert "proof of availability" not in query


def test_hhs_ori_award_survives_long_objective_and_peripheral_stays_unrelated():
    """The generic fix recovers modernization scope without admitting hardware."""
    objective = (
        "Identify evidence-grounded enterprise software and AI modernization "
        "opportunities and evaluate procurement history without treating missing "
        "contracts as proof of commercial availability."
    )
    retriever = HHSQuerySensitiveAwardRetriever()
    service = RAGResearchService(retriever)

    batch = service.research(
        query=build_procurement_retrieval_query(objective=objective),
        agency_id="us-hhs",
        document_types=["rfp", "intent-to-award", "confirmed-award"],
    )

    by_id = {item.document_id: item for item in batch.evidence}
    assert by_id["document-hhs-ori"].procurement_relevance == "relevant"
    assert "legacy" in (by_id["document-hhs-ori"].procurement_relevance_reason or "")
    assert by_id["document-peripheral"].procurement_relevance == "unrelated"
    assert "confirmed-award" not in batch.missing_categories
    assert set(batch.coverage_gaps) == {
        "rfp: unavailable preceding procurement stage",
        "intent-to-award: unavailable preceding procurement stage",
    }


def test_graph_preserves_full_objective_for_reasoning_but_not_retrieval():
    """Only local procurement search is compacted; graph input remains intact."""
    objective = (
        "Identify evidence-grounded enterprise software and AI modernization "
        "opportunities while preserving every commercial-availability caveat."
    )
    service = FakeLocalEvidenceService()
    state = _initial_state(max_iterations=1)
    state["objective"] = objective

    final_state = build_research_graph(service).invoke(state)

    result = ResearchGraphResult.model_validate(final_state["final_result"])
    procurement_query = next(
        query for query, categories in service.calls if "confirmed-award" in categories
    )
    assert result.objective == objective
    assert objective not in procurement_query
    assert "contract award solicitation vendor scope" in procurement_query


def test_shared_rag_service_serializes_parallel_model_and_index_access():
    """Parallel graph branches must not race one lazy model/vector index."""
    retriever = OverlapDetectingRetriever()
    service = RAGResearchService(retriever)

    def research(document_type: str):
        return service.research(
            query="test",
            agency_id="synthetic-ndrs",
            document_types=[document_type],
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(research, ["strategy", "rfp"]))

    assert retriever.maximum_active == 1


def test_existing_cli_commands_remain_parseable():
    """Adding the graph command must not remove the earlier MVP commands."""
    parser = build_parser()
    assert parser.parse_args(["list-documents"]).command == "list-documents"
    assert parser.parse_args(["semantic-search", "--query", "test"]).command == (
        "semantic-search"
    )
    graph_args = parser.parse_args(
        [
            "run-agent-graph",
            "--agency-id",
            "synthetic-ndrs",
            "--objective",
            "test",
            "--procurement-context-tokens",
            "3500",
        ]
    )
    assert graph_args.command == "run-agent-graph"
    assert graph_args.procurement_context_tokens == 3500


def test_workflow_budget_defaults_are_documented_and_shared():
    """Mock, graph, settings, and CLI help should expose the 50k input default."""
    settings = Settings(database_path=Path("test.sqlite3"))
    tracker = AgentBudgetTracker()
    parser = build_parser()
    command_action = next(
        action
        for action in parser._actions
        if getattr(action, "choices", None)
        and "run-agent-graph" in action.choices
    )
    graph_help = command_action.choices["run-agent-graph"].format_help()
    research_help = command_action.choices["research-agency"].format_help()

    assert settings.langgraph_max_agent_calls == 6
    assert settings.langgraph_max_input_tokens == 50000
    assert settings.langgraph_max_output_tokens == 6000
    assert settings.strategy_context_tokens == 3000
    assert settings.procurement_context_tokens == 3000
    assert tracker.max_calls == 6
    assert tracker.max_input_tokens == 50000
    assert tracker.max_output_tokens == 6000
    assert "default 50000" in graph_help
    assert "default 50000" in research_help


def test_large_strategy_corpus_is_diverse_deduplicated_and_bounded():
    """A DOL-sized result set must not serialize every retrieved chunk."""
    records = []
    for category in ("strategy", "budget", "program-plan", "audit-report"):
        for index in range(10):
            base = _evidence("us-dol", category).model_copy(update={
                "chunk_id": f"{category}-chunk-{index}",
                "document_id": f"{category}-document-{index // 2}",
                "document_title": f"DOL {category} source {index // 2}",
                "text": (f"{category} priority evidence section {index}. " +
                         "Modernization workforce service budget program integrity. " * 40),
                "reranker_rank": index + 1,
            })
            records.append(base)

    references = compact_strategy_evidence(records,
        focus_terms=("priority", "budget", "program integrity"),
        max_context_tokens=3000)

    assert sum(estimate_tokens(item.excerpt) for item in references) <= 3000
    assert {item.document_type for item in references} == {
        "strategy", "budget", "program-plan", "audit-report"
    }
    assert len(references) < len(records)
    assert all(sum(item.document_id == candidate.document_id for candidate in references) <= 2
               for item in references)


def test_evaluator_rejects_unknown_opportunity_citation():
    """Synthesis cannot cite a chunk that was not retrieved into graph state."""
    state = build_research_graph(FakeLocalEvidenceService()).invoke(_initial_state())
    synthesis = deepcopy(state["synthesis"])
    synthesis["opportunities"][0]["supporting_chunk_ids"].append("chunk-unknown")

    update = independent_evaluator_node(
        {**state, "synthesis": synthesis, "max_iterations": 1}
    )
    assert update["route"] == "human_review"
    assert update["evaluation"]["citation_integrity"] is False


def test_evaluator_rejects_award_that_predates_need_and_rfp():
    """Structured dates, not narrative guesses, control chronology checks."""
    state = build_research_graph(FakeLocalEvidenceService()).invoke(_initial_state())
    synthesis = deepcopy(state["synthesis"])
    for item in synthesis["evidence"]:
        if item["document_type"] == "confirmed-award":
            item["publication_date"] = "2025-01-01"
            item["effective_date"] = "2025-01-01"

    update = independent_evaluator_node(
        {**state, "synthesis": synthesis, "max_iterations": 1}
    )
    assert update["route"] == "human_review"
    assert update["evaluation"]["chronology_valid"] is False


def test_evaluator_rejects_status_contradiction_and_unsupported_claim():
    """A confirmed award cannot be called potential or definitely open."""
    state = build_research_graph(FakeLocalEvidenceService()).invoke(_initial_state())
    synthesis = deepcopy(state["synthesis"])
    modernization = synthesis["opportunities"][0]
    modernization["status"] = "potential"
    modernization["commercial_implications"] = "This is definitely open."

    update = independent_evaluator_node(
        {**state, "synthesis": synthesis, "max_iterations": 1}
    )
    assert update["route"] == "human_review"
    assert update["evaluation"]["contradictions"]
    assert update["evaluation"]["unsupported_commercial_claims"]


def test_evaluator_separates_awarded_scope_from_actionable_scope():
    """Executed core scope cannot also be labeled commercially actionable."""
    state = build_research_graph(FakeLocalEvidenceService()).invoke(_initial_state())
    synthesis = deepcopy(state["synthesis"])
    synthesis["opportunities"][0]["commercially_actionable"] = True

    update = independent_evaluator_node(
        {**state, "synthesis": synthesis, "max_iterations": 1}
    )
    assert update["route"] == "human_review"
    assert "already-awarded scope" in " ".join(
        update["evaluation"]["contradictions"]
    )


def test_evaluator_requires_confirmed_status_and_scope_for_awarded_candidate():
    """An already-awarded label needs both executed status and scope evidence."""
    state = build_research_graph(FakeLocalEvidenceService()).invoke(_initial_state())
    synthesis = deepcopy(state["synthesis"])
    modernization = synthesis["opportunities"][0]
    allowed = {
        item["finding_id"]
        for item in synthesis["findings"]
        if item["finding_type"] in {"mission_priority", "operational_need"}
    }
    modernization["supporting_finding_ids"] = [
        item for item in modernization["supporting_finding_ids"] if item in allowed
    ]

    update = independent_evaluator_node(
        {**state, "synthesis": synthesis, "max_iterations": 1}
    )
    assert update["route"] == "human_review"
    assert update["evaluation"]["evidence_sufficiency"] is False


def test_evaluator_accepts_verified_usaspending_award_only_exclusion():
    """Official confirmed-award wording need not contain the word executed."""
    state = build_research_graph(FakeLocalEvidenceService()).invoke(_initial_state())
    synthesis = deepcopy(state["synthesis"])
    modernization = synthesis["opportunities"][0]
    award_support = {
        item["finding_id"]
        for item in synthesis["findings"]
        if item["finding_type"]
        in {"award_status", "contract_scope", "vendor", "procurement_date"}
        and "document-confirmed-award" in item["document_ids"]
    }
    modernization["supporting_finding_ids"] = [
        item for item in modernization["supporting_finding_ids"] if item in award_support
    ]
    for finding in synthesis["findings"]:
        if (
            finding["finding_id"] in award_support
            and finding["finding_type"] == "award_status"
        ):
            finding["statement"] = (
                "USAspending confirms this as a confirmed federal award record."
            )

    update = independent_evaluator_node(
        {**state, "synthesis": synthesis, "max_iterations": 1}
    )

    assert update["evaluation"]["chronology_valid"] is True
    assert update["evaluation"]["contradictions"] == []
    assert update["evaluation"]["evidence_sufficiency"] is True
    assert any(
        "award-only exclusion" in item
        for item in update["evaluation"]["chronology_notes"]
    )


def test_evaluator_rejects_awarded_conclusion_from_unrelated_contract():
    """An executed record needs positive objective-scope overlap, not just status."""
    state = build_research_graph(FakeLocalEvidenceService()).invoke(_initial_state())
    synthesis = deepcopy(state["synthesis"])
    for evidence in synthesis["evidence"]:
        if evidence["document_type"] == "confirmed-award":
            evidence["document_title"] = "USAspending peripheral purchase"
            evidence["procurement_relevance"] = "unrelated"
            evidence["procurement_relevance_reason"] = "Negative scope relevance."

    update = independent_evaluator_node(
        {**state, "synthesis": synthesis, "max_iterations": 1}
    )

    assert update["route"] == "human_review"
    assert any("already_awarded without" in item for item in update["evaluation"]["contradictions"])


def test_mock_synthesis_does_not_invent_target_opportunity_count():
    """No documented need means the mock returns zero opportunities."""
    model = SyntheticMockAgentModel()
    invocation = model.synthesize(
        objective="Find opportunities.",
        findings=[],
        evidence=[],
        iteration=1,
    )
    assert invocation.output.opportunities == []


def test_mock_model_enforces_call_budget_before_extra_stage():
    """Budget exhaustion becomes a typed human-review result, not a traceback."""
    model = SyntheticMockAgentModel(AgentBudgetTracker(max_calls=2))
    graph = build_research_graph(
        FakeLocalEvidenceService(), agent_model=model
    )
    state = graph.invoke(_initial_state(max_iterations=1))
    result = ResearchGraphResult.model_validate(state["final_result"])
    assert result.status == "human_review"
    assert result.budget_exceeded
    assert result.budget_exceeded[0].stage == "synthesis_agent"
    assert result.budget_exceeded[0].budget_kind == "calls"


def test_input_budget_exhaustion_is_structured_not_an_uncaught_traceback():
    """A rejected oversized researcher payload must still finish as reviewable JSON."""
    model = SyntheticMockAgentModel(
        AgentBudgetTracker(max_calls=6, max_input_tokens=100, max_output_tokens=6000)
    )
    state = build_research_graph(FakeLocalEvidenceService(), agent_model=model).invoke(
        _initial_state(max_iterations=1)
    )
    result = ResearchGraphResult.model_validate(state["final_result"])

    assert result.status == "human_review"
    assert result.budget_exceeded
    assert {item.budget_kind for item in result.budget_exceeded} == {"input_tokens"}
    assert all(item.projected_tokens > item.limit for item in result.budget_exceeded)
    assert all("evidence" not in item.message.casefold() for item in result.budget_exceeded)
