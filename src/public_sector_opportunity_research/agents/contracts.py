"""Define the explicit data contracts shared by all graph nodes.

This module is the vocabulary of the working offline LangGraph workflow.  The Research
Manager creates :class:`ResearchTask` records, both researcher nodes return
:class:`ResearcherOutput`, the Synthesis Agent produces
:class:`SynthesisOutput`, and the Independent Evaluator returns
:class:`EvaluationOutput`.  ``graph.py`` stores dictionary versions of these
models in graph state so local checkpoints remain portable and inspectable.

Nothing in this file performs retrieval, calls a model, or writes a database.
Pydantic validates data at node boundaries before the next workflow stage uses
it.  This is especially useful while placeholders are gradually replaced by
live agents: an implementation can change internally without changing its
declared inputs and outputs.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ResearchRole = Literal["strategy", "procurement"]
RouteDecision = Literal["complete", "further_research", "human_review"]
FindingType = Literal[
    "mission_priority",
    "operational_need",
    "budget_evidence",
    "rfp",
    "award_status",
    "contract_scope",
    "vendor",
    "procurement_date",
]
OpportunityStatus = Literal[
    "potential", "already_awarded", "partially_addressed", "insufficient_evidence"
]


class StrictContract(BaseModel):
    """Reject undeclared fields so graph handoffs cannot silently drift."""

    model_config = ConfigDict(extra="forbid")


class ResearchTask(StrictContract):
    """Describe one deterministic assignment created by the manager.

    Attributes:
        task_id: Stable identifier containing the graph iteration and role.
        role: Which specialist is responsible for the task.
        question: Evidence question sent to the shared RAG service.
        document_types: Exact source categories searched for this assignment.
        iteration: One-based research cycle number.

    A future manager may create a richer plan with an LLM, but it should still
    emit this validated contract.  The current placeholder never performs
    autonomous delegation.
    """

    task_id: str
    role: ResearchRole
    question: str
    document_types: list[str]
    iteration: int = Field(ge=1)


class EvidenceRecord(StrictContract):
    """Carry one retrieved passage with complete provenance and ranking data.

    The record mirrors the existing semantic retrieval result and adds the
    trace ID plus a deterministic freshness label.  Full text is retained so
    synthesis can inspect evidence without issuing another query.  All IDs and
    dates originate in SQLite; no placeholder invents citations or timestamps.
    """

    chunk_id: str
    document_id: str
    agency_id: str
    agency_name: str
    document_title: str
    document_type: str
    source_url: str | None = None
    publication_date: date | None = None
    effective_date: date | None = None
    last_updated_date: date | None = None
    capture_date: date
    section_heading: str | None = None
    page_number: int | None = None
    text: str
    vector_rank: int = Field(ge=1)
    vector_similarity_score: float
    reranker_rank: int = Field(ge=1)
    reranker_score: float
    retrieval_trace_id: str
    freshness_status: Literal[
        "not_checked", "undated", "meets_cutoff", "older_than_cutoff"
    ]


class EvidenceReference(StrictContract):
    """Carry compact evidence between agents instead of a complete chunk.

    The researcher sees the complete retrieved chunk only long enough to build
    a focused excerpt.  Downstream state stores this smaller record with stable
    IDs, dates, and ranking metadata.  The 800-character limit bounds prompt
    and checkpoint growth while retaining human-reviewable context.
    """

    chunk_id: str
    document_id: str
    agency_id: str
    document_title: str
    document_type: str
    publication_date: date | None = None
    effective_date: date | None = None
    section_heading: str | None = None
    excerpt: str = Field(min_length=1, max_length=800)
    reranker_rank: int = Field(ge=1)
    retrieval_trace_id: str


class ResearchFinding(StrictContract):
    """Represent one grounded fact extracted by a researcher role."""

    finding_id: str
    finding_type: FindingType
    statement: str = Field(min_length=1, max_length=500)
    chunk_ids: list[str] = Field(min_length=1)
    document_ids: list[str] = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]


class ModelCallRecord(StrictContract):
    """Record one mock-model invocation for future live budget controls."""

    call_id: str
    stage: Literal["strategy_researcher", "procurement_researcher", "synthesis_agent"]
    model_name: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class BudgetSummary(StrictContract):
    """Summarize actual calls/tokens against configured workflow limits."""

    calls_used: int = Field(ge=0)
    max_calls: int = Field(ge=1)
    input_tokens_used: int = Field(ge=0)
    max_input_tokens: int = Field(ge=1)
    output_tokens_used: int = Field(ge=0)
    max_output_tokens: int = Field(ge=1)
    within_budget: bool


class ResearchAnalysis(StrictContract):
    """Validate the compact mock-model response for one researcher."""

    summary: str = Field(min_length=1, max_length=700)
    findings: list[ResearchFinding]


class ResearcherOutput(StrictContract):
    """Represent one researcher's schema-validated mock-LLM response.

    Findings may cite only the accompanying compact evidence IDs.  Missing
    categories identify local research gaps; they never prove that evidence or
    a contract does not exist outside the current corpus.
    """

    role: ResearchRole
    task_id: str
    iteration: int = Field(ge=1)
    summary: str
    evidence: list[EvidenceReference]
    findings: list[ResearchFinding]
    retrieval_trace_ids: list[str]
    requested_categories: list[str]
    missing_categories: list[str]
    implementation: Literal["mock_llm"] = "mock_llm"


class ResearchEvidenceBatch(StrictContract):
    """Return one shared-service lookup across requested document categories."""

    evidence: list[EvidenceRecord]
    retrieval_trace_ids: list[str]
    requested_categories: list[str]
    missing_categories: list[str]


class OpportunityScore(StrictContract):
    """Expose the four transparent factors used for opportunity ranking."""

    mandate: int = Field(ge=0, le=3)
    repeated_evidence: int = Field(ge=0, le=3)
    funding: int = Field(ge=0, le=3)
    commercial_readiness: int = Field(ge=0, le=3)
    total: int = Field(ge=0, le=12)


class OpportunityCandidate(StrictContract):
    """Describe one evidence-grounded enterprise-software opportunity."""

    opportunity_id: str
    title: str = Field(min_length=1, max_length=160)
    need_summary: str = Field(min_length=1, max_length=600)
    status: OpportunityStatus
    rank: int = Field(ge=1)
    score: OpportunityScore
    supporting_finding_ids: list[str] = Field(min_length=1)
    supporting_chunk_ids: list[str] = Field(min_length=1)
    supporting_document_ids: list[str] = Field(min_length=1)
    documented_facts: list[str]
    inferences: list[str]
    commercial_implications: str = Field(min_length=1, max_length=700)
    uncertainties: list[str]


class SynthesisAnalysis(StrictContract):
    """Validate the mock-model response before graph metadata is attached."""

    summary: str = Field(min_length=1, max_length=700)
    opportunities: list[OpportunityCandidate]


class SynthesisOutput(StrictContract):
    """Describe mock-LLM consolidation and ranked grounded opportunities.

    The offline mock merges overlapping findings and emits opportunities only
    when a documented need exists.  It does not target a fixed opportunity
    count and must preserve awarded scope and uncertainty distinctions.
    """

    iteration: int = Field(ge=1)
    evidence: list[EvidenceReference]
    evidence_by_role: dict[ResearchRole, list[str]]
    findings: list[ResearchFinding]
    unresolved_categories: list[str]
    opportunities: list[OpportunityCandidate]
    summary: str
    implementation: Literal["mock_llm"] = "mock_llm"


class EvaluationOutput(StrictContract):
    """Store deterministic grounding/business-rule checks and graph route.

    The evaluator never calls a model.  It verifies citations, minimum evidence
    for each status, structured chronology, contradictions, and unsafe claims
    of commercial availability or proven absence.
    """

    iteration: int = Field(ge=1)
    decision: RouteDecision
    branch_completeness: bool
    citation_integrity: bool
    evidence_sufficiency: bool
    chronology_valid: bool
    chronology_notes: list[str]
    contradictions: list[str]
    unsupported_commercial_claims: list[str]
    unresolved_categories: list[str]
    rationale: str
    implementation: Literal["deterministic"] = "deterministic"


class TraceEvent(StrictContract):
    """Record a compact, deterministic event in the final workflow trace."""

    event_id: str
    iteration: int = Field(ge=1)
    node: str
    detail: str


class ResearchGraphResult(StrictContract):
    """Return the complete offline run in a stable machine-readable shape."""

    thread_id: str
    agency_id: str
    objective: str
    status: RouteDecision
    iterations_completed: int = Field(ge=1)
    tasks: list[ResearchTask]
    researcher_outputs: list[ResearcherOutput]
    synthesis: SynthesisOutput
    evaluation: EvaluationOutput
    retrieval_trace_ids: list[str]
    model_calls: list[ModelCallRecord]
    budget: BudgetSummary
    execution_trace: list[TraceEvent]
    placeholder_notice: str
