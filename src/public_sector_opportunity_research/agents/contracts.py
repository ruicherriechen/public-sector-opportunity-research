"""Define the explicit data contracts shared by all graph nodes.

This module is the vocabulary of the working offline LangGraph workflow.  The Research
Manager creates :class:`ResearchTask` records, both researcher nodes return
:class:`ResearcherOutput`, the Synthesis Agent produces
:class:`SynthesisOutput`, and the Independent Evaluator returns
:class:`EvaluationOutput`.  ``graph.py`` stores dictionary versions of these
models in graph state so local checkpoints remain portable and inspectable.

Nothing in this file performs retrieval, calls a model, or writes a database.
Pydantic validates data at node boundaries before the next workflow stage uses
it.  This is especially useful while the synthetic mock is gradually replaced
by a live model adapter: an implementation can change internally without changing its
declared inputs and outputs.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ResearchRole = Literal["strategy", "procurement"]
RouteDecision = Literal["complete", "further_research", "human_review"]
ModelProvider = Literal["mock", "openai", "codex"]
WorkflowProvider = Literal["mock", "openai", "codex", "mixed"]
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
OpportunityReadiness = Literal[
    "potentially_commercially_actionable",
    "confirmed_procurement_opportunity",
    "already_awarded",
    "research_lead",
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
    emit this validated contract.  The current deterministic manager never performs
    autonomous delegation.
    """

    task_id: str
    role: ResearchRole
    question: str
    document_types: list[str]
    iteration: int = Field(ge=1)


class PriorHumanFeedback(StrictContract):
    """Carry compact reviewer guidance without treating it as source evidence.

    Feedback has no chunk or document citation fields. Researchers may use it
    to focus retrieval and check a correction, but every resulting finding must
    still cite ordinary retrieved government evidence.
    """

    feedback_id: str
    opportunity_id: str
    feedback_type: Literal["agree", "disagree"]
    guidance: str = Field(min_length=1, max_length=800)
    applies_to: list[ResearchRole]
    reviewed_at: str


class EvidenceRecord(StrictContract):
    """Carry one retrieved passage with complete provenance and ranking data.

    The record mirrors the existing semantic retrieval result and adds the
    trace ID plus a deterministic freshness label. Full text exists only at
    this retrieval/researcher boundary; ``EvidenceReference`` carries a bounded
    excerpt into synthesis. All IDs and dates originate in SQLite; the mock
    never invents citations or timestamps.
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
    vector_rank: int | None = Field(default=None, ge=1)
    vector_similarity_score: float | None = None
    reranker_rank: int = Field(ge=1)
    reranker_score: float
    retrieval_trace_id: str
    freshness_status: Literal[
        "not_checked", "undated", "meets_cutoff", "older_than_cutoff"
    ]
    retrieval_method: Literal["semantic", "metadata_supplement"] = "semantic"
    metadata_supplement_reason: str | None = None
    original_reranker_rank: int | None = Field(default=None, ge=1)
    procurement_relevance: Literal["relevant", "unrelated", "unknown"] = "unknown"
    procurement_relevance_reason: str | None = None
    content_trust: Literal["untrusted_external_evidence"] = "untrusted_external_evidence"
    suspicious_indicators: list[str] = Field(default_factory=list)
    secret_like_values_redacted: list[str] = Field(default_factory=list)


class EvidenceReference(StrictContract):
    """Carry compact evidence between agents instead of a complete chunk.

    The researcher sees the complete retrieved chunk only long enough to build
    a focused excerpt.  Downstream state stores this smaller record with stable
    IDs, dates, and ranking metadata. Strategy excerpts remain capped at 800
    characters. Procurement may retain complete short source chunks, while a
    separate 2,000--4,000-token aggregate budget bounds prompt/checkpoint growth.
    """

    chunk_id: str
    document_id: str
    agency_id: str
    document_title: str
    document_type: str
    publication_date: date | None = None
    effective_date: date | None = None
    last_updated_date: date | None = None
    capture_date: date | None = None
    section_heading: str | None = None
    excerpt: str = Field(min_length=1, max_length=24000)
    reranker_rank: int = Field(ge=1)
    retrieval_trace_id: str
    retrieval_method: Literal["semantic", "metadata_supplement"] = "semantic"
    metadata_supplement_reason: str | None = None
    original_reranker_rank: int | None = Field(default=None, ge=1)
    procurement_relevance: Literal["relevant", "unrelated", "unknown"] = "unknown"
    procurement_relevance_reason: str | None = None
    content_trust: Literal["untrusted_external_evidence"] = "untrusted_external_evidence"
    suspicious_indicators: list[str] = Field(default_factory=list)
    secret_like_values_redacted: list[str] = Field(default_factory=list)


class ResearchFinding(StrictContract):
    """Represent one grounded fact extracted by a researcher role."""

    finding_id: str
    finding_type: FindingType
    statement: str = Field(min_length=1, max_length=500)
    chunk_ids: list[str] = Field(min_length=1)
    document_ids: list[str] = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]


class ModelCallRecord(StrictContract):
    """Record one structured-model invocation and its token accounting."""

    call_id: str
    stage: Literal["strategy_researcher", "procurement_researcher", "synthesis_agent"]
    provider: ModelProvider = "mock"
    model_name: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    usage_source: Literal["provider", "estimated"] = "estimated"
    status: Literal["succeeded", "failed", "timed_out"] = "succeeded"
    retry_count: int = Field(default=0, ge=0)
    error_category: str | None = None


class BudgetExceededDiagnostic(StrictContract):
    """Describe a rejected model call without exposing its prompt or evidence."""

    call_id: str
    stage: Literal["strategy_researcher", "procurement_researcher", "synthesis_agent"]
    budget_kind: Literal["calls", "input_tokens", "output_tokens"]
    attempted_tokens: int = Field(ge=0)
    used_tokens: int = Field(ge=0)
    projected_tokens: int = Field(ge=0)
    limit: int = Field(ge=1)
    message: str


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
    """Validate the compact structured-model response for one researcher."""

    summary: str = Field(min_length=1, max_length=700)
    findings: list[ResearchFinding]


class ResearcherOutput(StrictContract):
    """Represent one researcher's schema-validated model response.

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
    coverage_gaps: list[str] = Field(default_factory=list)
    implementation: Literal["mock", "mock_llm", "openai", "codex", "mixed"] = "mock"


class ResearchEvidenceBatch(StrictContract):
    """Return one shared-service lookup across requested document categories."""

    evidence: list[EvidenceRecord]
    retrieval_trace_ids: list[str]
    requested_categories: list[str]
    missing_categories: list[str]
    coverage_gaps: list[str] = Field(default_factory=list)


class OpportunityScore(StrictContract):
    """Expose the four transparent factors used for opportunity ranking."""

    mandate: int = Field(ge=0, le=3)
    repeated_evidence: int = Field(ge=0, le=3)
    funding: int = Field(ge=0, le=3)
    commercial_readiness: int = Field(ge=0, le=3)
    total: int = Field(ge=0, le=12)


class OpportunityCandidate(StrictContract):
    """Describe one evidence-grounded enterprise-software opportunity.

    ``commercially_actionable`` remains a boolean for compatibility with saved
    runs and the review dashboard. ``readiness`` explains what that boolean
    means: a cited open procurement, an evidence-supported potential sales
    lead, awarded scope, or a lead needing more research. Older saved results
    load as ``research_lead`` until they are deterministically re-evaluated.
    """

    opportunity_id: str
    title: str = Field(min_length=1, max_length=160)
    need_summary: str = Field(min_length=1, max_length=600)
    status: OpportunityStatus
    rank: int = Field(ge=1)
    score: OpportunityScore
    commercially_actionable: bool
    readiness: OpportunityReadiness
    supporting_finding_ids: list[str] = Field(min_length=1)
    supporting_chunk_ids: list[str] = Field(min_length=1)
    supporting_document_ids: list[str] = Field(min_length=1)
    documented_facts: list[str]
    inferences: list[str]
    commercial_implications: str = Field(min_length=1, max_length=700)
    uncertainties: list[str]

    @model_validator(mode="before")
    @classmethod
    def load_legacy_readiness(cls, value):
        """Load pre-policy saved JSON without weakening the strict live schema.

        OpenAI Structured Outputs requires every declared property to appear in
        the schema's ``required`` list, so ``readiness`` has no field default.
        This compatibility shim supplies ``research_lead`` only when reading an
        older dictionary that predates the field.
        """
        if isinstance(value, dict) and "readiness" not in value:
            return {**value, "readiness": "research_lead"}
        return value


class SynthesisAnalysis(StrictContract):
    """Validate a structured synthesis response before metadata is attached."""

    summary: str = Field(min_length=1, max_length=700)
    opportunities: list[OpportunityCandidate]


class SynthesisOutput(StrictContract):
    """Describe structured consolidation and ranked grounded opportunities.

    Providers merge overlapping findings and emit opportunities only when a
    documented need exists. They do not target a fixed opportunity count and
    must preserve awarded scope and uncertainty distinctions.
    """

    iteration: int = Field(ge=1)
    evidence: list[EvidenceReference]
    evidence_by_role: dict[ResearchRole, list[str]]
    findings: list[ResearchFinding]
    unresolved_categories: list[str]
    coverage_gaps: list[str] = Field(default_factory=list)
    opportunities: list[OpportunityCandidate]
    summary: str
    implementation: Literal["mock", "mock_llm", "openai", "codex", "mixed"] = "mock"


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
    grounding_review_required: list[str] = Field(default_factory=list)
    evidence_sufficiency: bool
    chronology_valid: bool
    chronology_notes: list[str]
    contradictions: list[str]
    unsupported_commercial_claims: list[str]
    unresolved_categories: list[str]
    coverage_gaps: list[str] = Field(default_factory=list)
    coverage_by_stage: dict[str, list[str]] = Field(default_factory=dict)
    security_warnings: list[str] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)
    rationale: str
    implementation: Literal["deterministic"] = "deterministic"


class TraceEvent(StrictContract):
    """Record a compact, deterministic event in the final workflow trace."""

    event_id: str
    iteration: int = Field(ge=1)
    node: str
    detail: str


class ResearchGraphResult(StrictContract):
    """Return the complete graph run in a stable machine-readable shape."""

    thread_id: str
    agency_id: str
    objective: str
    model_provider: WorkflowProvider = "mock"
    status: RouteDecision
    iterations_completed: int = Field(ge=1)
    tasks: list[ResearchTask]
    researcher_outputs: list[ResearcherOutput]
    synthesis: SynthesisOutput
    evaluation: EvaluationOutput
    retrieval_trace_ids: list[str]
    model_calls: list[ModelCallRecord]
    budget: BudgetSummary
    budget_exceeded: list[BudgetExceededDiagnostic] = Field(default_factory=list)
    execution_trace: list[TraceEvent]
    placeholder_notice: str
    prior_human_feedback: list[PriorHumanFeedback] = Field(default_factory=list)
