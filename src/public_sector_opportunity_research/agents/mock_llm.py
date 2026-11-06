"""Provide reproducible schema-validated mock LLM responses and budgets.

This milestone must exercise the complete multi-agent handoff without making a
live API call.  ``SyntheticMockAgentModel`` therefore behaves like a structured
model boundary: each role sends compact evidence, receives a Pydantic response,
and records estimated input/output tokens.  Its response logic is deliberately
limited to phrases present in the six labeled synthetic documents.

The mock is not a general language model and must not be used on real agencies.
It emits no target number of opportunities: candidates appear only when cited
need evidence matches one of the synthetic demonstration themes.  A future live
implementation can satisfy ``StructuredAgentModel`` while reusing the same
contracts, budget records, nodes, evaluator, and tests.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from threading import Lock
from typing import Generic, Protocol, Sequence, TypeVar

from pydantic import BaseModel

from ..ids import sha256_hex
from .contracts import (
    EvidenceReference,
    ModelCallRecord,
    OpportunityCandidate,
    OpportunityScore,
    ResearchAnalysis,
    ResearchFinding,
    SynthesisAnalysis,
)

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
T = TypeVar("T", bound=BaseModel)


class AgentBudgetExceeded(RuntimeError):
    """Report that the configured call or approximate token budget was exceeded."""


@dataclass(frozen=True)
class ModelInvocation(Generic[T]):
    """Return one validated mock response together with its budget record."""

    output: T
    call: ModelCallRecord


class StructuredAgentModel(Protocol):
    """Declare the three structured model operations used by graph nodes."""

    @property
    def model_name(self) -> str:
        """Return the model identity written to call-budget records."""

    def analyze_strategy(
        self,
        *,
        objective: str,
        evidence: Sequence[EvidenceReference],
        iteration: int,
    ) -> ModelInvocation[ResearchAnalysis]:
        """Extract strategy, need, and budget findings."""

    def analyze_procurement(
        self,
        *,
        objective: str,
        evidence: Sequence[EvidenceReference],
        iteration: int,
    ) -> ModelInvocation[ResearchAnalysis]:
        """Extract solicitation, award, scope, vendor, and date findings."""

    def synthesize(
        self,
        *,
        objective: str,
        findings: Sequence[ResearchFinding],
        evidence: Sequence[EvidenceReference],
        iteration: int,
    ) -> ModelInvocation[SynthesisAnalysis]:
        """Merge findings into ranked evidence-grounded opportunities."""


class AgentBudgetTracker:
    """Enforce and record a process-local call/token budget safely.

    The two researcher nodes can call this object concurrently.  A lock keeps
    cumulative counts consistent.  Token counts are transparent lexical
    estimates, not provider billing tokens.  The mock can calculate output size
    before recording it; a future live implementation should reserve a call and
    input allowance before contacting a provider.
    """

    def __init__(
        self,
        *,
        max_calls: int = 6,
        max_input_tokens: int = 12000,
        max_output_tokens: int = 6000,
    ) -> None:
        """Create a tracker with positive workflow-wide limits."""
        if min(max_calls, max_input_tokens, max_output_tokens) < 1:
            raise ValueError("All agent budget limits must be positive")
        self.max_calls = max_calls
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self._records: dict[str, ModelCallRecord] = {}
        self._lock = Lock()

    def record(
        self,
        *,
        call_id: str,
        stage: str,
        model_name: str,
        input_payload: object,
        output_payload: BaseModel,
    ) -> ModelCallRecord:
        """Validate cumulative limits and return an idempotent call record."""
        input_tokens = _estimate_tokens(_json_text(input_payload))
        output_tokens = _estimate_tokens(output_payload.model_dump_json())
        with self._lock:
            if call_id in self._records:
                return self._records[call_id]
            calls = len(self._records) + 1
            inputs = sum(item.input_tokens for item in self._records.values()) + input_tokens
            outputs = (
                sum(item.output_tokens for item in self._records.values()) + output_tokens
            )
            if calls > self.max_calls:
                raise AgentBudgetExceeded(
                    f"Agent call budget exceeded ({calls}>{self.max_calls})"
                )
            if inputs > self.max_input_tokens:
                raise AgentBudgetExceeded(
                    "Agent input-token budget exceeded "
                    f"({inputs}>{self.max_input_tokens})"
                )
            if outputs > self.max_output_tokens:
                raise AgentBudgetExceeded(
                    "Agent output-token budget exceeded "
                    f"({outputs}>{self.max_output_tokens})"
                )
            record = ModelCallRecord(
                call_id=call_id,
                stage=stage,
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            self._records[call_id] = record
            return record


class SyntheticMockAgentModel:
    """Return deterministic structured responses for the synthetic corpus only."""

    model_name = "synthetic-structured-mock-v1"

    def __init__(self, budget: AgentBudgetTracker | None = None) -> None:
        """Attach a shared budget tracker without loading any external model."""
        self.budget = budget or AgentBudgetTracker()

    def analyze_strategy(
        self,
        *,
        objective: str,
        evidence: Sequence[EvidenceReference],
        iteration: int,
    ) -> ModelInvocation[ResearchAnalysis]:
        """Extract documented strategy, operational, and funding facts."""
        findings: list[ResearchFinding] = []
        for item in evidence:
            text = item.excerpt.casefold()
            if item.document_type == "strategy":
                if "single" in text or "unified" in text:
                    findings.append(
                        _finding(
                            "strategy-unified-access",
                            "mission_priority",
                            "Create a single accessible resident-service experience with mobile, multilingual, and application-status capabilities.",
                            item,
                            "high",
                        )
                    )
                if "api" in text or "nightly" in text or "re-enter" in text:
                    findings.append(
                        _finding(
                            "strategy-integration",
                            "operational_need",
                            "Replace fragmented nightly exchanges and duplicate staff entry with governed APIs, event notifications, and a consolidated service view.",
                            item,
                            "high",
                        )
                    )
                if "wcag" in text or "translated" in text:
                    findings.append(
                        _finding(
                            "strategy-inclusive-delivery",
                            "mission_priority",
                            "Deliver accessible and translated digital services while retaining assisted-service options.",
                            item,
                            "high",
                        )
                    )
            elif item.document_type == "budget" and (
                "$14,000,000" in item.excerpt or "14,000,000" in item.excerpt
            ):
                findings.append(
                    _finding(
                        "budget-digital-renewal",
                        "budget_evidence",
                        "The fictional FY2028 request totals $14.0 million for portal implementation, data/API enablement, assurance, and program staffing, subject to acquisition and architecture approvals.",
                        item,
                        "high",
                    )
                )
            elif item.document_type == "program-plan":
                if "payment accuracy" in text or "program integrity" in text:
                    findings.append(
                        _finding(
                            "integrity-priority",
                            "mission_priority",
                            "Improve benefits payment accuracy and program integrity while distinguishing errors from deliberate fraud and protecting review rights.",
                            item,
                            "high",
                        )
                    )
                if "configurable rules" in text or "entity resolution" in text:
                    findings.append(
                        _finding(
                            "integrity-technology-need",
                            "operational_need",
                            "The plan documents a need for configurable rules, entity resolution, case-review workflow, evidence tracking, audit history, and outcome measurement.",
                            item,
                            "high",
                        )
                    )

        output = ResearchAnalysis(
            summary=(
                f"Extracted {len(findings)} cited strategy, operational-need, and "
                "budget findings from the available synthetic evidence."
            ),
            findings=_deduplicate_findings(findings),
        )
        return self._invocation(
            stage="strategy_researcher",
            iteration=iteration,
            input_payload={"objective": objective, "evidence": evidence},
            output=output,
        )

    def analyze_procurement(
        self,
        *,
        objective: str,
        evidence: Sequence[EvidenceReference],
        iteration: int,
    ) -> ModelInvocation[ResearchAnalysis]:
        """Extract procurement facts while preserving award-status distinctions."""
        findings: list[ResearchFinding] = []
        for item in evidence:
            text = item.excerpt.casefold()
            if item.document_type == "rfp" and (
                "ndrs-2028-017" in text or "resident access" in text
            ):
                findings.extend(
                    [
                        _finding(
                            "rfp-resident-platform",
                            "rfp",
                            "RFP NDRS-2028-017 sought a contractor for a resident access and integration platform.",
                            item,
                            "high",
                        ),
                        _finding(
                            "rfp-resident-scope",
                            "contract_scope",
                            "The solicitation scope included a responsive portal, federated identity, guided intake, document upload, multilingual notices, status tracking, APIs, testing, training, deployment, and support.",
                            item,
                            "high",
                        ),
                    ]
                )
            elif item.document_type == "intent-to-award":
                findings.extend(
                    [
                        _finding(
                            "intent-award-status",
                            "award_status",
                            "A notice stated an intent to award NDRS-2028-017; it was not proof of an executed contract.",
                            item,
                            "high",
                        ),
                        _finding(
                            "intent-award-vendor",
                            "vendor",
                            "The intent notice named Aurora Civic Systems (FICTIONAL) as the selected vendor.",
                            item,
                            "high",
                        ),
                    ]
                )
            elif item.document_type == "confirmed-award" and (
                "executed" in text or "contract status" in text
            ):
                findings.extend(
                    [
                        _finding(
                            "confirmed-award-status",
                            "award_status",
                            "Executed contract NDRS-2028-017-C1 was confirmed as signed by both fictional parties on 2028-02-15.",
                            item,
                            "high",
                        ),
                        _finding(
                            "confirmed-award-scope",
                            "contract_scope",
                            "The executed scope covers the resident portal, identity, guided intake, document exchange, multilingual communications, status tracking, APIs, assurance, training, and phased rollout.",
                            item,
                            "high",
                        ),
                        _finding(
                            "confirmed-award-vendor",
                            "vendor",
                            "Aurora Civic Systems (FICTIONAL) is the vendor named in executed contract NDRS-2028-017-C1.",
                            item,
                            "high",
                        ),
                        _finding(
                            "confirmed-award-date",
                            "procurement_date",
                            "The fictional contract execution and award date is 2028-02-15; publication followed on 2028-02-20.",
                            item,
                            "high",
                        ),
                    ]
                )

        output = ResearchAnalysis(
            summary=(
                f"Extracted {len(findings)} cited RFP, award, scope, vendor, and "
                "date findings from the available synthetic evidence."
            ),
            findings=_deduplicate_findings(findings),
        )
        return self._invocation(
            stage="procurement_researcher",
            iteration=iteration,
            input_payload={"objective": objective, "evidence": evidence},
            output=output,
        )

    def synthesize(
        self,
        *,
        objective: str,
        findings: Sequence[ResearchFinding],
        evidence: Sequence[EvidenceReference],
        iteration: int,
    ) -> ModelInvocation[SynthesisAnalysis]:
        """Merge synthetic themes and rank only opportunities supported by needs."""
        candidates: list[OpportunityCandidate] = []
        modernization = _theme_findings(
            findings, ("resident", "portal", "integration", "digital service")
        )
        integrity = _theme_findings(
            findings, ("payment accuracy", "program integrity", "entity resolution")
        )

        if _contains_need(modernization):
            candidates.append(_modernization_opportunity(modernization))
        if _contains_need(integrity):
            candidates.append(_integrity_opportunity(integrity))

        # Rank by the declared score factors only; no candidate is added merely
        # to reach a preferred count.
        ordered = sorted(
            candidates,
            key=lambda item: (-item.score.total, item.opportunity_id),
        )
        ranked = [item.model_copy(update={"rank": index}) for index, item in enumerate(ordered, 1)]
        output = SynthesisAnalysis(
            summary=(
                f"Merged {len(findings)} grounded findings into {len(ranked)} "
                "opportunity candidate(s); awarded scope remains explicitly labeled."
            ),
            opportunities=ranked,
        )
        return self._invocation(
            stage="synthesis_agent",
            iteration=iteration,
            input_payload={
                "objective": objective,
                "findings": findings,
                "evidence_ids": [item.chunk_id for item in evidence],
            },
            output=output,
        )

    def _invocation(
        self,
        *,
        stage: str,
        iteration: int,
        input_payload: object,
        output: T,
    ) -> ModelInvocation[T]:
        """Attach one validated response to an idempotent budget record."""
        call_id = f"{iteration}:{stage}"
        record = self.budget.record(
            call_id=call_id,
            stage=stage,
            model_name=self.model_name,
            input_payload=input_payload,
            output_payload=output,
        )
        return ModelInvocation(output=output, call=record)


def _finding(
    label: str,
    finding_type: str,
    statement: str,
    evidence: EvidenceReference,
    confidence: str,
) -> ResearchFinding:
    """Build a stable cited finding from one compact evidence reference."""
    return ResearchFinding(
        finding_id=f"finding_{sha256_hex(label)[:16]}",
        finding_type=finding_type,
        statement=statement,
        chunk_ids=[evidence.chunk_id],
        document_ids=[evidence.document_id],
        confidence=confidence,
    )


def _deduplicate_findings(findings: Sequence[ResearchFinding]) -> list[ResearchFinding]:
    """Return one finding per stable ID in deterministic order."""
    by_id = {item.finding_id: item for item in findings}
    return [by_id[key] for key in sorted(by_id)]


def _theme_findings(
    findings: Sequence[ResearchFinding], terms: Sequence[str]
) -> list[ResearchFinding]:
    """Select findings whose statements explicitly mention a synthetic theme."""
    return [
        item
        for item in findings
        if any(term in item.statement.casefold() for term in terms)
    ]


def _contains_need(findings: Sequence[ResearchFinding]) -> bool:
    """Require a mission or operational need before creating an opportunity."""
    return any(
        item.finding_type in {"mission_priority", "operational_need"}
        for item in findings
    )


def _modernization_opportunity(
    findings: Sequence[ResearchFinding],
) -> OpportunityCandidate:
    """Build the synthetic resident-platform candidate from cited findings."""
    confirmed = any(
        item.finding_type == "award_status" and "executed contract" in item.statement.casefold()
        for item in findings
    )
    funding = any(item.finding_type == "budget_evidence" for item in findings)
    rfp = any(item.finding_type == "rfp" for item in findings)
    document_ids = _unique(value for item in findings for value in item.document_ids)
    score = OpportunityScore(
        mandate=3,
        repeated_evidence=min(3, len(document_ids)),
        funding=3 if funding else 0,
        commercial_readiness=0 if confirmed else (3 if rfp else (1 if funding else 0)),
        total=0,
    )
    score = score.model_copy(
        update={
            "total": score.mandate
            + score.repeated_evidence
            + score.funding
            + score.commercial_readiness
        }
    )
    status = "already_awarded" if confirmed else "potential"
    return OpportunityCandidate(
        opportunity_id="opp_resident_access_platform",
        title="Resident access and integration platform modernization",
        need_summary=(
            "NDRS documents a unified, accessible resident-service need, legacy "
            "integration problems, and supporting fictional budget/procurement evidence."
        ),
        status=status,
        rank=1,
        score=score,
        supporting_finding_ids=sorted(item.finding_id for item in findings),
        supporting_chunk_ids=_unique(
            value for item in findings for value in item.chunk_ids
        ),
        supporting_document_ids=document_ids,
        documented_facts=[item.statement for item in findings[:6]],
        inferences=(
            [
                "The documented need and executed scope substantially overlap, so the identified platform implementation is treated as already awarded."
            ]
            if confirmed
            else [
                "The evidence indicates a potential need, but the local corpus does not establish whether procurement is currently open."
            ]
        ),
        commercial_implications=(
            "The core platform scope is not treated as a new available opportunity because an executed synthetic contract covers substantially the same work."
            if confirmed
            else "The documented need warrants procurement-status research; missing award evidence would not prove that no contract exists."
        ),
        uncertainties=(
            [
                "The corpus does not establish future amendments, recompetes, performance gaps, or out-of-scope follow-on work."
            ]
            if confirmed
            else [
                "The corpus does not prove an open solicitation or the absence of another contract."
            ]
        ),
    )


def _integrity_opportunity(findings: Sequence[ResearchFinding]) -> OpportunityCandidate:
    """Build the synthetic benefits-integrity candidate without claiming availability."""
    document_ids = _unique(value for item in findings for value in item.document_ids)
    score = OpportunityScore(
        mandate=3,
        repeated_evidence=min(3, len(document_ids)),
        funding=0,
        commercial_readiness=0,
        total=3 + min(3, len(document_ids)),
    )
    return OpportunityCandidate(
        opportunity_id="opp_benefits_integrity_tooling",
        title="Benefits payment accuracy and program-integrity tooling",
        need_summary=(
            "The plan documents payment-accuracy, fraud-risk, configurable-rules, "
            "entity-resolution, review-workflow, evidence, audit, and measurement needs."
        ),
        status="potential",
        rank=1,
        score=score,
        supporting_finding_ids=sorted(item.finding_id for item in findings),
        supporting_chunk_ids=_unique(
            value for item in findings for value in item.chunk_ids
        ),
        supporting_document_ids=document_ids,
        documented_facts=[item.statement for item in findings],
        inferences=[
            "The documented technology need could align with enterprise rules, analytics, entity-resolution, case-workflow, and audit capabilities."
        ],
        commercial_implications=(
            "Treat this as a research lead, not a known open procurement: the current corpus contains no linked budget, RFP, or confirmed award for this scope."
        ),
        uncertainties=[
            "Absent linked procurement evidence is not proof that no contract exists.",
            "No dedicated funding or acquisition timeline is documented in the available evidence.",
        ],
    )


def _unique(values) -> list[str]:
    """Return strings once, preserving their first evidence order."""
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _estimate_tokens(text: str) -> int:
    """Estimate tokens with the project's transparent lexical convention."""
    return len(_TOKEN_PATTERN.findall(text))


def _json_text(value: object) -> str:
    """Serialize Pydantic-rich payloads consistently for token estimation."""
    def default(item: object):
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json")
        return str(item)

    return json.dumps(value, default=default, sort_keys=True)
