"""Route the three model-backed graph stages to explicit providers.

The LangGraph topology remains unchanged: Strategy, Procurement, and Synthesis
still call one ``StructuredAgentModel`` interface.  This small facade delegates
each method to the provider chosen for that stage, which permits a hybrid run
without teaching graph nodes about OpenAI, Codex, or authentication.

Provider adapters share one workflow budget tracker.  Routing occurs after CLI
configuration and before graph construction; model-call records emitted by the
selected adapter preserve the actual stage provider and model identity.
"""

from __future__ import annotations

from typing import Sequence

from .contracts import EvidenceReference, ResearchAnalysis, ResearchFinding, SynthesisAnalysis
from .mock_llm import ModelInvocation, StructuredAgentModel


class StageRoutedAgentModel:
    """Delegate each structured operation to its explicitly selected adapter."""

    def __init__(
        self,
        *,
        strategy: StructuredAgentModel,
        procurement: StructuredAgentModel,
        synthesis: StructuredAgentModel,
    ) -> None:
        """Store the three stage adapters without invoking any provider."""
        self._strategy = strategy
        self._procurement = procurement
        self._synthesis = synthesis

    @property
    def provider_name(self) -> str:
        """Return one provider name, or ``mixed`` for a hybrid workflow."""
        names = {
            self._strategy.provider_name,
            self._procurement.provider_name,
            self._synthesis.provider_name,
        }
        return next(iter(names)) if len(names) == 1 else "mixed"

    @property
    def model_name(self) -> str:
        """Return a concise workflow model label for protocol compatibility."""
        names = {
            self._strategy.model_name,
            self._procurement.model_name,
            self._synthesis.model_name,
        }
        return next(iter(names)) if len(names) == 1 else "stage-routed"

    def analyze_strategy(
        self, *, objective: str, evidence: Sequence[EvidenceReference], iteration: int
    ) -> ModelInvocation[ResearchAnalysis]:
        """Send Strategy input only to the configured Strategy adapter."""
        return self._strategy.analyze_strategy(
            objective=objective, evidence=evidence, iteration=iteration
        )

    def analyze_procurement(
        self, *, objective: str, evidence: Sequence[EvidenceReference], iteration: int
    ) -> ModelInvocation[ResearchAnalysis]:
        """Send Procurement input only to the configured Procurement adapter."""
        return self._procurement.analyze_procurement(
            objective=objective, evidence=evidence, iteration=iteration
        )

    def synthesize(
        self,
        *,
        objective: str,
        findings: Sequence[ResearchFinding],
        evidence: Sequence[EvidenceReference],
        iteration: int,
    ) -> ModelInvocation[SynthesisAnalysis]:
        """Send validated findings only to the configured Synthesis adapter."""
        return self._synthesis.synthesize(
            objective=objective,
            findings=findings,
            evidence=evidence,
            iteration=iteration,
        )
