"""Expose the typed contracts and graph runner for the agent foundation.

The :mod:`public_sector_opportunity_research.agents` package is the new
orchestration layer above the existing RAG modules.  It does not replace
ingestion, storage, embeddings, retrieval, assessment, evaluation, or
reporting.  Instead, its placeholder agents call a shared retrieval service
and exchange validated, JSON-serializable records through LangGraph state.

Beginners should read ``contracts.py`` first, then each role module, and finish
with ``graph.py``.  Keeping the public imports here small makes future real
agent implementations easier to substitute without changing CLI callers.
"""

from .contracts import ResearchGraphResult
from .graph import run_offline_research_graph

__all__ = ["ResearchGraphResult", "run_offline_research_graph"]
