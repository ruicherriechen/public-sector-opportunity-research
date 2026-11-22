"""Build truthful architecture and saved-run execution views for Streamlit.

This module contains no Streamlit calls. It describes the implemented graph,
turns that structure into Graphviz DOT, and derives per-iteration execution
facts only from ``ResearchGraphResult`` fields. The dashboard renders these
small data structures on its Architecture & Execution page.

Architecture precedes execution: ``agents.graph`` defines the fixed nodes and
edges, while a saved result records trace events, researcher outputs, model
calls, budget failures, and the final evaluator route. Missing timestamps or
checkpoint snapshots remain explicit limitations rather than being inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .agents.contracts import ResearchGraphResult


@dataclass(frozen=True)
class ArchitectureNode:
    """Describe one selectable agent, terminal, or supporting component."""

    node_id: str
    label: str
    category: str
    responsibility: str
    inputs: str
    outputs: str
    uses_llm: str
    design_decision: str


@dataclass(frozen=True)
class ExecutionStage:
    """Summarize verifiable activity for one node in one graph iteration."""

    iteration: int
    node: str
    label: str
    status: str
    detail: str | None
    evidence_count: int | None = None
    findings_count: int | None = None
    model_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    providers: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    missing_categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionView:
    """Carry a run's trace-derived stages and explicitly missing metadata."""

    stages: tuple[ExecutionStage, ...]
    route_by_iteration: dict[int, str]
    limitations: tuple[str, ...]


ARCHITECTURE_NODES = (
    ArchitectureNode(
        "sources", "Government sources + USAspending", "supporting",
        "Provide official documents and structured federal award records.",
        "Explicit bounded discovery requests or configured source URLs.",
        "Verified content sent through normal ingestion and provenance checks.",
        "No",
        "Search candidates are not evidence until verified and stored.",
    ),
    ArchitectureNode(
        "rag", "SQLite evidence + local RAG", "supporting",
        "Store source text/provenance and retrieve relevant cited passages.",
        "Agency, research query, source category, and optional date filters.",
        "BGE/sqlite-vec candidates, MiniLM ranks, excerpts, IDs, and traces.",
        "No generative LLM",
        "Durable evidence is separate from transient graph/checkpoint state.",
    ),
    ArchitectureNode(
        "manager", "Research Manager", "deterministic",
        "Validate the objective, increment the bounded iteration, and create two specialist assignments.",
        "Agency, objective, iteration limit, and compact prior human feedback.",
        "Strategy and procurement tasks with exact source categories.",
        "No",
        "Planning remains deterministic so delegation is inspectable and bounded.",
    ),
    ArchitectureNode(
        "strategy", "Strategy Researcher", "llm",
        "Extract cited mission priorities, operational needs, and budget evidence.",
        "Strategy, budget, program-plan, and audit excerpts from local RAG.",
        "Compact structured findings with source chunk/document IDs.",
        "Yes in OpenAI mode; deterministic mock offline",
        "Only bounded excerpts enter the model; full documents remain in SQLite.",
    ),
    ArchitectureNode(
        "procurement", "Procurement Researcher", "llm",
        "Extract RFP, award status, scope, vendor, value, and date facts.",
        "RFP, intent-to-award, and confirmed-award evidence from local RAG.",
        "Cited structured procurement findings and explicit coverage gaps.",
        "Yes in OpenAI mode; deterministic mock offline",
        "Intent and executed awards remain distinct; unknown facts stay unknown.",
    ),
    ArchitectureNode(
        "synthesis", "Synthesis Agent", "llm",
        "Merge supported findings and rank concrete opportunity candidates.",
        "Both researcher outputs; no full documents or hidden conversations.",
        "Opportunity assessments, facts, inferences, scores, uncertainty, and citations.",
        "Yes in OpenAI mode; deterministic mock offline",
        "Deterministic Python scoring is reapplied after structured reasoning.",
    ),
    ArchitectureNode(
        "evaluator", "Independent Evaluator", "deterministic",
        "Check citations, sufficiency, chronology, contradictions, and unsupported commercial claims.",
        "Structured synthesis, evidence IDs/dates, and configured iteration limit.",
        "Complete, further-research, or human-review route with exact reasons.",
        "No",
        "It is independent of the synthesis prompt and never treats missing evidence as proof of absence.",
    ),
    ArchitectureNode(
        "complete", "Complete", "terminal",
        "End a run whose deterministic checks pass with no unresolved category.",
        "Evaluator route: complete.",
        "Schema-validated terminal result.",
        "No",
        "Completion is a graph outcome, not proof of an open procurement.",
    ),
    ArchitectureNode(
        "human_review", "Human Review", "human",
        "Let a reviewer inspect evidence, record judgment, and optionally request one follow-up.",
        "Saved result, SQLite evidence/traces, evaluator reasons, and procurement records.",
        "Separate decision/feedback rows; original research remains immutable.",
        "No",
        "Agreement is evidence-bound feedback, not model-weight training or commercial proof.",
    ),
    ArchitectureNode(
        "feedback", "Human feedback memory", "supporting",
        "Reuse compact reviewer guidance for the same agency/opportunity.",
        "Agree/disagree snapshot and exact assessment/evidence fingerprint.",
        "At most three non-citable guidance records for future planning/research.",
        "No",
        "Feedback guides retrieval but can never substitute for official-source citations.",
    ),
)


def architecture_node(node_id: str) -> ArchitectureNode:
    """Return one declared component or raise a clear error."""
    try:
        return next(item for item in ARCHITECTURE_NODES if item.node_id == node_id)
    except StopIteration as exc:
        raise ValueError(f"Unknown architecture node: {node_id}") from exc


def architecture_dot(selected_node: str | None = None) -> str:
    """Return a compact top-to-bottom diagram matching the implemented graph.

    Top-to-bottom flow gives the two parallel researcher boxes enough width for
    readable labels without forcing the whole dashboard to scroll sideways.
    Every implemented edge remains present; rank hints affect layout only.
    """
    lines = [
        "digraph architecture {",
        'graph [rankdir=TB, bgcolor="transparent", pad="0.25", nodesep="0.55", ranksep="0.68", splines=ortho, newrank=true];',
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=14, color="#a1a1aa", penwidth=1.2, margin="0.22,0.14", width=2.15];',
        'edge [fontname="Arial", fontsize=10, color="#71717a", fontcolor="#52525b", arrowsize=0.75, penwidth=1.15];',
    ]
    colors = {
        "llm": "#dbeafe",
        "deterministic": "#f1f5f9",
        "supporting": "#f8fafc",
        "terminal": "#dcfce7",
        "human": "#f3e8ff",
    }
    for item in ARCHITECTURE_NODES:
        penwidth = "2.4" if item.node_id == selected_node else "1.2"
        color = "#2563eb" if item.node_id == selected_node else "#94a3b8"
        lines.append(
            f'{item.node_id} [label="{_dot_escape(_architecture_label(item))}", '
            f'fillcolor="{colors[item.category]}", color="{color}", penwidth={penwidth}];'
        )
    lines.extend(
        [
            # Evidence and planning sit on one level above the two researcher
            # branches. This keeps supporting systems secondary without hiding
            # their real handoff edges.
            '{rank=same; rag; manager;}',
            'sources -> rag [label="verified documents\\n+ awards"];',
            'rag -> strategy [label="cited strategy\\nexcerpts"];',
            'rag -> procurement [label="cited procurement\\nrecords"];',
            'feedback -> manager [label="non-citable guidance", style=dashed];',
            'manager -> strategy [label="strategy task"];',
            'manager -> procurement [label="procurement task"];',
            '{rank=same; strategy; procurement;}',
            'strategy -> synthesis [label="grounded findings"];',
            'procurement -> synthesis [label="grounded findings + gaps"];',
            'synthesis -> evaluator [label="ranked opportunities + citations"];',
            'evaluator -> complete [label="checks pass", color="#16a34a"];',
            'evaluator -> human_review [label="review required", color="#7c3aed"];',
            '{rank=same; complete; human_review;}',
            'evaluator -> manager [label="further research\\n(bounded)", style=dashed, color="#d97706", constraint=false];',
            'human_review -> feedback [label="explicit reviewer feedback", style=dashed];',
            "}",
        ]
    )
    return "\n".join(lines)


def _architecture_label(node: ArchitectureNode) -> str:
    """Wrap only long support labels while preserving concise agent names."""
    return {
        "sources": "Government sources\n+ USAspending",
        "rag": "SQLite evidence\n+ local RAG",
        "human_review": "Human Review",
        "feedback": "Human feedback\nmemory",
    }.get(node.node_id, node.label)


def build_execution_view(result: ResearchGraphResult) -> ExecutionView:
    """Derive execution stages only from saved records and trace events."""
    events = {(item.iteration, item.node): item for item in result.execution_trace}
    outputs = {
        (item.iteration, item.role): item for item in result.researcher_outputs
    }
    calls: dict[tuple[int, str], list] = {}
    unassigned_calls = 0
    for call in result.model_calls:
        iteration = _call_iteration(call.call_id)
        if iteration is None:
            unassigned_calls += 1
            continue
        calls.setdefault((iteration, call.stage), []).append(call)
    budget = {(item.call_id, item.stage): item for item in result.budget_exceeded}
    stages: list[ExecutionStage] = []
    routes: dict[int, str] = {}

    for iteration in range(1, result.iterations_completed + 1):
        for node, label in (
            ("research_manager", "Research Manager"),
            ("strategy_researcher", "Strategy Researcher"),
            ("procurement_researcher", "Procurement Researcher"),
            ("synthesis_agent", "Synthesis Agent"),
            ("independent_evaluator", "Independent Evaluator"),
        ):
            event = events.get((iteration, node))
            # Budget event IDs use an extra suffix while retaining the same node.
            if event is None:
                event = next(
                    (
                        item for item in result.execution_trace
                        if item.iteration == iteration and item.node == node
                    ),
                    None,
                )
            role = {
                "strategy_researcher": "strategy",
                "procurement_researcher": "procurement",
            }.get(node)
            output = outputs.get((iteration, role)) if role else None
            stage_calls = calls.get((iteration, node), [])
            diagnostic = next(
                (
                    item for (call_id, stage), item in budget.items()
                    if stage == node and _call_iteration(call_id) == iteration
                ),
                None,
            )
            detail = event.detail if event else None
            status = _execution_status(detail, diagnostic is not None)
            evidence_count = len(output.evidence) if output is not None else None
            findings_count = len(output.findings) if output is not None else None
            if node == "synthesis_agent" and detail:
                findings_count = _number_from_detail(detail, r"Merged (\d+) findings")
            if node == "independent_evaluator" and detail:
                route = _route_from_detail(detail)
                if route:
                    routes[iteration] = route
            missing = tuple(output.missing_categories) if output is not None else ()
            stages.append(
                ExecutionStage(
                    iteration=iteration,
                    node=node,
                    label=label,
                    status=status,
                    detail=detail,
                    evidence_count=evidence_count,
                    findings_count=findings_count,
                    model_calls=len(stage_calls),
                    input_tokens=(sum(item.input_tokens for item in stage_calls) if stage_calls else None),
                    output_tokens=(sum(item.output_tokens for item in stage_calls) if stage_calls else None),
                    providers=tuple(sorted({item.provider for item in stage_calls})),
                    models=tuple(sorted({item.model_name for item in stage_calls})),
                    missing_categories=missing,
                )
            )

    terminal = next(
        (
            item for item in result.execution_trace
            if item.node in {"human_review", "complete"}
        ),
        None,
    )
    if terminal:
        stages.append(
            ExecutionStage(
                terminal.iteration,
                terminal.node,
                "Human Review" if terminal.node == "human_review" else "Complete",
                "needs_human_review" if terminal.node == "human_review" else "completed",
                terminal.detail,
            )
        )
    limitations = [
        "Node start/end timestamps and durations were not recorded.",
        "Strategy and Procurement are known parallel branches; their within-step completion order is not recorded.",
        "Checkpoint snapshots are stored separately and their database path is not embedded in this result.",
    ]
    if unassigned_calls:
        limitations.append(
            f"{unassigned_calls} model call(s) lack an iteration prefix and cannot be assigned to a visual stage."
        )
    return ExecutionView(tuple(stages), routes, tuple(limitations))


def execution_dot(result: ResearchGraphResult, view: ExecutionView) -> str:
    """Render the verified per-iteration path without implying timing."""
    by_key = {(item.iteration, item.node): item for item in view.stages}
    lines = [
        "digraph execution {",
        'graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.6", splines=ortho];',
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10, color="#94a3b8", margin="0.10,0.07"];',
        'edge [fontname="Arial", fontsize=9, color="#64748b", fontcolor="#475569", arrowsize=0.7];',
    ]
    for iteration in range(1, result.iterations_completed + 1):
        lines.append(f'subgraph cluster_{iteration} {{ label="Iteration {iteration}"; color="#cbd5e1"; style="rounded";')
        for node in (
            "research_manager", "strategy_researcher", "procurement_researcher",
            "synthesis_agent", "independent_evaluator",
        ):
            stage = by_key[(iteration, node)]
            node_id = f"i{iteration}_{node}"
            lines.append(
                f'{node_id} [label="{_dot_escape(_execution_label(stage))}", '
                f'fillcolor="{_status_color(stage.status)}", color="{_status_border(stage.status)}"];'
            )
        lines.append(f'{{rank=same; i{iteration}_strategy_researcher; i{iteration}_procurement_researcher;}}')
        lines.append(f'i{iteration}_research_manager -> i{iteration}_strategy_researcher;')
        lines.append(f'i{iteration}_research_manager -> i{iteration}_procurement_researcher;')
        lines.append(f'i{iteration}_strategy_researcher -> i{iteration}_synthesis_agent;')
        lines.append(f'i{iteration}_procurement_researcher -> i{iteration}_synthesis_agent;')
        lines.append(f'i{iteration}_synthesis_agent -> i{iteration}_independent_evaluator;')
        lines.append("}")
        if iteration < result.iterations_completed:
            route = view.route_by_iteration.get(iteration, "route not recorded")
            lines.append(
                f'i{iteration}_independent_evaluator -> i{iteration + 1}_research_manager '
                f'[label="{_dot_escape(route)}", style=dashed, color="#d97706"];'
            )
    final_iteration = result.iterations_completed
    # Current completed graph runs terminate as ``complete`` or
    # ``human_review``. Keep the third contract value explicit so a partial or
    # diagnostic result is never mislabeled as successfully complete.
    terminal_id = f"terminal_{result.status}"
    terminal_label = {
        "human_review": "Human Review",
        "complete": "Complete",
        "further_research": "Further Research",
    }[result.status]
    terminal_status = {
        "human_review": "needs_human_review",
        "complete": "completed",
        "further_research": "skipped",
    }[result.status]
    lines.append(
        f'{terminal_id} [label="{terminal_label}", fillcolor="{_status_color(terminal_status)}", '
        f'color="{_status_border(terminal_status)}"];'
    )
    route = view.route_by_iteration.get(final_iteration, result.status)
    lines.append(
        f'i{final_iteration}_independent_evaluator -> {terminal_id} '
        f'[label="{_dot_escape(route)}"];'
    )
    lines.append("}")
    return "\n".join(lines)


def _execution_status(detail: str | None, budget_exceeded: bool) -> str:
    """Classify only explicit trace/budget evidence."""
    if budget_exceeded:
        return "failed"
    if detail is None:
        return "not_recorded"
    lowered = detail.casefold()
    if "skipped" in lowered:
        return "skipped"
    if "reused" in lowered:
        return "reused"
    return "completed"


def _execution_label(stage: ExecutionStage) -> str:
    """Create a concise node label from recorded metrics only."""
    lines = [stage.label, stage.status.replace("_", " ").title()]
    if stage.evidence_count is not None:
        lines.append(f"Evidence {stage.evidence_count} | Findings {stage.findings_count or 0}")
    if stage.model_calls:
        lines.append(
            f"Calls {stage.model_calls} | Tokens {stage.input_tokens or 0}/{stage.output_tokens or 0}"
        )
    return "\n".join(lines)


def _status_color(status: str) -> str:
    return {
        "completed": "#dcfce7",
        "reused": "#e0f2fe",
        "skipped": "#fef3c7",
        "failed": "#fee2e2",
        "needs_human_review": "#f3e8ff",
        "not_recorded": "#f1f5f9",
    }.get(status, "#f1f5f9")


def _status_border(status: str) -> str:
    return {
        "completed": "#16a34a",
        "reused": "#0284c7",
        "skipped": "#d97706",
        "failed": "#dc2626",
        "needs_human_review": "#7c3aed",
        "not_recorded": "#64748b",
    }.get(status, "#64748b")


def _call_iteration(call_id: str) -> int | None:
    match = re.match(r"^(\d+):", call_id)
    return int(match.group(1)) if match else None


def _route_from_detail(detail: str) -> str | None:
    match = re.search(r"Selected route '([^']+)'", detail)
    return match.group(1) if match else None


def _number_from_detail(detail: str, pattern: str) -> int | None:
    match = re.search(pattern, detail)
    return int(match.group(1)) if match else None


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
