from __future__ import annotations

from datetime import date

import pytest

from public_sector_opportunity_research.embeddings import (
    BGE_QUERY_INSTRUCTION,
    SentenceTransformerEmbedder,
)
from public_sector_opportunity_research.indexing import VectorIndexer
from public_sector_opportunity_research.models import Chunk, DocumentMetadata
from public_sector_opportunity_research.retrieval import SemanticRetriever
from public_sector_opportunity_research.storage import (
    SQLiteStore,
    VectorIndexConfigurationError,
)


class FakeEmbedder:
    """Deterministic test double; production demonstrations use the real model."""

    def __init__(self, vectors, model_name="test-embedding-v1"):
        self.vectors = vectors
        self._model_name = model_name
        self.document_batches = []

    @property
    def model_name(self):
        return self._model_name

    @property
    def dimension(self):
        return 2

    @property
    def query_instruction(self):
        return "test query: "

    def embed_documents(self, texts):
        self.document_batches.append(list(texts))
        return [self.vectors[text] for text in texts]

    def embed_query(self, query):
        return self.vectors[query]


class FakeReranker:
    """Deterministic score provider for fast, network-free unit tests."""

    model_name = "test-reranker-v1"

    def __init__(self, scores=None):
        self.scores = scores or {}
        self.calls = []

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        return [self.scores.get(passage, 0.0) for passage in passages]


class _ArrayLike:
    """Minimal stand-in for a NumPy result returned by Sentence Transformers."""

    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class _RecordingSentenceModel:
    def __init__(self):
        self.inputs = []

    def get_embedding_dimension(self):
        return 2

    def encode(self, value, **kwargs):
        self.inputs.append(value)
        if isinstance(value, list):
            return _ArrayLike([[1.0, 0.0] for _ in value])
        return _ArrayLike([1.0, 0.0])


def _save_document(
    store,
    document_id,
    agency_id,
    document_type,
    publication_date,
    texts,
):
    metadata = DocumentMetadata(
        agency_id=agency_id,
        agency_name=f"{agency_id} name",
        title=f"{document_id} title",
        document_type=document_type,
        source_url=f"https://example.invalid/{document_id}",
        publication_date=publication_date,
        effective_date=date(2026, 7, 1),
        last_updated_date=date(2026, 8, 1),
        capture_date=date(2026, 10, 24),
    )
    chunks = [
        Chunk(
            chunk_id=f"chk_{document_id}_{index}",
            document_id=document_id,
            chunk_index=index,
            text=text,
            token_count=len(text.split()),
            section_heading=f"Section {index}",
            page_number=index + 1,
        )
        for index, text in enumerate(texts)
    ]
    store.save_document(
        document_id=document_id,
        metadata=metadata,
        file_hash=f"hash-{document_id}",
        local_path=None,
        chunks=chunks,
    )


def test_real_embedder_wrapper_applies_bge_instruction_only_to_queries():
    embedder = SentenceTransformerEmbedder()
    recording_model = _RecordingSentenceModel()
    embedder._model = recording_model

    assert embedder.dimension == 2
    assert embedder.embed_documents(["stored passage"]) == [[1.0, 0.0]]
    assert embedder.embed_query("research question") == [1.0, 0.0]
    assert recording_model.inputs == [
        ["stored passage"],
        f"{BGE_QUERY_INSTRUCTION}research question",
    ]


def test_vector_index_creation_persistence_and_incremental_embedding(store):
    vectors = {"first chunk": [1.0, 0.0], "second chunk": [0.0, 1.0]}
    embedder = FakeEmbedder(vectors)
    indexer = VectorIndexer(store, embedder)
    _save_document(store, "doc_one", "agency-a", "strategy", date(2026, 1, 1), ["first chunk"])

    first = indexer.sync()
    second = indexer.sync()

    assert first.embedded_count == 1
    assert first.existing_count == 1
    assert second.embedded_count == 0
    assert embedder.document_batches == [["first chunk"]]
    config = store.get_embedding_config()
    assert config["model_name"] == "test-embedding-v1"
    assert config["embedding_dimension"] == 2

    # Reopen the same file to prove vectors are persisted rather than held in memory.
    reopened = SQLiteStore(store.database_path)
    assert reopened.count_embeddings() == 1

    _save_document(store, "doc_two", "agency-a", "budget", date(2026, 2, 1), ["second chunk"])
    incremental = indexer.sync()
    assert incremental.embedded_count == 1
    assert incremental.existing_count == 2
    assert embedder.document_batches[-1] == ["second chunk"]


def test_model_change_requires_and_supports_rebuild(store):
    _save_document(store, "doc_one", "agency-a", "strategy", date(2026, 1, 1), ["first"])
    VectorIndexer(store, FakeEmbedder({"first": [1.0, 0.0]}, "model-a")).sync()
    changed = VectorIndexer(store, FakeEmbedder({"first": [0.0, 1.0]}, "model-b"))

    with pytest.raises(VectorIndexConfigurationError):
        changed.sync()

    rebuilt = changed.sync(rebuild=True)
    assert rebuilt.rebuilt is True
    assert rebuilt.embedded_count == 1
    assert store.get_embedding_config()["model_name"] == "model-b"


def test_metadata_filters_are_applied_before_vector_top_k(store):
    vectors = {"question": [1.0, 0.0]}
    wrong_agency = [f"wrong agency {index}" for index in range(25)]
    wrong_type = [f"wrong type {index}" for index in range(25)]
    future = [f"future {index}" for index in range(25)]
    eligible = [f"eligible {index}" for index in range(5)]
    for index, text in enumerate(wrong_agency + wrong_type + future):
        vectors[text] = [1.0, 0.0001 * (index + 1)]
    for index, text in enumerate(eligible):
        vectors[text] = [0.8, 0.6 + 0.001 * index]

    _save_document(
        store, "doc_wrong_agency", "other-agency", "strategy", date(2026, 1, 1), wrong_agency
    )
    _save_document(
        store, "doc_wrong_type", "target-agency", "budget", date(2026, 1, 1), wrong_type
    )
    _save_document(
        store, "doc_future", "target-agency", "strategy", date(2028, 1, 1), future
    )
    _save_document(
        store, "doc_right", "target-agency", "strategy", date(2026, 6, 1), eligible
    )
    retriever = SemanticRetriever(store, FakeEmbedder(vectors), FakeReranker())

    response = retriever.retrieve(
        "question",
        agency_id="target-agency",
        document_type="strategy",
        as_of_date="2026-12-31",
        top_k=20,
        final_k=10,
    )

    assert len(response.results) == 5
    assert {result.agency_id for result in response.results} == {"target-agency"}
    assert {result.document_id for result in response.results} == {"doc_right"}
    assert len(response.trace["initial_candidates"]) == 5


def test_reranking_reduces_twenty_candidates_to_ten_and_preserves_ranks(store):
    texts = [f"candidate {index}" for index in range(20)]
    vectors = {"question": [1.0, 0.0]}
    scores = {}
    for index, text in enumerate(texts):
        vectors[text] = [1.0, 0.01 * index]
        scores[text] = float(index)
    _save_document(store, "doc_many", "agency-a", "strategy", date(2026, 1, 1), texts)
    retriever = SemanticRetriever(store, FakeEmbedder(vectors), FakeReranker(scores))

    response = retriever.retrieve("question", top_k=20, final_k=10)

    assert len(response.trace["initial_candidates"]) == 20
    assert len(response.trace["reranked_candidates"]) == 20
    assert len(response.results) == 10
    assert response.results[0].text == "candidate 19"
    assert response.results[0].vector_rank == 20
    assert response.results[0].reranker_rank == 1
    assert [result.reranker_rank for result in response.results] == list(range(1, 11))


def test_empty_small_results_trace_and_vector_source_mapping(store):
    vectors = {
        "question": [1.0, 0.0],
        "best evidence": [1.0, 0.0],
        "second evidence": [0.0, 1.0],
    }
    _save_document(
        store,
        "doc_small",
        "agency-a",
        "program-plan",
        date(2026, 5, 10),
        ["best evidence", "second evidence"],
    )
    retriever = SemanticRetriever(
        store,
        FakeEmbedder(vectors),
        FakeReranker({"best evidence": 2.0, "second evidence": 1.0}),
    )

    small = retriever.retrieve("question", top_k=20, final_k=10)
    assert len(small.results) == 2
    assert small.results[0].chunk_id == "chk_doc_small_0"
    assert small.results[0].document_id == "doc_small"
    assert small.results[0].text == "best evidence"
    assert small.results[0].document_title == "doc_small title"
    assert small.results[0].publication_date == "2026-05-10"

    persisted = store.get_retrieval_trace(small.trace_id)
    assert persisted["original_research_question"] == "question"
    assert persisted["embedding_model"]["name"] == "test-embedding-v1"
    assert persisted["initial_candidates"]
    assert persisted["reranked_candidates"]
    assert persisted["final_selected_evidence"]
    assert persisted["source_document_ids"] == ["doc_small"]

    empty = retriever.retrieve("question", agency_id="missing-agency")
    assert empty.results == []
    assert empty.trace["initial_candidates"] == []
    assert empty.trace["final_selected_evidence"] == []
    assert store.get_retrieval_trace(empty.trace_id) is not None


def test_as_of_date_excludes_undated_sources_instead_of_using_capture_date(store):
    vectors = {"question": [1.0, 0.0], "undated": [1.0, 0.0]}
    _save_document(store, "doc_undated", "agency-a", "strategy", None, ["undated"])
    # Remove effective date too; capture date remains populated.
    with store.connect() as connection:
        connection.execute(
            "UPDATE documents SET effective_date = NULL WHERE document_id = 'doc_undated'"
        )
    retriever = SemanticRetriever(store, FakeEmbedder(vectors), FakeReranker())

    response = retriever.retrieve("question", as_of_date="2026-12-31")

    assert response.results == []
