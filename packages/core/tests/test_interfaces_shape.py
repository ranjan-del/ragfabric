from datetime import UTC, datetime

from ragfabric_core.auth.base import AuthProvider
from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.connectors.base import Connector, SourceDocument
from ragfabric_core.stores.base import Cache, GraphStore, LexicalStore, VectorStore
from ragfabric_core.strategies.base import RetrievedChunk


class MemVector:
    name = "memory"

    def upsert(self, chunk_ids, vectors, payloads): ...
    def query(self, vector, top_k, access, filters=None):
        return []

    def delete_document(self, document_id): ...
    def count(self):
        return 0

    def access_stats(self, filters, access):
        return (0, 0)


class MemLexical:
    name = "bm25"

    def index(self, chunk_ids, texts, payloads): ...
    def search(self, query, top_k, access, filters=None):
        return []

    def delete_document(self, document_id): ...


class MemGraph:
    name = "none"

    def upsert_entities(self, entities): ...
    def upsert_relationships(self, relationships): ...
    def neighbours(self, entity_ids, hops, access, max_nodes):
        return {"nodes": [], "edges": []}

    def delete_document(self, document_id): ...


class MemCache:
    def __init__(self):
        self.d = {}

    def get(self, key):
        return self.d.get(key)

    def set(self, key, value, ttl_seconds=None):
        self.d[key] = value

    def incr(self, key, ttl_seconds=None):
        self.d[key] = int(self.d.get(key, 0)) + 1
        return self.d[key]


class StaticAuth:
    name = "static"

    def authenticate(self, credential):
        return (
            Principal(user_id=1, email="a@b.c", role="admin", group_ids=[], api_key_id=None)
            if credential == "ok"
            else None
        )


class FolderConnector:
    name = "folder"

    def list_documents(self):
        yield SourceDocument(
            source_id="a.pdf",
            name="a.pdf",
            content_type="application/pdf",
            size_bytes=10,
            modified_at=datetime.now(UTC),
            metadata={},
        )

    def fetch(self, source_id):
        return b"%PDF"


def test_shapes_are_recognised_at_runtime():
    assert isinstance(MemVector(), VectorStore)
    assert isinstance(MemLexical(), LexicalStore)
    assert isinstance(MemGraph(), GraphStore)
    assert isinstance(MemCache(), Cache)
    assert isinstance(StaticAuth(), AuthProvider)
    assert isinstance(FolderConnector(), Connector)


def test_a_store_missing_a_method_is_rejected():
    class Broken:
        name = "x"

        def upsert(self, *a): ...

    assert not isinstance(Broken(), VectorStore)


def test_protocol_methods_run_with_access_filter():
    empty = MemVector().query([0.0], top_k=3, access=AccessFilter.unrestricted())
    assert empty == [] and MemCache().incr("k") == 1
    assert (
        StaticAuth().authenticate("ok").role == "admin" and StaticAuth().authenticate("no") is None
    )
    doc = next(FolderConnector().list_documents())
    assert isinstance(doc, SourceDocument) and FolderConnector().fetch(doc.source_id).startswith(
        b"%PDF"
    )
    RetrievedChunk(chunk_id=1, document_id=1, collection_id=None, text="t")
