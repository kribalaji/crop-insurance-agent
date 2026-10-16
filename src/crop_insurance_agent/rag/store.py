"""Chroma index over policy PDFs.

Each PDF in the policies folder is one policy; its id is the file name without
`.pdf`. `sync()` re-indexes PDFs whose content changed and drops removed ones,
so callers can run it before every query.
"""

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

log = logging.getLogger(__name__)

Embedder = Callable[[list[str]], list[list[float]]]
COLLECTION = "policies"


class PolicyNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class Chunk:
    policy_id: str
    page: int
    text: str


def default_embedder() -> Embedder:
    """Chroma's bundled all-MiniLM-L6-v2 (ONNX, local; downloaded once on first use)."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    ef = DefaultEmbeddingFunction()
    return lambda texts: [[float(x) for x in v] for v in ef(texts)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pdf_chunks(path: Path, chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for page_no, page in enumerate(PdfReader(path).pages, start=1):
        text = page.extract_text() or ""
        chunks += [Chunk(path.stem, page_no, t) for t in splitter.split_text(text)]
    return chunks


class PolicyStore:
    def __init__(self, persist_dir: Path, embedder: Embedder | None = None) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(persist_dir), settings=chromadb.Settings(anonymized_telemetry=False)
        )
        # Embeddings are always passed explicitly, so Chroma's own function is disabled.
        self._col = client.get_or_create_collection(
            COLLECTION, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
        self._embed = embedder or default_embedder()

    def _indexed(self) -> dict[str, str]:
        """policy_id -> sha256 of the indexed PDF."""
        metas = self._col.get(include=["metadatas"])["metadatas"] or []
        return {str(m["policy_id"]): str(m["sha256"]) for m in metas}

    def sync(self, policies_dir: Path) -> list[str]:
        """Bring the index in line with the PDFs on disk; return ids (re)indexed."""
        on_disk = {p.stem: p for p in sorted(policies_dir.glob("*.pdf"))}
        indexed = self._indexed()

        for policy_id in indexed.keys() - on_disk.keys():
            self._col.delete(where={"policy_id": policy_id})

        changed = []
        for policy_id, path in on_disk.items():
            sha = _sha256(path)
            if indexed.get(policy_id) == sha:
                continue
            self._col.delete(where={"policy_id": policy_id})
            chunks = pdf_chunks(path)
            if not chunks:
                log.warning("no extractable text in %s (scanned PDF?); skipped", path)
                continue
            self._col.add(
                ids=[f"{policy_id}:p{c.page}:{i}" for i, c in enumerate(chunks)],
                documents=[c.text for c in chunks],
                embeddings=self._embed([c.text for c in chunks]),
                metadatas=[{"policy_id": policy_id, "page": c.page, "sha256": sha} for c in chunks],
            )
            changed.append(policy_id)
        return changed

    def policy_ids(self) -> list[str]:
        return sorted(self._indexed())

    def search(self, policy_id: str, query: str, k: int = 4) -> list[Chunk]:
        if policy_id not in self._indexed():
            raise PolicyNotFoundError(f"policy {policy_id!r} is not indexed")
        res = self._col.query(
            query_embeddings=self._embed([query]),
            n_results=k,
            where={"policy_id": policy_id},
            include=["documents", "metadatas"],
        )
        docs = (res["documents"] or [[]])[0]
        metas = (res["metadatas"] or [[]])[0]
        return [Chunk(policy_id, int(m["page"]), d) for d, m in zip(docs, metas, strict=True)]
