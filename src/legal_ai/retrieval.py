import uuid
from dataclasses import dataclass
from pathlib import Path
import re

import chromadb
from openai import OpenAI

from legal_ai.chunking import chunk_document
from legal_ai.config import OPENAI_EMBEDDING_MODEL, UPLOAD_DB_DIR
from legal_ai.parsing import parse_document


@dataclass
class UploadedDocument:
    collection_name: str
    source_document: str
    num_chunks: int
    raw_text_preview: str


class ContractRetriever:
    def __init__(self, openai_client: OpenAI, db_path: Path = UPLOAD_DB_DIR):
        db_path.mkdir(parents=True, exist_ok=True)
        self.openai_client = openai_client
        self.chroma_client = chromadb.PersistentClient(path=str(db_path))

    def get_embedding(self, text: str) -> list[float]:
        response = self.openai_client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    def index_upload(self, file_path: str | Path) -> UploadedDocument:
        file_path = Path(file_path)
        source_document = file_path.name
        raw_text = parse_document(file_path)
        chunks = chunk_document(raw_text)

        if not chunks:
            raise ValueError("No chunks were created from the uploaded document.")

        collection_name = f"uploaded_doc_{uuid.uuid4().hex[:10]}"
        collection = self.chroma_client.create_collection(name=collection_name)

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for chunk in chunks:
            chunk_id = f"{source_document}::chunk_{chunk['chunk_index']}"
            clause_tags = chunk["clause_tags"]
            ids.append(chunk_id)
            documents.append(chunk["chunk_text"])
            metadatas.append(
                {
                    "source_document": source_document,
                    "chunk_index": int(chunk["chunk_index"]),
                    "token_count": int(chunk["token_count"]),
                    "clause_tags_str": " | ".join(clause_tags),
                }
            )
            embeddings.append(self.get_embedding(chunk["chunk_text"]))

        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

        return UploadedDocument(
            collection_name=collection_name,
            source_document=source_document,
            num_chunks=len(chunks),
            raw_text_preview=raw_text[:1200],
        )

    def retrieve(self, query: str, collection_name: str, top_k: int = 5, initial_k: int = 15) -> list[dict]:
        collection = self.chroma_client.get_collection(collection_name)
        results = collection.query(
            query_embeddings=[self.get_embedding(query)],
            n_results=initial_k,
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0] if "distances" in results else []

        rows = []
        for index, document in enumerate(docs):
            rows.append(
                {
                    "id": ids[index],
                    "document": document,
                    "metadata": metas[index],
                    "distance": distances[index] if distances else None,
                }
            )
        return rows[:top_k]

    def retrieve_keyword(
        self,
        collection_name: str,
        keywords: list[str],
        top_k: int = 4,
        require_money: bool = False,
    ) -> list[dict]:
        collection = self.chroma_client.get_collection(collection_name)
        results = collection.get(include=["documents", "metadatas"])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        ids = results.get("ids", [])

        scored_rows = []
        for index, document in enumerate(docs):
            lowered = document.lower()
            score = sum(lowered.count(keyword.lower()) for keyword in keywords)
            has_money = bool(re.search(r"\$\s?\d[\d,]*(?:\.\d{2})?", document))
            if require_money and has_money:
                score += 8
            if score <= 0:
                continue
            scored_rows.append(
                (
                    score,
                    {
                        "id": ids[index],
                        "document": document,
                        "metadata": metas[index],
                        "distance": None,
                    },
                )
            )

        scored_rows.sort(key=lambda item: item[0], reverse=True)
        return [row for _score, row in scored_rows[:top_k]]


def build_context(rows: list[dict]) -> str:
    if not rows:
        return "NO_RELEVANT_CONTEXT_FOUND"

    parts = []
    for index, row in enumerate(rows, start=1):
        meta = row["metadata"]
        parts.append(
            f"[Chunk {index}]\n"
            f"Chunk ID: {row['id']}\n"
            f"Source Document: {meta.get('source_document')}\n"
            f"Chunk Index: {meta.get('chunk_index')}\n"
            f"Clause Tags: {meta.get('clause_tags_str')}\n"
            f"Distance: {row.get('distance')}\n"
            f"Text:\n{row['document']}\n"
        )
    return "\n\n".join(parts)
