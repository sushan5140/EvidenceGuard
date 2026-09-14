from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from app.schemas import DocumentCreate, DocumentRecord


class DocumentStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = RLock()
        self._documents: dict[str, DocumentRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for item in raw:
                record = DocumentRecord.model_validate(item)
                self._documents[record.id] = record
        except (json.JSONDecodeError, OSError, ValueError):
            # A broken local cache must not prevent the API from starting.
            self._documents = {}

    def _persist(self) -> None:
        payload = [
            record.model_dump(mode="json")
            for record in sorted(self._documents.values(), key=lambda d: d.created_at)
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add(self, document: DocumentCreate, *, injected: bool = False) -> DocumentRecord:
        with self._lock:
            record = DocumentRecord(
                id=uuid4().hex[:12],
                created_at=datetime.now(timezone.utc),
                injected=injected,
                **document.model_dump(),
            )
            self._documents[record.id] = record
            self._persist()
            return record

    def list(self) -> list[DocumentRecord]:
        with self._lock:
            return list(self._documents.values())

    def delete(self, document_id: str) -> bool:
        with self._lock:
            if document_id not in self._documents:
                return False
            del self._documents[document_id]
            self._persist()
            return True

    def clear_injected(self) -> int:
        with self._lock:
            targets = [key for key, doc in self._documents.items() if doc.injected]
            for key in targets:
                del self._documents[key]
            if targets:
                self._persist()
            return len(targets)

    def count(self) -> int:
        return len(self._documents)
