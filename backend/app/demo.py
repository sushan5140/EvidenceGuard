from __future__ import annotations

from app.schemas import DocumentCreate
from app.services.store import DocumentStore


DEMO_DOCUMENTS = [
    DocumentCreate(
        title="Paris Exposition historical note",
        text=(
            "The Eiffel Tower was constructed for the 1889 Exposition Universelle in Paris. "
            "It was completed in 1889 and became one of the best known landmarks in France. "
            "The tower was initially criticized by a number of artists and writers."
        ),
        source_reliability=0.92,
        tags=["demo", "history"],
    ),
    DocumentCreate(
        title="Eiffel Tower visitor guide",
        text=(
            "The Eiffel Tower opened to the public in 1889. It stands in Paris, France, "
            "near the Champ de Mars. The structure was designed by engineers working "
            "for Gustave Eiffel's company."
        ),
        source_reliability=0.88,
        tags=["demo", "travel"],
    ),
    DocumentCreate(
        title="Unverified Eiffel Tower blog",
        text=(
            "A frequently repeated online claim says the Eiffel Tower opened in 1905. "
            "The same article says the tower is located in Paris. The article gives no "
            "primary historical citation for the 1905 date."
        ),
        source_reliability=0.35,
        tags=["demo", "unverified"],
    ),
]


def load_demo(store: DocumentStore) -> int:
    existing = {doc.title for doc in store.list()}
    count = 0
    for document in DEMO_DOCUMENTS:
        if document.title not in existing:
            store.add(document)
            count += 1
    return count
