from __future__ import annotations

import numpy as np
import pytest

from mention_graph.reviewer2_morphology import (
    build_lemma_edge_index,
    create_ukrainian_analyzer,
    lemma_normalize_surface,
)
from mention_graph.schema import Document, Mention
from mention_graph.relational import build_relational_source


def _mention(
    mention_id: str,
    doc_id: str,
    start: int,
    text: str,
    label: str = "PERS",
) -> Mention:
    return Mention(
        mention_id=mention_id,
        doc_id=doc_id,
        start=start,
        end=start + len(text),
        text=text,
        label=label,
        sentence_id=0,
        sentence_start=0,
        sentence_end=100,
    )


def test_ukrainian_inflection_normalizes_to_same_lemma() -> None:
    pytest.importorskip("pymorphy3")
    analyzer = create_ukrainian_analyzer()
    assert lemma_normalize_surface("Катерини Кобченко", analyzer) == (
        lemma_normalize_surface("Катерина Кобченко", analyzer)
    )
    assert lemma_normalize_surface("«України»", analyzer) == "україна"


def test_lemma_edges_are_intra_document_and_label_free() -> None:
    pytest.importorskip("pymorphy3")
    analyzer = create_ukrainian_analyzer()
    documents = [
        Document(
            doc_id="a",
            source="test",
            text="Катерина Катерини",
            sentence_spans=[(0, 18)],
            mentions=[
                _mention("a1", "a", 0, "Катерина", "PERS"),
                _mention("a2", "a", 9, "Катерини", "ORG"),
            ],
            split="test",
        ),
        Document(
            doc_id="b",
            source="test",
            text="Катерини",
            sentence_spans=[(0, 8)],
            mentions=[_mention("b1", "b", 0, "Катерини", "LOC")],
            split="test",
        ),
    ]
    source = build_relational_source(
        documents,
        label_to_id={"PERS": 0, "ORG": 1, "LOC": 2},
        near_threshold_tokens=50,
        alias_threshold=0.9,
    )
    edge_index, _ = build_lemma_edge_index(documents, source, analyzer)
    assert np.array_equal(edge_index, np.asarray([[0, 1], [1, 0]]))
