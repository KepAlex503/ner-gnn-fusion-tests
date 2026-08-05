from __future__ import annotations

import unittest
from pathlib import Path

from mention_graph.data import (
    corpus_statistics,
    documents_for_split,
    load_neruk,
    validate_document_splits,
)
from mention_graph.synthetic import generate_synthetic_corpus


class SyntheticDataTests(unittest.TestCase):
    def test_splits_are_document_disjoint(self) -> None:
        documents = generate_synthetic_corpus(
            train_documents=8,
            validation_documents=3,
            test_documents=4,
            seed=10,
        )
        validate_document_splits(documents)
        split_ids = {
            split: {
                document.doc_id
                for document in documents_for_split(documents, split)
            }
            for split in ("train", "validation", "test")
        }
        self.assertFalse(split_ids["train"] & split_ids["validation"])
        self.assertFalse(split_ids["train"] & split_ids["test"])
        self.assertFalse(split_ids["validation"] & split_ids["test"])

    def test_mention_offsets_round_trip(self) -> None:
        documents = generate_synthetic_corpus(
            train_documents=1,
            validation_documents=1,
            test_documents=1,
            seed=11,
        )
        for document in documents:
            for mention in document.mentions:
                self.assertEqual(
                    document.text[mention.start : mention.end],
                    mention.text,
                )


class NerUkAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path("data/external/ner-uk")
        if not (cls.root / "v2.0/data/dev-test-split.txt").exists():
            raise unittest.SkipTest("NER-UK 2.0 is not downloaded")

    def test_official_corpus_counts_and_nested_spans(self) -> None:
        documents = load_neruk(
            self.root,
            validation_fraction=0.15,
            split_seed=2026,
        )
        statistics = corpus_statistics(documents)
        self.assertEqual(statistics["documents"], 560)
        self.assertEqual(statistics["mentions"], 21_993)
        self.assertEqual(len(statistics["labels"]), 13)
        self.assertEqual(statistics["mentions_crossing_line_break"], 49)
        self.assertEqual(
            sum(
                statistics["splits"][split]["documents"]
                for split in ("train", "validation", "test")
            ),
            560,
        )


if __name__ == "__main__":
    unittest.main()

