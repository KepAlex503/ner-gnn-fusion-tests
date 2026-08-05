from __future__ import annotations

import random
from dataclasses import dataclass, field

from .data import sentence_spans
from .schema import Document, Mention


SYNTHETIC_LABELS = ("DOC", "LOC", "ORG", "PERS")
SYLLABLES = (
    "Бар",
    "Вер",
    "Гор",
    "Дан",
    "Жив",
    "Кал",
    "Лук",
    "Мир",
    "Над",
    "Ор",
    "Рад",
    "Слав",
    "Тер",
    "Фед",
    "Яр",
)


@dataclass
class _DocumentBuilder:
    doc_id: str
    source: str
    split: str
    parts: list[str] = field(default_factory=list)
    annotations: list[tuple[int, int, str]] = field(default_factory=list)

    def add_sentence(
        self,
        pieces: list[tuple[str, str | None]],
    ) -> None:
        if self.parts:
            self.parts.append("\n")
        for text, label in pieces:
            start = sum(len(part) for part in self.parts)
            self.parts.append(text)
            if label is not None:
                self.annotations.append((start, start + len(text), label))

    def build(self) -> Document:
        text = "".join(self.parts)
        spans = sentence_spans(text)
        mentions: list[Mention] = []
        for index, (start, end, label) in enumerate(self.annotations, 1):
            sentence_id = next(
                position
                for position, (left, right) in enumerate(spans)
                if left <= start < right
            )
            sentence_start, sentence_end = spans[sentence_id]
            mentions.append(
                Mention(
                    mention_id=f"{self.doc_id}:T{index}",
                    doc_id=self.doc_id,
                    start=start,
                    end=end,
                    text=text[start:end],
                    label=label,
                    sentence_id=sentence_id,
                    sentence_start=sentence_start,
                    sentence_end=max(sentence_end, end),
                )
            )
        return Document(
            doc_id=self.doc_id,
            source=self.source,
            text=text,
            sentence_spans=spans,
            mentions=mentions,
            split=self.split,
        )


def _unique_name(generator: random.Random, index: int) -> str:
    del index
    left, middle, right = generator.sample(SYLLABLES, 3)
    return f"{left}{middle}{right}{generator.randint(1000, 9999)}"


def generate_synthetic_corpus(
    *,
    train_documents: int = 80,
    validation_documents: int = 20,
    test_documents: int = 30,
    seed: int = 2026,
) -> list[Document]:
    """Generate document-level examples where repeated mentions carry context."""
    generator = random.Random(seed)
    split_counts = (
        ("train", train_documents),
        ("validation", validation_documents),
        ("test", test_documents),
    )
    documents: list[Document] = []
    global_index = 0
    for split, count in split_counts:
        for local_index in range(count):
            global_index += 1
            org = _unique_name(generator, global_index)
            person = _unique_name(generator, global_index + 10_000)
            location = _unique_name(generator, global_index + 20_000)
            document_name = f"{_unique_name(generator, global_index + 30_000)}-{100 + global_index}"
            builder = _DocumentBuilder(
                doc_id=f"syn-{split}-{local_index:03d}",
                source="synthetic-a" if local_index % 2 == 0 else "synthetic-b",
                split=split,
            )
            builder.add_sentence(
                [
                    ("Компанія ", None),
                    (org, "ORG"),
                    (" уклала угоду.", None),
                ]
            )
            builder.add_sentence(
                [
                    ("Посадовець ", None),
                    (person, "PERS"),
                    (" прокоментував подію.", None),
                ]
            )
            builder.add_sentence(
                [
                    ("Зустріч відбулася у місті ", None),
                    (location, "LOC"),
                    (".", None),
                ]
            )
            builder.add_sentence(
                [
                    ("Офіційний документ ", None),
                    (document_name, "DOC"),
                    (" набув чинності.", None),
                ]
            )
            # These contexts deliberately use the same template for every class.
            builder.add_sentence(
                [
                    ("У повідомленні згадали ", None),
                    (org, "ORG"),
                    (".", None),
                ]
            )
            builder.add_sentence(
                [
                    ("У повідомленні згадали ", None),
                    (person, "PERS"),
                    (".", None),
                ]
            )
            builder.add_sentence(
                [
                    ("У повідомленні згадали ", None),
                    (location, "LOC"),
                    (".", None),
                ]
            )
            if local_index % 3 == 0:
                builder.add_sentence(
                    [
                        ("Згодом ", None),
                        (org, "ORG"),
                        (" знову згадали.", None),
                    ]
                )
            documents.append(builder.build())
    return documents
