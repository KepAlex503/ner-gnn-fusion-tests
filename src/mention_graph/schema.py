from __future__ import annotations

from dataclasses import dataclass, field


NER_UK_LABELS = (
    "ART",
    "DATE",
    "DOC",
    "JOB",
    "LOC",
    "MISC",
    "MON",
    "ORG",
    "PCT",
    "PERIOD",
    "PERS",
    "QUANT",
    "TIME",
)


@dataclass(slots=True)
class Mention:
    mention_id: str
    doc_id: str
    start: int
    end: int
    text: str
    label: str
    sentence_id: int
    sentence_start: int
    sentence_end: int


@dataclass(slots=True)
class Document:
    doc_id: str
    source: str
    text: str
    sentence_spans: list[tuple[int, int]]
    mentions: list[Mention] = field(default_factory=list)
    split: str = ""

    def sorted_mentions(self) -> list[Mention]:
        return sorted(
            self.mentions,
            key=lambda mention: (mention.start, mention.end, mention.mention_id),
        )

