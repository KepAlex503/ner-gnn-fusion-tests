from __future__ import annotations

import bisect
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from .graph import normalize_surface
from .schema import Document, Mention, NER_UK_LABELS


ANNOTATION_RE = re.compile(
    r"^(T\d+)\t([A-Z]+)\t(\d+)\t(\d+)\t",
    flags=re.MULTILINE,
)
NONEMPTY_LINE_RE = re.compile(r"[^\r\n]+")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Treat each non-empty source line as a sentence while preserving offsets."""
    spans: list[tuple[int, int]] = []
    for match in NONEMPTY_LINE_RE.finditer(text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    if not spans and text:
        spans.append((0, len(text)))
    return spans


def _mention_sentence(
    start: int,
    end: int,
    spans: list[tuple[int, int]],
) -> tuple[int, int, int]:
    starts = [span[0] for span in spans]
    sentence_id = max(0, bisect.bisect_right(starts, start) - 1)
    if sentence_id >= len(spans):
        sentence_id = len(spans) - 1
    sentence_start, sentence_end = spans[sentence_id]
    if not (sentence_start <= start <= sentence_end):
        for candidate, (left, right) in enumerate(spans):
            if left <= start < right or (start < left and end > left):
                sentence_id = candidate
                sentence_start, sentence_end = left, right
                break
    return sentence_id, sentence_start, max(sentence_end, end)


def parse_brat_document(
    text_path: Path,
    annotation_path: Path,
    *,
    split: str = "",
) -> Document:
    # Universal newline conversion is intentional: NER-UK offsets use LF semantics
    # even when Git checks files out with CRLF line endings on Windows.
    with text_path.open("r", encoding="utf-8", newline=None) as stream:
        text = stream.read()
    annotation_text = annotation_path.read_text(encoding="utf-8")
    spans = sentence_spans(text)
    source = text_path.parent.name
    doc_id = text_path.stem
    mentions: list[Mention] = []

    for match in ANNOTATION_RE.finditer(annotation_text):
        raw_id, label, raw_start, raw_end = match.groups()
        start, end = int(raw_start), int(raw_end)
        if label not in NER_UK_LABELS:
            raise ValueError(f"{annotation_path}: unknown label {label!r}")
        if not (0 <= start < end <= len(text)):
            raise ValueError(
                f"{annotation_path}: invalid span {start}:{end} for {len(text)} chars"
            )
        sentence_id, sentence_start, sentence_end = _mention_sentence(
            start, end, spans
        )
        mentions.append(
            Mention(
                mention_id=f"{doc_id}:{raw_id}",
                doc_id=doc_id,
                start=start,
                end=end,
                text=text[start:end],
                label=label,
                sentence_id=sentence_id,
                sentence_start=sentence_start,
                sentence_end=sentence_end,
            )
        )

    return Document(
        doc_id=doc_id,
        source=source,
        text=text,
        sentence_spans=spans,
        mentions=sorted(
            mentions,
            key=lambda mention: (mention.start, mention.end, mention.mention_id),
        ),
        split=split,
    )


def read_official_split(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    current = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in {"DEV", "TEST"}:
            current = line
            continue
        if not current:
            raise ValueError(f"{path}: document appears before a split header")
        if line in mapping:
            raise ValueError(f"{path}: duplicate document id {line}")
        mapping[line] = current
    return mapping


def _split_dev_documents(
    documents: list[Document],
    *,
    validation_fraction: float,
    seed: int,
) -> None:
    """Create a deterministic source- and label-aware validation subset."""
    by_source: dict[str, list[Document]] = {}
    for document in documents:
        if document.split == "DEV":
            by_source.setdefault(document.source, []).append(document)

    validation_ids: set[str] = set()
    for source, source_documents in sorted(by_source.items()):
        ordered = sorted(source_documents, key=lambda document: document.doc_id)
        random.Random(f"{seed}:{source}").shuffle(ordered)
        count = max(1, round(len(ordered) * validation_fraction))
        validation_ids.update(document.doc_id for document in ordered[:count])

    dev_documents = sorted(
        (document for document in documents if document.split == "DEV"),
        key=lambda document: document.doc_id,
    )
    labels = list(NER_UK_LABELS)
    label_to_index = {label: index for index, label in enumerate(labels)}
    document_vectors: dict[str, np.ndarray] = {}
    for document in dev_documents:
        vector = np.zeros(len(labels), dtype=np.float64)
        for mention in document.mentions:
            vector[label_to_index[mention.label]] += 1.0
        document_vectors[document.doc_id] = vector
    total_counts = sum(
        (document_vectors[document.doc_id] for document in dev_documents),
        start=np.zeros(len(labels), dtype=np.float64),
    )
    target_counts = total_counts * validation_fraction
    # A relative objective gives rare classes enough influence to avoid a
    # validation set with only one example of labels such as TIME.
    denominators = np.maximum(target_counts, 1.0)

    def objective(counts: np.ndarray) -> float:
        relative_error = ((counts - target_counts) / denominators) ** 2
        total_target = max(1.0, float(target_counts.sum()))
        size_error = ((float(counts.sum()) - total_target) / total_target) ** 2
        return float(relative_error.sum() + 0.25 * size_error)

    current_counts = sum(
        (
            document_vectors[document.doc_id]
            for document in dev_documents
            if document.doc_id in validation_ids
        ),
        start=np.zeros(len(labels), dtype=np.float64),
    )
    current_objective = objective(current_counts)
    # Pairwise same-source swaps preserve exact genre quotas while improving
    # the full label distribution. The procedure is deterministic.
    for _ in range(200):
        selected = [
            document
            for document in dev_documents
            if document.doc_id in validation_ids
        ]
        unselected = [
            document
            for document in dev_documents
            if document.doc_id not in validation_ids
        ]
        best: tuple[float, str, str, np.ndarray] | None = None
        for incoming in unselected:
            for outgoing in selected:
                if incoming.source != outgoing.source:
                    continue
                candidate_counts = (
                    current_counts
                    + document_vectors[incoming.doc_id]
                    - document_vectors[outgoing.doc_id]
                )
                candidate_objective = objective(candidate_counts)
                improvement = current_objective - candidate_objective
                candidate = (
                    improvement,
                    incoming.doc_id,
                    outgoing.doc_id,
                    candidate_counts,
                )
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
        if best is None or best[0] <= 1e-12:
            break
        _, incoming_id, outgoing_id, current_counts = best
        validation_ids.remove(outgoing_id)
        validation_ids.add(incoming_id)
        current_objective = objective(current_counts)

    for document in documents:
        if document.split == "TEST":
            document.split = "test"
        elif document.doc_id in validation_ids:
            document.split = "validation"
        else:
            document.split = "train"


def load_neruk(
    root: Path,
    *,
    validation_fraction: float = 0.15,
    split_seed: int = 2026,
) -> list[Document]:
    data_root = root / "v2.0" / "data"
    split_path = data_root / "dev-test-split.txt"
    if not split_path.exists():
        raise FileNotFoundError(
            f"NER-UK 2.0 not found at {root}. Run scripts/download_neruk.py."
        )
    official = read_official_split(split_path)
    annotation_paths = sorted(data_root.glob("*/*.ann"))
    documents: list[Document] = []
    for annotation_path in annotation_paths:
        doc_id = annotation_path.stem
        if doc_id not in official:
            raise ValueError(f"{doc_id} is missing from {split_path}")
        documents.append(
            parse_brat_document(
                annotation_path.with_suffix(".txt"),
                annotation_path,
                split=official[doc_id],
            )
        )
    if set(official) != {document.doc_id for document in documents}:
        missing = sorted(set(official) - {document.doc_id for document in documents})
        raise ValueError(f"split lists documents without annotations: {missing[:5]}")
    _split_dev_documents(
        documents,
        validation_fraction=validation_fraction,
        seed=split_seed,
    )
    validate_document_splits(documents)
    return sorted(documents, key=lambda document: (document.split, document.doc_id))


def validate_document_splits(documents: Iterable[Document]) -> None:
    seen: dict[str, str] = {}
    valid_splits = {"train", "validation", "test"}
    for document in documents:
        if document.split not in valid_splits:
            raise ValueError(
                f"{document.doc_id}: expected one of {valid_splits}, got {document.split!r}"
            )
        previous = seen.setdefault(document.doc_id, document.split)
        if previous != document.split:
            raise ValueError(
                f"document leakage: {document.doc_id} is in {previous} and {document.split}"
            )


def documents_for_split(
    documents: Iterable[Document],
    split: str,
) -> list[Document]:
    return sorted(
        (document for document in documents if document.split == split),
        key=lambda document: document.doc_id,
    )


def flatten_mentions(documents: Iterable[Document]) -> list[Mention]:
    mentions: list[Mention] = []
    for document in sorted(documents, key=lambda item: item.doc_id):
        mentions.extend(document.sorted_mentions())
    return mentions


def corpus_statistics(documents: Iterable[Document]) -> dict[str, object]:
    documents = list(documents)
    split_stats: dict[str, dict[str, object]] = {}
    label_counts: Counter[str] = Counter()
    repetition_counts: Counter[str] = Counter()
    crossing_newline = 0
    for split in ("train", "validation", "test"):
        selected = [document for document in documents if document.split == split]
        selected_mentions = flatten_mentions(selected)
        labels = Counter(mention.label for mention in selected_mentions)
        label_counts.update(labels)
        split_stats[split] = {
            "documents": len(selected),
            "sentences": sum(len(document.sentence_spans) for document in selected),
            "mentions": len(selected_mentions),
            "classes_present": len(labels),
            "labels": dict(sorted(labels.items())),
            "sources": dict(
                sorted(Counter(document.source for document in selected).items())
            ),
        }
        for document in selected:
            normalized = Counter(
                normalize_surface(mention.text) for mention in document.mentions
            )
            for mention in document.mentions:
                frequency = normalized[normalize_surface(mention.text)]
                bucket = "1" if frequency == 1 else "2" if frequency == 2 else "3+"
                repetition_counts[bucket] += 1
                crossing_newline += int("\n" in mention.text)
    return {
        "documents": len(documents),
        "sentences": sum(len(document.sentence_spans) for document in documents),
        "mentions": sum(len(document.mentions) for document in documents),
        "labels": dict(sorted(label_counts.items())),
        "repetition_mentions": dict(repetition_counts),
        "mentions_crossing_line_break": crossing_newline,
        "splits": split_stats,
    }


def _document_shingles(text: str, size: int = 5) -> set[str]:
    tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    if len(tokens) < size:
        return {" ".join(tokens)} if tokens else set()
    return {
        " ".join(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    }


def _simhash(shingles: set[str]) -> int:
    if not shingles:
        return 0
    accumulator = [0] * 64
    for shingle in shingles:
        value = int.from_bytes(
            hashlib.blake2b(
                shingle.encode("utf-8"),
                digest_size=8,
            ).digest(),
            byteorder="big",
        )
        for bit in range(64):
            accumulator[bit] += 1 if value & (1 << bit) else -1
    fingerprint = 0
    for bit, weight in enumerate(accumulator):
        if weight >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def duplicate_diagnostics(
    documents: Iterable[Document],
    *,
    near_similarity_threshold: float = 0.9,
    maximum_simhash_distance: int = 10,
) -> dict[str, object]:
    """Find exact and high-similarity documents without using labels."""
    documents = sorted(documents, key=lambda document: document.doc_id)
    exact_groups: dict[str, list[Document]] = {}
    shingles: dict[str, set[str]] = {}
    fingerprints: dict[str, int] = {}
    for document in documents:
        normalized = re.sub(r"\s+", " ", document.text.casefold()).strip()
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        exact_groups.setdefault(digest, []).append(document)
        document_shingles = _document_shingles(document.text)
        shingles[document.doc_id] = document_shingles
        fingerprints[document.doc_id] = _simhash(document_shingles)

    exact = [
        {
            "document_ids": [document.doc_id for document in group],
            "splits": [document.split for document in group],
        }
        for group in exact_groups.values()
        if len(group) > 1
    ]
    near_pairs: list[dict[str, object]] = []
    for left_index, left in enumerate(documents):
        left_shingles = shingles[left.doc_id]
        for right in documents[left_index + 1 :]:
            if abs(
                fingerprints[left.doc_id] ^ fingerprints[right.doc_id]
            ).bit_count() > maximum_simhash_distance:
                continue
            right_shingles = shingles[right.doc_id]
            union = left_shingles | right_shingles
            similarity = (
                len(left_shingles & right_shingles) / len(union) if union else 1.0
            )
            if similarity < near_similarity_threshold:
                continue
            near_pairs.append(
                {
                    "left_document_id": left.doc_id,
                    "right_document_id": right.doc_id,
                    "left_split": left.split,
                    "right_split": right.split,
                    "left_source": left.source,
                    "right_source": right.source,
                    "jaccard_5gram": similarity,
                    "cross_split": left.split != right.split,
                }
            )
    return {
        "method": (
            "Exact normalized SHA-256 plus 5-token-shingle Jaccard for "
            f"SimHash candidates (Hamming <= {maximum_simhash_distance})."
        ),
        "near_similarity_threshold": near_similarity_threshold,
        "exact_duplicate_groups": exact,
        "near_duplicate_pairs": near_pairs,
        "cross_split_exact_groups": [
            group for group in exact if len(set(group["splits"])) > 1
        ],
        "cross_split_near_pairs": [
            pair for pair in near_pairs if pair["cross_split"]
        ],
    }


def write_dataset_manifest(documents: Iterable[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(corpus_statistics(documents), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
