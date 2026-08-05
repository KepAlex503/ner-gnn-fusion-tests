from __future__ import annotations

import math
import re
import string
import unicodedata
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .schema import Document, Mention


TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
QUOTE_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "«": '"',
        "»": '"',
        "’": "'",
        "‘": "'",
        "`": "'",
        "–": "-",
        "—": "-",
        "‑": "-",
    }
)


def normalize_surface(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).translate(QUOTE_TRANSLATION)
    normalized = re.sub(r"\s+", " ", normalized.casefold()).strip()
    while normalized and (
        normalized[0] in string.punctuation
        or unicodedata.category(normalized[0]).startswith("P")
    ):
        normalized = normalized[1:].lstrip()
    while normalized and (
        normalized[-1] in string.punctuation
        or unicodedata.category(normalized[-1]).startswith("P")
    ):
        normalized = normalized[:-1].rstrip()
    return normalized


def canonical_pair(left: int, right: int) -> tuple[int, int]:
    if left == right:
        raise ValueError("self-loops are added by the model, not the graph builder")
    return (left, right) if left < right else (right, left)


def _group_pairs(groups: Iterable[list[int]]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for group in groups:
        for position, left in enumerate(group):
            for right in group[position + 1 :]:
                if left != right:
                    edges.add(canonical_pair(left, right))
    return edges


def _mention_token_positions(
    document: Document,
    mentions: list[Mention],
) -> list[float]:
    token_spans = [match.span() for match in TOKEN_RE.finditer(document.text)]
    if not token_spans:
        return [0.0] * len(mentions)
    token_starts = [span[0] for span in token_spans]
    positions: list[float] = []
    for mention in mentions:
        left = max(0, np.searchsorted(token_starts, mention.start, side="right") - 1)
        right = max(left, np.searchsorted(token_starts, mention.end, side="left") - 1)
        positions.append((float(left) + float(right)) / 2.0)
    return positions


def build_edge_sets(
    document: Document,
    *,
    near_threshold_tokens: int,
) -> dict[str, set[tuple[int, int]]]:
    mentions = document.sorted_mentions()
    by_sentence: dict[int, list[int]] = defaultdict(list)
    by_surface: dict[str, list[int]] = defaultdict(list)
    for index, mention in enumerate(mentions):
        by_sentence[mention.sentence_id].append(index)
        by_surface[normalize_surface(mention.text)].append(index)

    sentence_edges = _group_pairs(by_sentence.values())
    repeat_edges = _group_pairs(by_surface.values())
    positions = _mention_token_positions(document, mentions)
    near_edges: set[tuple[int, int]] = set()
    for left in range(len(mentions)):
        for right in range(left + 1, len(mentions)):
            if abs(positions[left] - positions[right]) <= near_threshold_tokens:
                near_edges.add((left, right))
    return {
        "sent": sentence_edges,
        "repeat": repeat_edges,
        "near": near_edges,
    }


def randomize_edges_by_node_permutation(
    edges: set[tuple[int, int]],
    node_count: int,
    *,
    seed: int,
) -> set[tuple[int, int]]:
    """Randomize edge-to-mention assignment and exactly preserve graph topology."""
    if node_count < 2 or not edges:
        return set()
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(node_count)
    return {
        canonical_pair(int(permutation[left]), int(permutation[right]))
        for left, right in edges
    }


def _stable_document_seed(seed: int, document_id: str) -> int:
    return zlib.crc32(f"{seed}:{document_id}".encode("utf-8")) & 0xFFFFFFFF


def select_edges(
    edge_sets: dict[str, set[tuple[int, int]]],
    configuration: str,
    *,
    node_count: int,
    seed: int,
) -> tuple[set[tuple[int, int]], dict[str, set[tuple[int, int]]]]:
    enabled: dict[str, set[tuple[int, int]]] = {
        "sent": set(),
        "repeat": set(),
        "near": set(),
        "random": set(),
    }
    if configuration == "none":
        selected: set[tuple[int, int]] = set()
    elif configuration in {"sent", "repeat", "near"}:
        enabled[configuration] = set(edge_sets[configuration])
        selected = set(enabled[configuration])
    elif configuration == "all":
        for edge_type in ("sent", "repeat", "near"):
            enabled[edge_type] = set(edge_sets[edge_type])
        selected = set().union(*(enabled[key] for key in ("sent", "repeat", "near")))
    elif configuration == "random":
        meaningful = set().union(
            edge_sets["sent"],
            edge_sets["repeat"],
            edge_sets["near"],
        )
        enabled["random"] = randomize_edges_by_node_permutation(
            meaningful,
            node_count,
            seed=seed,
        )
        selected = set(enabled["random"])
    else:
        raise ValueError(f"unknown edge configuration {configuration!r}")
    return selected, enabled


@dataclass(slots=True)
class GraphBatch:
    mention_ids: list[str]
    document_ids: list[str]
    labels: np.ndarray
    structural_features: np.ndarray
    edge_index: np.ndarray
    edge_counts: dict[str, int]
    isolated_nodes: int
    document_offsets: dict[str, tuple[int, int]]


def _document_structural_features(
    document: Document,
    enabled_edges: dict[str, set[tuple[int, int]]],
) -> np.ndarray:
    mentions = document.sorted_mentions()
    count = len(mentions)
    surface_counts = Counter(normalize_surface(mention.text) for mention in mentions)
    degrees = {
        edge_type: np.zeros(count, dtype=np.float32)
        for edge_type in ("sent", "repeat", "near", "random")
    }
    for edge_type, edges in enabled_edges.items():
        for left, right in edges:
            degrees[edge_type][left] += 1.0
            degrees[edge_type][right] += 1.0

    rows: list[list[float]] = []
    sentence_denominator = max(1, len(document.sentence_spans) - 1)
    degree_denominator = math.log1p(max(1, count - 1))
    for index, mention in enumerate(mentions):
        normalized = normalize_surface(mention.text)
        rows.append(
            [
                min(1.0, len(mention.text) / 40.0),
                mention.sentence_id / sentence_denominator,
                min(1.0, math.log1p(surface_counts[normalized]) / math.log(4.0)),
                float(any(character.isupper() for character in mention.text)),
                *[
                    math.log1p(float(degrees[edge_type][index]))
                    / degree_denominator
                    for edge_type in ("sent", "repeat", "near", "random")
                ],
            ]
        )
    if not rows:
        return np.empty((0, 8), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def build_graph_batch(
    documents: list[Document],
    *,
    label_to_id: dict[str, int],
    configuration: str,
    near_threshold_tokens: int,
    seed: int,
) -> GraphBatch:
    mention_ids: list[str] = []
    document_ids: list[str] = []
    labels: list[int] = []
    structural_rows: list[np.ndarray] = []
    undirected_global_edges: set[tuple[int, int]] = set()
    edge_counts: Counter[str] = Counter()
    document_offsets: dict[str, tuple[int, int]] = {}
    isolated_nodes = 0
    offset = 0

    for document in sorted(documents, key=lambda item: item.doc_id):
        mentions = document.sorted_mentions()
        edge_sets = build_edge_sets(
            document,
            near_threshold_tokens=near_threshold_tokens,
        )
        selected, enabled = select_edges(
            edge_sets,
            configuration,
            node_count=len(mentions),
            seed=_stable_document_seed(seed, document.doc_id),
        )
        for edge_type, typed_edges in enabled.items():
            edge_counts[edge_type] += len(typed_edges)
        degrees = np.zeros(len(mentions), dtype=np.int64)
        for left, right in selected:
            undirected_global_edges.add((offset + left, offset + right))
            degrees[left] += 1
            degrees[right] += 1
        isolated_nodes += int((degrees == 0).sum())
        structural_rows.append(_document_structural_features(document, enabled))
        mention_ids.extend(mention.mention_id for mention in mentions)
        document_ids.extend(document.doc_id for _ in mentions)
        labels.extend(label_to_id[mention.label] for mention in mentions)
        document_offsets[document.doc_id] = (offset, offset + len(mentions))
        offset += len(mentions)

    directed_edges: list[tuple[int, int]] = []
    for left, right in sorted(undirected_global_edges):
        directed_edges.append((left, right))
        directed_edges.append((right, left))
    edge_index = (
        np.asarray(directed_edges, dtype=np.int64).T
        if directed_edges
        else np.empty((2, 0), dtype=np.int64)
    )
    structural_features = (
        np.concatenate(structural_rows, axis=0)
        if structural_rows
        else np.empty((0, 8), dtype=np.float32)
    )
    return GraphBatch(
        mention_ids=mention_ids,
        document_ids=document_ids,
        labels=np.asarray(labels, dtype=np.int64),
        structural_features=structural_features,
        edge_index=edge_index,
        edge_counts=dict(edge_counts),
        isolated_nodes=isolated_nodes,
        document_offsets=document_offsets,
    )
