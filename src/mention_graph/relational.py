from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from typing import Iterable

import numpy as np

from .graph import (
    build_edge_sets,
    canonical_pair,
    normalize_surface,
)
from .schema import Document


RELATION_NAMES = ("sent", "repeat", "near", "alias")
RELATIONAL_CONFIGURATIONS = (
    "none",
    "repeat",
    "alias_repeat",
    "context",
    "untyped_all",
    "untyped_random",
    "typed_all",
    "type_shuffle",
    "alias_all",
    "random",
)
WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
ACRONYM_STOPWORDS = {
    "в",
    "від",
    "до",
    "для",
    "з",
    "за",
    "і",
    "із",
    "й",
    "на",
    "по",
    "при",
    "та",
    "у",
    "зі",
}


def _surface_words(text: str) -> list[str]:
    return WORD_RE.findall(normalize_surface(text))


def _token_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _is_acronym(raw_text: str, expanded_words: list[str]) -> bool:
    compact = "".join(WORD_RE.findall(raw_text))
    content_words = [
        word for word in expanded_words if word not in ACRONYM_STOPWORDS
    ]
    all_initials = "".join(word[0] for word in expanded_words)
    content_initials = "".join(word[0] for word in content_words)
    return (
        2 <= len(compact) <= 8
        and compact.isupper()
        and compact.isalpha()
        and 2 <= len(expanded_words) <= 10
        and compact.casefold()
        in {all_initials.casefold(), content_initials.casefold()}
    )


def _starts_with_uppercase_letter(text: str) -> bool:
    return any(
        character.isupper()
        for character in text
        if character.isalpha()
    ) and next(
        (
            character.isupper()
            for character in text
            if character.isalpha()
        ),
        False,
    )


def _numeric_like(text: str) -> bool:
    return any(character.isdigit() for character in text) or any(
        marker in text for marker in ("%", "₴", "$", "€", "£")
    )


def surfaces_are_aliases(
    left: str,
    right: str,
    *,
    token_similarity_threshold: float,
) -> bool:
    """Conservative label-free alias rule for Ukrainian inflection/acronyms."""
    normalized_left = normalize_surface(left)
    normalized_right = normalize_surface(right)
    if (
        not normalized_left
        or not normalized_right
        or normalized_left == normalized_right
    ):
        return False
    left_words = _surface_words(left)
    right_words = _surface_words(right)
    if _is_acronym(left, right_words) or _is_acronym(right, left_words):
        return True
    # Equal-length token sequences capture case inflection without introducing
    # label-dependent entity-linking heuristics.
    if (
        not left_words
        or not 1 <= len(left_words) <= 4
        or len(left_words) != len(right_words)
        or not _starts_with_uppercase_letter(left)
        or not _starts_with_uppercase_letter(right)
        or _numeric_like(left)
        or _numeric_like(right)
    ):
        return False
    for left_word, right_word in zip(left_words, right_words, strict=True):
        if left_word == right_word:
            continue
        if (
            min(len(left_word), len(right_word)) < 5
            or abs(len(left_word) - len(right_word)) > 3
            or left_word[0] != right_word[0]
            or _token_similarity(left_word, right_word)
            < token_similarity_threshold
        ):
            return False
    return True


def build_alias_edges(
    document: Document,
    *,
    token_similarity_threshold: float,
) -> set[tuple[int, int]]:
    mentions = document.sorted_mentions()
    by_surface: dict[str, list[int]] = {}
    for index, mention in enumerate(mentions):
        by_surface.setdefault(normalize_surface(mention.text), []).append(index)
    surfaces = sorted(by_surface)
    candidates: list[tuple[int, str, str, int, int, str | None]] = []
    acronym_expansions: dict[str, set[str]] = {}
    for left_surface, right_surface in combinations(surfaces, 2):
        left_indices = by_surface[left_surface]
        right_indices = by_surface[right_surface]
        left_text = mentions[left_indices[0]].text
        right_text = mentions[right_indices[0]].text
        if not surfaces_are_aliases(
            left_text,
            right_text,
            token_similarity_threshold=token_similarity_threshold,
        ):
            continue
        acronym_surface: str | None = None
        if _is_acronym(left_text, _surface_words(right_text)):
            acronym_surface = left_surface
            acronym_expansions.setdefault(left_surface, set()).add(right_surface)
        elif _is_acronym(right_text, _surface_words(left_text)):
            acronym_surface = right_surface
            acronym_expansions.setdefault(right_surface, set()).add(left_surface)
        possible: list[tuple[int, int, int]] = []
        for left in left_indices:
            for right in right_indices:
                left_mention = mentions[left]
                right_mention = mentions[right]
                if not (
                    left_mention.end <= right_mention.start
                    or right_mention.end <= left_mention.start
                ):
                    continue
                distance = min(
                    abs(left_mention.start - right_mention.end),
                    abs(right_mention.start - left_mention.end),
                )
                possible.append((distance, left, right))
        if possible:
            distance, left, right = min(possible)
            candidates.append(
                (
                    distance,
                    left_surface,
                    right_surface,
                    left,
                    right,
                    acronym_surface,
                )
            )

    edges: set[tuple[int, int]] = set()
    surface_degrees: Counter[str] = Counter()
    for (
        _,
        left_surface,
        right_surface,
        left,
        right,
        acronym_surface,
    ) in sorted(candidates):
        if (
            acronym_surface is not None
            and len(acronym_expansions.get(acronym_surface, set())) > 1
        ):
            continue
        if surface_degrees[left_surface] >= 3 or surface_degrees[right_surface] >= 3:
            continue
        edges.add(canonical_pair(left, right))
        surface_degrees[left_surface] += 1
        surface_degrees[right_surface] += 1
    return edges


def degree_preserving_edge_rewire(
    edges: set[tuple[int, int]],
    node_count: int,
    *,
    seed: int,
    swap_multiplier: int = 10,
) -> set[tuple[int, int]]:
    """Randomize endpoints while preserving every node's undirected degree."""
    if node_count < 4 or len(edges) < 2:
        return set(edges)
    generator = np.random.default_rng(seed)
    rewired = set(edges)
    edge_list = sorted(rewired)
    target_swaps = max(1, swap_multiplier * len(edge_list))
    maximum_attempts = max(100, target_swaps * 30)
    successful = 0
    attempts = 0
    while successful < target_swaps and attempts < maximum_attempts:
        attempts += 1
        first_index, second_index = generator.choice(
            len(edge_list),
            size=2,
            replace=False,
        )
        first = edge_list[int(first_index)]
        second = edge_list[int(second_index)]
        a, b = first
        c, d = second
        if len({a, b, c, d}) < 4:
            continue
        if generator.random() < 0.5:
            proposed = (canonical_pair(a, c), canonical_pair(b, d))
        else:
            proposed = (canonical_pair(a, d), canonical_pair(b, c))
        if proposed[0] == proposed[1]:
            continue
        outside = rewired - {first, second}
        if proposed[0] in outside or proposed[1] in outside:
            continue
        rewired.remove(first)
        rewired.remove(second)
        rewired.add(proposed[0])
        rewired.add(proposed[1])
        edge_list[int(first_index)] = proposed[0]
        edge_list[int(second_index)] = proposed[1]
        successful += 1
    return rewired


def _stable_seed(seed: int, document_id: str, relation: str) -> int:
    return zlib.crc32(
        f"{seed}:{document_id}:{relation}".encode("utf-8")
    ) & 0xFFFFFFFF


def _structural_features(
    document: Document,
    relation_edges: dict[str, set[tuple[int, int]]],
) -> np.ndarray:
    """Invariant node features computed from the full semantic graph."""
    mentions = document.sorted_mentions()
    count = len(mentions)
    surface_counts = Counter(normalize_surface(mention.text) for mention in mentions)
    degrees = {
        relation: np.zeros(count, dtype=np.float32)
        for relation in RELATION_NAMES
    }
    for relation, edges in relation_edges.items():
        for left, right in edges:
            degrees[relation][left] += 1.0
            degrees[relation][right] += 1.0
    sentence_denominator = max(1, len(document.sentence_spans) - 1)
    degree_denominator = math.log1p(max(1, count - 1))
    rows: list[list[float]] = []
    for index, mention in enumerate(mentions):
        normalized = normalize_surface(mention.text)
        rows.append(
            [
                min(1.0, len(mention.text) / 40.0),
                mention.sentence_id / sentence_denominator,
                min(
                    1.0,
                    math.log1p(surface_counts[normalized]) / math.log(4.0),
                ),
                float(any(character.isupper() for character in mention.text)),
                *[
                    math.log1p(float(degrees[relation][index]))
                    / degree_denominator
                    for relation in RELATION_NAMES
                ],
            ]
        )
    if not rows:
        return np.empty((0, 8), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


@dataclass(slots=True)
class RelationalGraphSource:
    mention_ids: list[str]
    document_ids: list[str]
    labels: np.ndarray
    structural_features: np.ndarray
    document_offsets: dict[str, tuple[int, int]]
    document_relation_edges: dict[
        str,
        dict[str, set[tuple[int, int]]],
    ]


@dataclass(slots=True)
class RelationalGraphBatch:
    mention_ids: list[str]
    document_ids: list[str]
    labels: np.ndarray
    structural_features: np.ndarray
    edge_indices: dict[str, np.ndarray]
    edge_counts: dict[str, int]
    isolated_nodes: int
    document_offsets: dict[str, tuple[int, int]]
    configuration: str


def build_relational_source(
    documents: list[Document],
    *,
    label_to_id: dict[str, int],
    near_threshold_tokens: int,
    alias_threshold: float,
) -> RelationalGraphSource:
    mention_ids: list[str] = []
    document_ids: list[str] = []
    labels: list[int] = []
    structural_rows: list[np.ndarray] = []
    document_offsets: dict[str, tuple[int, int]] = {}
    document_relation_edges: dict[
        str,
        dict[str, set[tuple[int, int]]],
    ] = {}
    offset = 0
    for document in sorted(documents, key=lambda item: item.doc_id):
        mentions = document.sorted_mentions()
        base_edges = build_edge_sets(
            document,
            near_threshold_tokens=near_threshold_tokens,
        )
        relation_edges = {
            "sent": set(base_edges["sent"]),
            "repeat": set(base_edges["repeat"]),
            "near": set(base_edges["near"]),
            "alias": build_alias_edges(
                document,
                token_similarity_threshold=alias_threshold,
            ),
        }
        document_relation_edges[document.doc_id] = relation_edges
        structural_rows.append(_structural_features(document, relation_edges))
        mention_ids.extend(mention.mention_id for mention in mentions)
        document_ids.extend(document.doc_id for _ in mentions)
        labels.extend(label_to_id[mention.label] for mention in mentions)
        document_offsets[document.doc_id] = (offset, offset + len(mentions))
        offset += len(mentions)
    return RelationalGraphSource(
        mention_ids=mention_ids,
        document_ids=document_ids,
        labels=np.asarray(labels, dtype=np.int64),
        structural_features=(
            np.concatenate(structural_rows, axis=0)
            if structural_rows
            else np.empty((0, 8), dtype=np.float32)
        ),
        document_offsets=document_offsets,
        document_relation_edges=document_relation_edges,
    )


def _selected_edges(
    semantic: dict[str, set[tuple[int, int]]],
    configuration: str,
    *,
    node_count: int,
    seed: int,
    document_id: str,
) -> dict[str, set[tuple[int, int]]]:
    selected = {relation: set() for relation in RELATION_NAMES}
    if configuration == "none":
        return selected
    if configuration == "repeat":
        selected["repeat"] = set(semantic["repeat"])
    elif configuration == "alias_repeat":
        selected["repeat"] = set(semantic["repeat"])
        selected["alias"] = set(semantic["alias"])
    elif configuration == "context":
        selected["sent"] = set(semantic["sent"])
        selected["near"] = set(semantic["near"])
    elif configuration == "untyped_all":
        selected["sent"] = set().union(
            semantic["sent"],
            semantic["repeat"],
            semantic["near"],
        )
    elif configuration == "untyped_random":
        union = set().union(
            semantic["sent"],
            semantic["repeat"],
            semantic["near"],
        )
        selected["sent"] = degree_preserving_edge_rewire(
            union,
            node_count,
            seed=_stable_seed(seed, document_id, "untyped_random"),
        )
    elif configuration == "typed_all":
        for relation in ("sent", "repeat", "near"):
            selected[relation] = set(semantic[relation])
    elif configuration == "type_shuffle":
        relations = ("sent", "repeat", "near")
        union = sorted(
            set().union(*(semantic[relation] for relation in relations))
        )
        signatures = [
            tuple(
                relation
                for relation in relations
                if pair in semantic[relation]
            )
            for pair in union
        ]
        generator = np.random.default_rng(
            _stable_seed(seed, document_id, "type_shuffle")
        )
        generator.shuffle(signatures)
        for pair, signature in zip(union, signatures, strict=True):
            for relation in signature:
                selected[relation].add(pair)
    elif configuration == "alias_all":
        for relation in RELATION_NAMES:
            selected[relation] = set(semantic[relation])
    elif configuration == "random":
        for relation in ("sent", "repeat", "near"):
            selected[relation] = degree_preserving_edge_rewire(
                semantic[relation],
                node_count,
                seed=_stable_seed(seed, document_id, relation),
            )
    else:
        raise ValueError(f"unknown relational configuration {configuration!r}")
    return selected


def materialize_relational_batch(
    source: RelationalGraphSource,
    *,
    configuration: str,
    seed: int,
) -> RelationalGraphBatch:
    if configuration not in RELATIONAL_CONFIGURATIONS:
        raise ValueError(f"unknown relational configuration {configuration!r}")
    global_edges = {relation: set() for relation in RELATION_NAMES}
    edge_counts: Counter[str] = Counter()
    isolated_nodes = 0
    for document_id, (offset, end) in source.document_offsets.items():
        node_count = end - offset
        selected = _selected_edges(
            source.document_relation_edges[document_id],
            configuration,
            node_count=node_count,
            seed=seed,
            document_id=document_id,
        )
        degrees = np.zeros(node_count, dtype=np.int64)
        for relation, edges in selected.items():
            edge_counts[relation] += len(edges)
            for left, right in edges:
                global_edges[relation].add((offset + left, offset + right))
                degrees[left] += 1
                degrees[right] += 1
        isolated_nodes += int((degrees == 0).sum())

    edge_indices: dict[str, np.ndarray] = {}
    for relation, edges in global_edges.items():
        directed: list[tuple[int, int]] = []
        for left, right in sorted(edges):
            directed.append((left, right))
            directed.append((right, left))
        edge_indices[relation] = (
            np.asarray(directed, dtype=np.int64).T
            if directed
            else np.empty((2, 0), dtype=np.int64)
        )
    return RelationalGraphBatch(
        mention_ids=list(source.mention_ids),
        document_ids=list(source.document_ids),
        labels=source.labels.copy(),
        structural_features=source.structural_features.copy(),
        edge_indices=edge_indices,
        edge_counts={
            relation: int(edge_counts[relation])
            for relation in RELATION_NAMES
        },
        isolated_nodes=isolated_nodes,
        document_offsets=dict(source.document_offsets),
        configuration=configuration,
    )


def empty_edge_indices() -> dict[str, np.ndarray]:
    return {
        relation: np.empty((2, 0), dtype=np.int64)
        for relation in RELATION_NAMES
    }


def relation_signal_statistics(
    source: RelationalGraphSource,
    *,
    split: str,
) -> list[dict[str, object]]:
    labels = source.labels
    rows: list[dict[str, object]] = []
    relation_sets: dict[str, set[tuple[int, int]]] = {
        relation: set() for relation in RELATION_NAMES
    }
    null_weighted_sums: Counter[str] = Counter()
    edge_totals: Counter[str] = Counter()
    for document_id, semantic in source.document_relation_edges.items():
        offset, end = source.document_offsets[document_id]
        local_labels = labels[offset:end]
        counts = np.bincount(local_labels)
        denominator = len(local_labels) * max(0, len(local_labels) - 1)
        document_null = (
            float(np.sum(counts * (counts - 1)) / denominator)
            if denominator
            else 0.0
        )
        for relation, edges in semantic.items():
            relation_sets[relation].update(
                (offset + left, offset + right) for left, right in edges
            )
            null_weighted_sums[relation] += document_null * len(edges)
            edge_totals[relation] += len(edges)
        semantic_union = set().union(
            *(semantic[relation] for relation in ("sent", "repeat", "near"))
        )
        union_with_alias = semantic_union | semantic["alias"]
        null_weighted_sums["semantic_union"] += (
            document_null * len(semantic_union)
        )
        edge_totals["semantic_union"] += len(semantic_union)
        null_weighted_sums["union_with_alias"] += (
            document_null * len(union_with_alias)
        )
        edge_totals["union_with_alias"] += len(union_with_alias)
    relation_sets["semantic_union"] = set().union(
        relation_sets["sent"],
        relation_sets["repeat"],
        relation_sets["near"],
    )
    relation_sets["union_with_alias"] = set().union(
        *(relation_sets[relation] for relation in RELATION_NAMES)
    )
    for relation, edges in relation_sets.items():
        nodes = {node for edge in edges for node in edge}
        same = sum(int(labels[left] == labels[right]) for left, right in edges)
        agreement = same / len(edges) if edges else float("nan")
        null_expectation = (
            null_weighted_sums[relation] / edge_totals[relation]
            if edge_totals[relation]
            else float("nan")
        )
        if edges:
            z = 1.959963984540054
            denominator = 1.0 + z * z / len(edges)
            center = (
                agreement + z * z / (2.0 * len(edges))
            ) / denominator
            half_width = (
                z
                * math.sqrt(
                    agreement * (1.0 - agreement) / len(edges)
                    + z * z / (4.0 * len(edges) ** 2)
                )
                / denominator
            )
            wilson_lower = center - half_width
            wilson_upper = center + half_width
        else:
            wilson_lower = float("nan")
            wilson_upper = float("nan")
        rows.append(
            {
                "split": split,
                "relation": relation,
                "undirected_edges": len(edges),
                "covered_mentions": len(nodes),
                "coverage": len(nodes) / len(labels) if len(labels) else 0.0,
                "same_label_edges": same,
                "same_label_rate": agreement,
                "same_label_wilson_95_lower": wilson_lower,
                "same_label_wilson_95_upper": wilson_upper,
                "within_document_permutation_null": null_expectation,
                "same_label_lift": agreement - null_expectation,
            }
        )
    return rows


def relation_overlap_statistics(
    source: RelationalGraphSource,
    *,
    split: str,
) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str]] = Counter()
    for semantic in source.document_relation_edges.values():
        for left_index, left_relation in enumerate(RELATION_NAMES):
            for right_relation in RELATION_NAMES[left_index + 1 :]:
                counts[(left_relation, right_relation)] += len(
                    semantic[left_relation] & semantic[right_relation]
                )
    return [
        {
            "split": split,
            "relation_a": left,
            "relation_b": right,
            "overlap_edges": int(count),
        }
        for (left, right), count in sorted(counts.items())
    ]


def alias_threshold_diagnostics(
    documents: Iterable[Document],
    thresholds: Iterable[float],
) -> list[dict[str, object]]:
    documents = sorted(documents, key=lambda item: item.doc_id)
    mention_count = sum(len(document.mentions) for document in documents)
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        edge_count = 0
        same_label = 0
        covered: set[tuple[str, int]] = set()
        for document in documents:
            mentions = document.sorted_mentions()
            edges = build_alias_edges(
                document,
                token_similarity_threshold=float(threshold),
            )
            edge_count += len(edges)
            for left, right in edges:
                same_label += int(
                    mentions[left].label == mentions[right].label
                )
                covered.add((document.doc_id, left))
                covered.add((document.doc_id, right))
        agreement = same_label / edge_count if edge_count else float("nan")
        if edge_count:
            z = 1.959963984540054
            denominator = 1.0 + z * z / edge_count
            center = (
                agreement + z * z / (2.0 * edge_count)
            ) / denominator
            half_width = (
                z
                * math.sqrt(
                    agreement * (1.0 - agreement) / edge_count
                    + z * z / (4.0 * edge_count**2)
                )
                / denominator
            )
        else:
            center = float("nan")
            half_width = float("nan")
        rows.append(
            {
                "threshold": float(threshold),
                "undirected_edges": edge_count,
                "covered_mentions": len(covered),
                "coverage": (
                    len(covered) / mention_count if mention_count else 0.0
                ),
                "same_label_rate": agreement,
                "same_label_wilson_95_lower": center - half_width,
                "same_label_wilson_95_upper": center + half_width,
            }
        )
    return rows


def select_alias_threshold(
    diagnostics: list[dict[str, object]],
    *,
    minimum_same_label_rate: float,
) -> float:
    eligible = [
        row
        for row in diagnostics
        if float(row["same_label_rate"]) >= minimum_same_label_rate
    ]
    if not eligible:
        raise ValueError(
            "no alias threshold meets the validation agreement requirement"
        )
    best = max(
        eligible,
        key=lambda row: (
            float(row["coverage"]),
            float(row["same_label_rate"]),
            float(row["threshold"]),
        ),
    )
    return float(best["threshold"])
