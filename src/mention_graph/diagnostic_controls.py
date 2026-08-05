from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .graph import canonical_pair, normalize_surface
from .relational import RELATION_NAMES, degree_preserving_edge_rewire
from .schema import Document, Mention


DEFAULT_ORACLE_RELATION = "sent"


def _stable_hash_integer(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(),
        byteorder="big",
        signed=False,
    )


def _stable_hash_fraction(*parts: object) -> float:
    return _stable_hash_integer(*parts) / float(1 << 64)


def _validated_edge_index(
    edge_index: np.ndarray,
    *,
    node_count: int | None = None,
) -> np.ndarray:
    array = np.asarray(edge_index)
    if array.ndim != 2 or array.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, edge_count)")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("edge_index must contain integer node indices")
    result = array.astype(np.int64, copy=False)
    if result.size:
        if int(result.min()) < 0:
            raise ValueError("edge_index contains a negative node index")
        if node_count is not None and int(result.max()) >= node_count:
            raise ValueError("edge_index contains an out-of-range node index")
        if np.any(result[0] == result[1]):
            raise ValueError("diagnostic edge indices must not contain self-loops")
    return result


def _undirected_edges(
    edge_index: np.ndarray,
    *,
    node_count: int | None = None,
    require_symmetric: bool = False,
) -> set[tuple[int, int]]:
    array = _validated_edge_index(edge_index, node_count=node_count)
    directed = {
        (int(left), int(right))
        for left, right in array.T.tolist()
    }
    undirected = {
        canonical_pair(left, right)
        for left, right in directed
    }
    if require_symmetric:
        missing_reverse = [
            (left, right)
            for left, right in directed
            if (right, left) not in directed
        ]
        if missing_reverse:
            raise ValueError(
                "degree-preserving randomization requires a symmetric edge_index"
            )
    return undirected


def _directed_edge_index(
    undirected_edges: Iterable[tuple[int, int]],
) -> np.ndarray:
    directed: list[tuple[int, int]] = []
    canonical_edges = {
        canonical_pair(int(left), int(right))
        for left, right in undirected_edges
    }
    for canonical in sorted(canonical_edges):
        directed.append(canonical)
        directed.append((canonical[1], canonical[0]))
    if not directed:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray(directed, dtype=np.int64).T


def as_single_relation_edge_indices(
    edge_index: np.ndarray,
    *,
    relation: str = DEFAULT_ORACLE_RELATION,
    relation_names: Sequence[str] = RELATION_NAMES,
) -> dict[str, np.ndarray]:
    """Place a graph in one relation channel and keep every other channel empty."""
    names = tuple(str(name) for name in relation_names)
    if len(names) != len(set(names)):
        raise ValueError("relation_names must be unique")
    if relation not in names:
        raise ValueError(f"unknown target relation {relation!r}")
    validated = _validated_edge_index(edge_index)
    return {
        name: (
            validated.copy()
            if name == relation
            else np.empty((2, 0), dtype=np.int64)
        )
        for name in names
    }


def build_sparse_gold_oracle_edge_index(
    labels: Sequence[object] | np.ndarray,
    document_ids: Sequence[str],
    *,
    positions: Sequence[int | float] | None = None,
    offsets: Sequence[int] = (1, 2),
) -> np.ndarray:
    """Build a sparse, intra-document gold-label oracle graph.

    Mentions are ordered by ``positions`` within every document/label group.
    Each mention is linked to mentions one and two same-label positions away by
    default.  The resulting undirected maximum degree is therefore at most four.
    """
    label_values = np.asarray(labels, dtype=object)
    if label_values.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if len(document_ids) != len(label_values):
        raise ValueError("labels and document_ids must have the same length")
    node_count = len(label_values)
    if positions is None:
        position_values = np.arange(node_count, dtype=np.int64)
    else:
        position_values = np.asarray(positions)
        if position_values.ndim != 1 or len(position_values) != node_count:
            raise ValueError(
                "positions must be one-dimensional and aligned with labels"
            )
        if not np.issubdtype(position_values.dtype, np.number):
            raise ValueError("positions must be numeric")
        if not np.isfinite(position_values.astype(np.float64)).all():
            raise ValueError("positions must be finite")

    normalized_offsets_list: list[int] = []
    for offset in offsets:
        integer_offset = int(offset)
        if integer_offset != offset:
            raise ValueError("offsets must contain only integers")
        normalized_offsets_list.append(integer_offset)
    normalized_offsets = tuple(sorted(set(normalized_offsets_list)))
    if any(offset <= 0 for offset in normalized_offsets):
        raise ValueError("offsets must contain only positive integers")

    groups: dict[tuple[str, object], list[int]] = defaultdict(list)
    for index, (document_id, label) in enumerate(
        zip(document_ids, label_values.tolist(), strict=True)
    ):
        groups[(str(document_id), label)].append(index)

    edges: set[tuple[int, int]] = set()
    for nodes in groups.values():
        ordered = sorted(
            nodes,
            key=lambda index: (float(position_values[index]), index),
        )
        for offset in normalized_offsets:
            for left_position in range(max(0, len(ordered) - offset)):
                edges.add(
                    canonical_pair(
                        ordered[left_position],
                        ordered[left_position + offset],
                    )
                )
    return _directed_edge_index(edges)


def build_sparse_gold_oracle_edge_indices(
    labels: Sequence[object] | np.ndarray,
    document_ids: Sequence[str],
    *,
    positions: Sequence[int | float] | None = None,
    offsets: Sequence[int] = (1, 2),
    relation: str = DEFAULT_ORACLE_RELATION,
    relation_names: Sequence[str] = RELATION_NAMES,
) -> dict[str, np.ndarray]:
    """Return the sparse oracle graph with exactly one active relation channel."""
    return as_single_relation_edge_indices(
        build_sparse_gold_oracle_edge_index(
            labels,
            document_ids,
            positions=positions,
            offsets=offsets,
        ),
        relation=relation,
        relation_names=relation_names,
    )


def degree_preserving_randomize_edge_index_by_document(
    edge_index: np.ndarray,
    document_ids: Sequence[str],
    *,
    seed: int,
    swap_multiplier: int = 10,
) -> np.ndarray:
    """Rewire each document independently while preserving every node degree."""
    node_count = len(document_ids)
    edges = _undirected_edges(
        edge_index,
        node_count=node_count,
        require_symmetric=True,
    )
    by_document: dict[str, list[int]] = defaultdict(list)
    for index, document_id in enumerate(document_ids):
        by_document[str(document_id)].append(index)

    document_edges: dict[str, set[tuple[int, int]]] = {
        document_id: set() for document_id in by_document
    }
    for left, right in edges:
        left_document = str(document_ids[left])
        right_document = str(document_ids[right])
        if left_document != right_document:
            raise ValueError("edge_index contains a cross-document edge")
        document_edges[left_document].add((left, right))

    randomized: set[tuple[int, int]] = set()
    for document_id, global_nodes in sorted(by_document.items()):
        global_to_local = {
            global_index: local_index
            for local_index, global_index in enumerate(global_nodes)
        }
        local_to_global = {
            local_index: global_index
            for global_index, local_index in global_to_local.items()
        }
        local_edges = {
            canonical_pair(
                global_to_local[left],
                global_to_local[right],
            )
            for left, right in document_edges[document_id]
        }
        rewired = degree_preserving_edge_rewire(
            local_edges,
            len(global_nodes),
            seed=_stable_hash_integer(
                "diagnostic-edge-rewire",
                int(seed),
                document_id,
            )
            & 0xFFFFFFFF,
            swap_multiplier=int(swap_multiplier),
        )
        randomized.update(
            canonical_pair(
                local_to_global[left],
                local_to_global[right],
            )
            for left, right in rewired
        )
    return _directed_edge_index(randomized)


def degree_preserving_randomize_edge_indices_by_document(
    edge_indices: Mapping[str, np.ndarray],
    document_ids: Sequence[str],
    *,
    seed: int,
    relation: str = DEFAULT_ORACLE_RELATION,
    swap_multiplier: int = 10,
) -> dict[str, np.ndarray]:
    """Rewire the sole active relation and preserve the relation dictionary."""
    if relation not in edge_indices:
        raise ValueError(f"missing target relation {relation!r}")
    for name, index in edge_indices.items():
        if name != relation and np.asarray(index).size:
            raise ValueError(
                "only the target relation may be active in an oracle control"
            )
    randomized = degree_preserving_randomize_edge_index_by_document(
        edge_indices[relation],
        document_ids,
        seed=seed,
        swap_multiplier=swap_multiplier,
    )
    return {
        str(name): (
            randomized
            if name == relation
            else np.empty((2, 0), dtype=np.int64)
        )
        for name in edge_indices
    }


def semantic_union_pruned_to_same_label_edge_index(
    semantic_edge_indices: Mapping[str, np.ndarray],
    labels: Sequence[object] | np.ndarray,
    document_ids: Sequence[str],
    *,
    relations: Sequence[str] = ("sent", "repeat", "near"),
) -> np.ndarray:
    """Union semantic relations and retain only intra-document same-label edges."""
    label_values = np.asarray(labels, dtype=object)
    if label_values.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if len(document_ids) != len(label_values):
        raise ValueError("labels and document_ids must have the same length")
    requested = tuple(str(relation) for relation in relations)
    missing = [relation for relation in requested if relation not in semantic_edge_indices]
    if missing:
        raise ValueError(f"missing semantic relations: {missing}")

    union: set[tuple[int, int]] = set()
    for relation in requested:
        union.update(
            _undirected_edges(
                semantic_edge_indices[relation],
                node_count=len(label_values),
            )
        )

    pruned: set[tuple[int, int]] = set()
    for left, right in union:
        if str(document_ids[left]) != str(document_ids[right]):
            raise ValueError("semantic edge indices contain a cross-document edge")
        if label_values[left] == label_values[right]:
            pruned.add((left, right))
    return _directed_edge_index(pruned)


def semantic_union_pruned_to_same_label_edge_indices(
    semantic_edge_indices: Mapping[str, np.ndarray],
    labels: Sequence[object] | np.ndarray,
    document_ids: Sequence[str],
    *,
    relations: Sequence[str] = ("sent", "repeat", "near"),
    relation: str = DEFAULT_ORACLE_RELATION,
    relation_names: Sequence[str] = RELATION_NAMES,
) -> dict[str, np.ndarray]:
    """Return the pruned semantic union with exactly one active channel."""
    return as_single_relation_edge_indices(
        semantic_union_pruned_to_same_label_edge_index(
            semantic_edge_indices,
            labels,
            document_ids,
            relations=relations,
        ),
        relation=relation,
        relation_names=relation_names,
    )


def stable_nested_mention_mask(
    mention_ids: Sequence[str],
    severity: float,
    *,
    seed: int = 0,
    namespace: str = "diagnostic-mention-mask",
) -> np.ndarray:
    """Select mentions by a stable hash threshold.

    For a fixed ``seed`` and ``namespace``, masks at lower severities are exact
    subsets of masks at higher severities, and an ID's decision is independent
    of row order.
    """
    severity_value = float(severity)
    if not math.isfinite(severity_value) or not 0.0 <= severity_value <= 1.0:
        raise ValueError("severity must be a finite number in [0, 1]")
    normalized_ids = tuple(str(mention_id) for mention_id in mention_ids)
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("mention_ids must be unique")
    if severity_value == 0.0:
        return np.zeros(len(normalized_ids), dtype=bool)
    if severity_value == 1.0:
        return np.ones(len(normalized_ids), dtype=bool)
    return np.asarray(
        [
            _stable_hash_fraction(namespace, int(seed), mention_id)
            < severity_value
            for mention_id in normalized_ids
        ],
        dtype=bool,
    )


def stable_nested_mention_masks(
    mention_ids: Sequence[str],
    severities: Sequence[float],
    *,
    seed: int = 0,
    namespace: str = "diagnostic-mention-mask",
) -> dict[float, np.ndarray]:
    """Build multiple nested masks without using row positions."""
    values = sorted({float(severity) for severity in severities})
    return {
        severity: stable_nested_mention_mask(
            mention_ids,
            severity,
            seed=seed,
            namespace=namespace,
        )
        for severity in values
    }


def _normalized_probabilities(
    probabilities: np.ndarray,
    *,
    expected_rows: int,
    name: str,
) -> np.ndarray:
    source = np.asarray(probabilities)
    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim != 2 or len(array) != expected_rows or array.shape[1] == 0:
        raise ValueError(
            f"{name} must have shape ({expected_rows}, class_count)"
        )
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    row_sums = array.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError(f"{name} contains a row with zero probability mass")
    if np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-6):
        return source.astype(np.float32, copy=True)
    return (array / row_sums).astype(np.float32)


@dataclass(frozen=True, slots=True)
class EvidenceView:
    """Aligned evidence supplied both to node features and the local residual."""

    embeddings: np.ndarray
    local_probabilities: np.ndarray
    mask_indicator: np.ndarray
    mask: np.ndarray
    mode: str
    mention_ids: tuple[str, ...] | None = None

    @property
    def embeddings_with_mask(self) -> np.ndarray:
        """Embeddings augmented with the explicit corruption indicator."""
        return np.concatenate(
            [self.embeddings, self.mask_indicator],
            axis=1,
        ).astype(np.float32, copy=False)

    @property
    def aligned_features(self) -> np.ndarray:
        """Features containing the exact probability view used by the residual."""
        return np.concatenate(
            [
                self.embeddings,
                self.local_probabilities,
                self.mask_indicator,
            ],
            axis=1,
        ).astype(np.float32, copy=False)


def build_evidence_view(
    clean_embeddings: np.ndarray,
    clean_local_probabilities: np.ndarray,
    *,
    mode: str,
    mask: Sequence[bool] | np.ndarray | None = None,
    surface_embeddings: np.ndarray | None = None,
    surface_local_probabilities: np.ndarray | None = None,
    mix_weight: float = 1.0,
    mention_ids: Sequence[str] | None = None,
    surface_mention_ids: Sequence[str] | None = None,
) -> EvidenceView:
    """Create an aligned clean, surface-only, or zero-plus-uniform view.

    ``surface_only`` replaces (or convexly mixes) both masked embeddings and
    masked local probabilities with their surface-only counterparts.
    ``zero_uniform`` moves masked embeddings toward zero and their local
    probabilities toward the uniform distribution.  The same row mask is used
    in both channels, preventing accidental clean-local-probability leakage.
    """
    embeddings = np.asarray(clean_embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("clean_embeddings must be two-dimensional")
    if not np.isfinite(embeddings).all():
        raise ValueError("clean_embeddings must be finite")
    row_count = len(embeddings)
    clean_probabilities = _normalized_probabilities(
        clean_local_probabilities,
        expected_rows=row_count,
        name="clean_local_probabilities",
    )
    normalized_mode = str(mode).casefold().replace("+", "_")
    if normalized_mode == "zero__uniform":
        normalized_mode = "zero_uniform"
    if normalized_mode not in {"identity", "surface_only", "zero_uniform"}:
        raise ValueError(f"unknown evidence mode {mode!r}")
    weight = float(mix_weight)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("mix_weight must be a finite number in [0, 1]")

    clean_ids: tuple[str, ...] | None = None
    if mention_ids is not None:
        clean_ids = tuple(str(mention_id) for mention_id in mention_ids)
        if len(clean_ids) != row_count:
            raise ValueError("mention_ids are not aligned with clean embeddings")
        if len(clean_ids) != len(set(clean_ids)):
            raise ValueError("mention_ids must be unique")
    if surface_mention_ids is not None:
        if clean_ids is None:
            raise ValueError(
                "mention_ids are required when surface_mention_ids are supplied"
            )
        normalized_surface_ids = tuple(
            str(mention_id) for mention_id in surface_mention_ids
        )
        if normalized_surface_ids != clean_ids:
            raise ValueError(
                "surface evidence rows are not aligned with clean mention_ids"
            )

    if normalized_mode == "identity":
        active_mask = np.zeros(row_count, dtype=bool)
    else:
        if mask is None:
            raise ValueError(f"mask is required for {normalized_mode!r}")
        raw_mask = np.asarray(mask)
        if raw_mask.dtype.kind != "b":
            raise ValueError("mask must contain boolean values")
        active_mask = raw_mask.astype(bool, copy=False)
        if active_mask.ndim != 1 or len(active_mask) != row_count:
            raise ValueError("mask must be one-dimensional and row-aligned")

    viewed_embeddings = embeddings.copy()
    viewed_probabilities = clean_probabilities.copy()
    if normalized_mode == "surface_only":
        if clean_ids is None or surface_mention_ids is None:
            raise ValueError(
                "surface_only requires aligned mention_ids and "
                "surface_mention_ids"
            )
        if surface_embeddings is None or surface_local_probabilities is None:
            raise ValueError(
                "surface_only requires surface embeddings and probabilities"
            )
        surface_array = np.asarray(surface_embeddings, dtype=np.float32)
        if surface_array.shape != embeddings.shape:
            raise ValueError(
                "surface_embeddings must align with clean_embeddings"
            )
        if not np.isfinite(surface_array).all():
            raise ValueError("surface_embeddings must be finite")
        surface_probabilities = _normalized_probabilities(
            surface_local_probabilities,
            expected_rows=row_count,
            name="surface_local_probabilities",
        )
        if surface_probabilities.shape != clean_probabilities.shape:
            raise ValueError(
                "surface probabilities must match the clean class dimension"
            )
        viewed_embeddings[active_mask] = (
            (1.0 - weight) * embeddings[active_mask]
            + weight * surface_array[active_mask]
        )
        viewed_probabilities[active_mask] = (
            (1.0 - weight) * clean_probabilities[active_mask]
            + weight * surface_probabilities[active_mask]
        )
    elif normalized_mode == "zero_uniform":
        viewed_embeddings[active_mask] = (
            (1.0 - weight) * embeddings[active_mask]
        )
        uniform = np.full(
            clean_probabilities.shape[1],
            1.0 / clean_probabilities.shape[1],
            dtype=np.float32,
        )
        viewed_probabilities[active_mask] = (
            (1.0 - weight) * clean_probabilities[active_mask]
            + weight * uniform
        )

    viewed_probabilities = _normalized_probabilities(
        viewed_probabilities,
        expected_rows=row_count,
        name="viewed_local_probabilities",
    )
    indicator = active_mask.astype(np.float32).reshape(-1, 1)
    return EvidenceView(
        embeddings=viewed_embeddings.astype(np.float32, copy=False),
        local_probabilities=viewed_probabilities,
        mask_indicator=indicator,
        mask=active_mask.copy(),
        mode=normalized_mode,
        mention_ids=clean_ids,
    )


def stable_nested_document_subsets(
    documents: Sequence[Document],
    fractions: Sequence[float],
    *,
    seed: int = 0,
    train_split: str = "train",
) -> dict[float, tuple[str, ...]]:
    """Select nested source-stratified document subsets from training only."""
    normalized_fractions = sorted({float(fraction) for fraction in fractions})
    if any(
        not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0
        for fraction in normalized_fractions
    ):
        raise ValueError("fractions must be finite numbers in [0, 1]")

    document_ids = [str(document.doc_id) for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("document IDs must be unique")
    train_documents = [
        document
        for document in documents
        if document.split == train_split
    ]
    by_source: dict[str, list[Document]] = defaultdict(list)
    for document in train_documents:
        by_source[str(document.source)].append(document)
    ordered_by_source = {
        source: sorted(
            source_documents,
            key=lambda document: (
                _stable_hash_integer(
                    "diagnostic-document-subset",
                    int(seed),
                    source,
                    document.doc_id,
                ),
                document.doc_id,
            ),
        )
        for source, source_documents in sorted(by_source.items())
    }

    subsets: dict[float, tuple[str, ...]] = {}
    for fraction in normalized_fractions:
        selected: list[str] = []
        for source_documents in ordered_by_source.values():
            if fraction == 0.0:
                selected_count = 0
            else:
                selected_count = min(
                    len(source_documents),
                    max(1, math.ceil(fraction * len(source_documents))),
                )
            selected.extend(
                str(document.doc_id)
                for document in source_documents[:selected_count]
            )
        subsets[fraction] = tuple(sorted(selected))
    return subsets


@dataclass(frozen=True, slots=True)
class SurfaceLabelStats:
    support: int
    label_counts: tuple[tuple[str, int], ...]
    entropy_bits: float
    normalized_entropy: float
    majority_label: str
    majority_fraction: float

    @property
    def entropy(self) -> float:
        """Backward-compatible alias; entropy is measured in bits."""
        return self.entropy_bits

    @property
    def distinct_label_count(self) -> int:
        return len(self.label_counts)

    @property
    def ambiguous(self) -> bool:
        return self.distinct_label_count > 1


def build_train_surface_label_stats(
    documents: Sequence[Document],
    *,
    train_split: str = "train",
) -> dict[str, SurfaceLabelStats]:
    """Estimate surface ambiguity from training documents and no other split."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for document in documents:
        if document.split != train_split:
            continue
        for mention in document.mentions:
            counts[normalize_surface(mention.text)][str(mention.label)] += 1

    result: dict[str, SurfaceLabelStats] = {}
    for surface, label_counts in sorted(counts.items()):
        support = int(sum(label_counts.values()))
        ordered_counts = tuple(sorted(label_counts.items()))
        probabilities = np.asarray(
            [count / support for _, count in ordered_counts],
            dtype=np.float64,
        )
        entropy_bits = float(
            -np.sum(
                probabilities[probabilities > 0.0]
                * np.log2(probabilities[probabilities > 0.0])
            )
        )
        normalized_entropy = (
            entropy_bits / math.log2(len(probabilities))
            if len(probabilities) > 1
            else 0.0
        )
        majority_label, majority_count = min(
            ordered_counts,
            key=lambda item: (-item[1], item[0]),
        )
        result[surface] = SurfaceLabelStats(
            support=support,
            label_counts=ordered_counts,
            entropy_bits=entropy_bits,
            normalized_entropy=float(normalized_entropy),
            majority_label=majority_label,
            majority_fraction=float(majority_count / support),
        )
    return result


def repeat_ambiguity_cohort_masks(
    mentions: Sequence[Mention],
    train_surface_stats: Mapping[str, SurfaceLabelStats],
    *,
    ambiguity_min_support: int = 5,
    ambiguity_min_entropy_bits: float = 0.5,
) -> dict[str, np.ndarray]:
    """Build deployable repeat/ambiguity cohorts without reading target labels."""
    if int(ambiguity_min_support) != ambiguity_min_support:
        raise ValueError("ambiguity_min_support must be an integer")
    minimum_support = int(ambiguity_min_support)
    minimum_entropy = float(ambiguity_min_entropy_bits)
    if minimum_support < 1:
        raise ValueError("ambiguity_min_support must be positive")
    if not math.isfinite(minimum_entropy) or minimum_entropy < 0.0:
        raise ValueError(
            "ambiguity_min_entropy_bits must be finite and non-negative"
        )
    within_document_counts = Counter(
        (str(mention.doc_id), normalize_surface(mention.text))
        for mention in mentions
    )
    repeated = np.asarray(
        [
            within_document_counts[
                (str(mention.doc_id), normalize_surface(mention.text))
            ]
            >= 2
            for mention in mentions
        ],
        dtype=bool,
    )
    seen = np.asarray(
        [
            normalize_surface(mention.text) in train_surface_stats
            for mention in mentions
        ],
        dtype=bool,
    )
    ambiguous = np.asarray(
        [
            (
                train_surface_stats.get(normalize_surface(mention.text))
                is not None
                and (
                    train_surface_stats[
                        normalize_surface(mention.text)
                    ].support
                    >= minimum_support
                )
                and (
                    train_surface_stats[
                        normalize_surface(mention.text)
                    ].entropy_bits
                    >= minimum_entropy
                )
            )
            for mention in mentions
        ],
        dtype=bool,
    )
    train_repeated = np.asarray(
        [
            (
                train_surface_stats.get(normalize_surface(mention.text))
                is not None
                and train_surface_stats[
                    normalize_surface(mention.text)
                ].support
                >= 2
            )
            for mention in mentions
        ],
        dtype=bool,
    )
    return {
        "all": np.ones(len(mentions), dtype=bool),
        "repeat": repeated,
        "singleton": ~repeated,
        "train_seen": seen,
        "train_unseen": ~seen,
        "train_repeat": train_repeated,
        "train_ambiguous": ambiguous,
        "train_unambiguous": seen & ~ambiguous,
        "repeat_train_ambiguous": repeated & ambiguous,
        "repeat_train_unambiguous": repeated & seen & ~ambiguous,
    }
