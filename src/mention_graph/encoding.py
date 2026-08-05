from __future__ import annotations

import hashlib
import json
import math
import re
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np

from .data import flatten_mentions
from .schema import Document, Mention


WORD_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
ASR_SUBSTITUTIONS = {
    "і": "и",
    "и": "і",
    "г": "х",
    "х": "г",
    "е": "и",
    "у": "ю",
    "я": "а",
    "є": "е",
}


def _context_window(
    document: Document,
    mention: Mention,
    *,
    max_characters: int,
) -> tuple[str, int, int]:
    sentence_left = min(mention.sentence_start, mention.start)
    sentence_right = max(mention.sentence_end, mention.end)
    if sentence_right - sentence_left <= max_characters:
        left, right = sentence_left, sentence_right
    else:
        half = max_characters // 2
        center = (mention.start + mention.end) // 2
        left = max(sentence_left, center - half)
        right = min(sentence_right, left + max_characters)
        left = max(sentence_left, right - max_characters)
    return (
        document.text[left:right],
        mention.start - left,
        mention.end - left,
    )


def _noise_seed(mention_id: str, seed: int) -> int:
    return zlib.crc32(f"{seed}:{mention_id}".encode("utf-8")) & 0xFFFFFFFF


def apply_controlled_noise(
    text: str,
    mention_start: int,
    mention_end: int,
    *,
    kind: str,
    level: float,
    seed: int,
) -> tuple[str, int, int]:
    if not 0.0 <= level <= 1.0:
        raise ValueError(f"noise level must be in [0, 1], got {level}")
    if kind == "lowercase":
        return text.lower(), mention_start, mention_end

    generator = np.random.default_rng(seed)
    characters = list(text)
    if kind == "context_dropout":
        for match in WORD_RE.finditer(text):
            left, right = match.span()
            overlaps_mention = left < mention_end and right > mention_start
            if not overlaps_mention and generator.random() < level:
                for position in range(left, right):
                    if not characters[position].isspace():
                        characters[position] = " "
    elif kind == "asr":
        for position, character in enumerate(characters):
            if mention_start <= position < mention_end:
                continue
            lower = character.lower()
            if lower in ASR_SUBSTITUTIONS and generator.random() < level:
                replacement = ASR_SUBSTITUTIONS[lower]
                characters[position] = (
                    replacement.upper() if character.isupper() else replacement
                )
    else:
        raise ValueError(f"unknown controlled noise kind {kind!r}")
    return "".join(characters), mention_start, mention_end


def mention_contexts(
    documents: Iterable[Document],
    *,
    max_characters: int,
    noise: dict[str, object] | None = None,
) -> tuple[list[Mention], list[str], list[tuple[int, int]]]:
    documents = sorted(documents, key=lambda document: document.doc_id)
    document_by_id = {document.doc_id: document for document in documents}
    mentions = flatten_mentions(documents)
    contexts: list[str] = []
    spans: list[tuple[int, int]] = []
    for mention in mentions:
        context, start, end = _context_window(
            document_by_id[mention.doc_id],
            mention,
            max_characters=max_characters,
        )
        if noise is not None:
            context, start, end = apply_controlled_noise(
                context,
                start,
                end,
                kind=str(noise["kind"]),
                level=float(noise["level"]),
                seed=_noise_seed(mention.mention_id, int(noise.get("seed", 0))),
            )
        contexts.append(context)
        spans.append((start, end))
    return mentions, contexts, spans


def _hash_index(feature: str, dimension: int) -> tuple[int, float]:
    checksum = zlib.crc32(feature.encode("utf-8")) & 0xFFFFFFFF
    index = checksum % dimension
    sign = 1.0 if checksum & 0x80000000 else -1.0
    return index, sign


def _hashing_embedding(
    text: str,
    mention_span: tuple[int, int],
    *,
    dimension: int,
) -> np.ndarray:
    start, end = mention_span
    vector = np.zeros(dimension, dtype=np.float32)
    matches = list(WORD_RE.finditer(text))
    mention_token_positions = [
        position
        for position, match in enumerate(matches)
        if match.start() < end and match.end() > start
    ]
    mention_center = (
        sum(mention_token_positions) / len(mention_token_positions)
        if mention_token_positions
        else 0.0
    )
    normalized_tokens = [match.group(0).casefold() for match in matches]

    features: list[tuple[str, float]] = []
    for position, match in enumerate(matches):
        token = normalized_tokens[position]
        if match.start() < end and match.end() > start:
            region = "M"
            distance = 0
        elif match.end() <= start:
            region = "L"
            distance = min(8, round(abs(position - mention_center)))
        else:
            region = "R"
            distance = min(8, round(abs(position - mention_center)))
        features.append((f"w:{region}:{distance}:{token}", 1.0))
        if position + 1 < len(matches):
            features.append(
                (
                    f"b:{region}:{token}_{normalized_tokens[position + 1]}",
                    0.75,
                )
            )

    surface = text[start:end].casefold()
    compact_surface = re.sub(r"\s+", " ", surface).strip()
    for size in (2, 3, 4):
        for position in range(max(0, len(compact_surface) - size + 1)):
            features.append(
                (f"c{size}:{compact_surface[position:position + size]}", 0.5)
            )
    features.extend(
        [
            (f"shape:uppercase:{any(char.isupper() for char in text[start:end])}", 1.0),
            (f"shape:digits:{any(char.isdigit() for char in text[start:end])}", 1.0),
            (f"shape:length:{min(10, len(text[start:end]) // 4)}", 1.0),
        ]
    )
    for feature, value in features:
        index, sign = _hash_index(feature, dimension)
        vector[index] += sign * value
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


def encode_hashing(
    contexts: list[str],
    spans: list[tuple[int, int]],
    *,
    dimension: int,
) -> np.ndarray:
    return np.stack(
        [
            _hashing_embedding(text, span, dimension=dimension)
            for text, span in zip(contexts, spans, strict=True)
        ],
        axis=0,
    ).astype(np.float32)


def encode_transformer(
    contexts: list[str],
    spans: list[tuple[int, int]],
    *,
    model_name: str,
    model_cache: Path,
    batch_size: int,
    max_tokens: int,
    device: str,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_cache.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(model_cache),
        use_fast=True,
    )
    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=str(model_cache),
    )
    model.eval()
    model.to(device)
    rows: list[np.ndarray] = []
    use_autocast = device.startswith("cuda")

    for left in range(0, len(contexts), batch_size):
        batch_contexts = contexts[left : left + batch_size]
        batch_spans = spans[left : left + batch_size]
        encoded = tokenizer(
            batch_contexts,
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        model_inputs = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_autocast,
            ):
                hidden = model(**model_inputs).last_hidden_state
        offsets_array = offsets.cpu().numpy()
        for batch_index, (mention_start, mention_end) in enumerate(batch_spans):
            mask = np.asarray(
                [
                    token_start < mention_end
                    and token_end > mention_start
                    and token_end > token_start
                    for token_start, token_end in offsets_array[batch_index]
                ],
                dtype=bool,
            )
            if not mask.any():
                mask[0] = True
            token_mask = torch.as_tensor(mask, device=hidden.device)
            pooled = hidden[batch_index, token_mask].float().mean(dim=0)
            pooled /= pooled.norm(p=2).clamp_min(1e-12)
            rows.append(pooled.cpu().numpy().astype(np.float32))
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.stack(rows, axis=0)


def _documents_fingerprint(documents: Iterable[Document]) -> str:
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda item: item.doc_id):
        digest.update(document.doc_id.encode("utf-8"))
        digest.update(document.split.encode("utf-8"))
        digest.update(hashlib.sha256(document.text.encode("utf-8")).digest())
        for mention in document.sorted_mentions():
            digest.update(
                f"{mention.mention_id}:{mention.start}:{mention.end}:{mention.label}".encode(
                    "utf-8"
                )
            )
    return digest.hexdigest()


def _cache_key(
    documents: Iterable[Document],
    encoder: dict[str, object],
    noise: dict[str, object] | None,
) -> str:
    payload = {
        "documents": _documents_fingerprint(documents),
        "encoder": encoder,
        "noise": noise,
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]


def encode_with_cache(
    documents: list[Document],
    *,
    encoder: dict[str, object],
    cache_root: Path,
    model_cache: Path,
    device: str,
    noise: dict[str, object] | None = None,
) -> tuple[list[str], np.ndarray, Path]:
    cache_root.mkdir(parents=True, exist_ok=True)
    mentions, contexts, spans = mention_contexts(
        documents,
        max_characters=int(encoder.get("max_characters", 600)),
        noise=noise,
    )
    mention_ids = [mention.mention_id for mention in mentions]
    key = _cache_key(documents, encoder, noise)
    cache_path = cache_root / f"embeddings-{key}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_ids = cached["mention_ids"].tolist()
        if cached_ids != mention_ids:
            raise ValueError(f"{cache_path}: cached mention order does not match corpus")
        return mention_ids, cached["embeddings"].astype(np.float32), cache_path

    kind = str(encoder["kind"])
    if kind == "hashing":
        embeddings = encode_hashing(
            contexts,
            spans,
            dimension=int(encoder.get("dimension", 256)),
        )
    elif kind == "transformer":
        embeddings = encode_transformer(
            contexts,
            spans,
            model_name=str(encoder["model_name"]),
            model_cache=model_cache,
            batch_size=int(encoder.get("batch_size", 32)),
            max_tokens=int(encoder.get("max_tokens", 256)),
            device=device,
        )
    else:
        raise ValueError(f"unknown encoder kind {kind!r}")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(mention_ids):
        raise RuntimeError("encoder returned an invalid embedding matrix")
    np.savez_compressed(
        cache_path,
        mention_ids=np.asarray(mention_ids),
        embeddings=embeddings.astype(np.float32),
    )
    return mention_ids, embeddings, cache_path

