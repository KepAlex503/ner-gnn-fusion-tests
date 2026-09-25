# Reviewer 1 Exact-Repeat Coverage and Missed-Link Audit

## Status and purpose

This descriptive protocol is fixed before inspecting the audit sample. It addresses Reviewer 1, Comment 4 by separating exact-repeat edge coverage from the unobserved recall of links between mentions of the same real-world entity.

## Population and exact coverage

The population is the unchanged official NER-UK 2.0 test split used in the manuscript. A mention is *covered* when its deterministically normalized surface form occurs at least twice in the same document and it therefore has at least one exact-repeat neighbor. Coverage is reported as a numerator, denominator and percentage overall and for each of the 13 entity types.

NER-UK 2.0 Brat annotations provide mention spans and types but no entity identifiers or coreference chains. Consequently, exact same-entity link recall cannot be computed automatically and exact-repeat coverage must not be described as recall.

## Deterministic manual-audit sample

The audit population comprises test mentions with no exact-repeat neighbor. Mentions are ordered by the SHA-256 digest of the UTF-8 string `reviewer1-exact-repeat-audit-v1\0<mention_id>`; the first 100 mentions form the audit sample. This produces a reproducible uniform sample of uncovered mentions without selecting examples on the basis of their content or outcome.

For each sampled mention, the auditor inspects the complete source document, the mention's sentence, and all other annotated mentions in that document. The question is whether a clearly coreferential mention is present but missed because its normalized surface form differs.

## Fixed categories

Each sampled mention receives exactly one primary category:

- `genuine_singleton_or_no_identifiable_coreferent`: no clearly coreferential annotated mention is identifiable elsewhere in the document;
- `inflectional_variant`: a clear coreferent differs primarily through Ukrainian inflection;
- `abbreviation_or_name_shortening`: a clear coreferent uses an abbreviation, acronym, surname-only form, first-name-only form, or another systematic shortening/expansion;
- `orthographic_variant`: a clear coreferent differs through spelling, punctuation, spacing, transliteration, capitalization beyond deterministic normalization, or typographic variation;
- `other_lexical_variant`: a clear coreferent uses a different lexicalized name or description not covered above;
- `uncertain`: the document does not support a sufficiently reliable decision.

The audit records the candidate coreferential surface form and a brief rationale for every non-singleton or uncertain decision.

## Reporting boundary

The audit is a small, single-auditor descriptive analysis on a previously inspected test set. Its proportions are reported with raw counts and a Wilson 95% interval for the aggregate fraction of uncovered mentions with a manually identifiable missed coreferent. They are not treated as a gold-standard coreference annotation, a new confirmatory test, or an evaluation of a particular morphological analyzer.
