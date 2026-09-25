# Reviewer 2 Morphological Edge Control Protocol

Status: fixed before installing the morphological analyzer, constructing lemma graphs, inspecting lemma-based test outcomes, or training new heads.

Date fixed: 2026-09-23, Europe/Kyiv.

## Reviewer request

Reviewer 2 asks for an alternative edge rule based on lemma or morphological normalization because exact surface equality misses inflectional variants in Ukrainian.

## Scientific question

Does deterministic token-level Ukrainian lemmatization improve graph coverage or intervention-free mention-type classification relative to exact normalized repetition?

This is a post-review robustness analysis on the previously inspected official test split. It is not an independent confirmation.

## Morphological normalizer

- Analyzer: `pymorphy3==2.0.6`.
- Ukrainian dictionary: `pymorphy3-dicts-uk==2.4.1.1.1663094765`.
- Analyzer initialization: `pymorphy3.MorphAnalyzer(lang="uk")`.
- Input preprocessing: apply the manuscript's deterministic Unicode, quote, case, whitespace, and outer-punctuation normalization; split the remaining mention into alphanumeric word tokens and intervening non-space punctuation tokens.
- Each word token is replaced with the `normal_form` of the analyzer's highest-ranked parse. Non-word tokens are retained in normalized form.
- The normalized token sequence is joined deterministically with single spaces.
- The analyzer is context-free. Its first-ranked analysis is used without gold labels, split-specific tuning, or manual correction.

## Graph rule

- Construct a separate graph for every document.
- Connect every pair of distinct mentions whose complete lemma-normalized token sequences are equal.
- Use a complete undirected clique within each lemma group, matching the topology convention of the exact-repeat graph.
- Do not connect mentions across documents.
- Do not use entity types, test labels, model predictions, or manual audit decisions in graph construction.

## Compared configurations

- `N0_node_only`: no-edge baseline.
- `G_repeat`: original exact normalized-repeat graph.
- `G_lemma`: lemma-normalized repeat graph.
- `AVG_repeat`: validation-selected simple averaging on the exact-repeat graph.
- `AVG_lemma`: validation-selected simple averaging on the lemma graph.

`N0_node_only`, `G_repeat`, and `G_lemma` will be trained in the same new run so that software environment, local probabilities, validation rule, and optimization seeds are matched.

## Fixed data and optimization settings

- Same NER-UK revision and document splits as the manuscript.
- Same frozen XLM-RoBERTa-base embedding cache.
- Same local classifier and graph-head hyperparameters as the corrected full-data intervention-free experiment.
- Same ten optimization seeds: 17, 29, 43, 59, 71, 89, 101, 113, 127, 149.
- Validation metric for trained heads: macro-F1 on the full validation split.
- Averaging grid: `{0, 0.25, 0.50, 0.75, 1}` selected separately for the exact and lemma graphs using validation macro-F1.
- Test evaluation: the full 6,931-mention test split.
- No degradation indicator; `q_i` is held at zero throughout this intervention-free run.

## Structural outcomes

Report, for exact and lemma graphs:

- undirected edge count;
- number and percentage of covered mentions;
- label homogeneity as a descriptive diagnostic only;
- number of newly covered mentions under lemmatization;
- number and homogeneity of lemma edges that connect different original normalized surface forms.

Coverage is graph participation, not recall of true same-entity links. The corpus has no entity identifiers or coreference chains.

## Predictive outcomes

Primary metric: macro-F1 on the full intervention-free test split.

Secondary metric: accuracy.

Primary post-review family, Holm-corrected across two contrasts:

1. `G_lemma - G_repeat` in macro-F1.
2. `G_lemma - N0_node_only` in macro-F1.

Secondary reference contrasts:

- `AVG_lemma - N0_node_only` in macro-F1;
- `G_lemma - AVG_lemma` in macro-F1;
- corresponding accuracy contrasts.

## Uncertainty

- 20,000 crossed-bootstrap replicates.
- Optimization seeds and source-stratified test documents are resampled as in the manuscript.
- Report the mean paired contrast, 95% confidence interval, two-sided centered-bootstrap p-value, and Holm-adjusted p-value for the primary family.
- Practical threshold for macro-F1 remains 0.010.

## Interpretation boundaries

- A lemma graph is an alternative deterministic edge rule, not gold coreference.
- Increased coverage can add both useful and false links.
- Label homogeneity is a diagnostic for the present untyped aggregation, not a universal edge-quality measure.
- `pymorphy3` is context-free and can assign incorrect lemmas to proper names or ambiguous forms.
- Results on the reused test split are controlled post-review evidence, not independent confirmation.
- The experiment does not change the known-boundary task, frozen encoder, or artificial context-recovery conclusions.

## Required artifacts

- exact configuration and software versions;
- lemma normalization unit tests;
- structural graph statistics by split and entity type;
- raw per-seed metrics and predictions;
- trained head checkpoints;
- bootstrap contrasts;
- a concise manuscript table and point-by-point response.
