# Reviewer 1 Control Experiment Without the Degradation Indicator

## Status and purpose

This protocol is fixed before executing the new revision experiment. The official test split has already been inspected in the original study, so the resulting evidence is controlled but not independently confirmatory.

The experiment quantifies whether document-level recovery through exact-repeat links persists when the model is not explicitly told which mentions have been artificially weakened. It directly addresses Reviewer 1, Comment 1 and the corresponding editorial clarification.

## Control definition

The original model input contains a binary degradation indicator `q_i`. The new control keeps the feature column and the entire model architecture unchanged, but holds the indicator at zero for every mention during training, validation and testing. This choice avoids a test-only distribution shift and preserves the nominal parameter count and head parameterization.

Two indicator regimes are trained in the same run:

- `q_on`: the original protocol, with `q_i=1` for weakened mentions and `q_i=0` otherwise;
- `q_off`: the matched control, with the same weakening masks but `q_i=0` for all mentions.

Local embeddings and local-classifier probabilities are identical across the two regimes. Only the degradation-indicator values supplied to the relational heads differ.

## Models and training

For each optimization seed, a single full-data local model and its five-fold out-of-fold training probabilities are shared across both indicator regimes. The following relational heads are retrained separately in each regime:

- `N0_node_only`: no inter-mention edges;
- `G_repeat`: exact normalized-repeat edges.

`AVG_repeat` is retained as a non-trainable reference. Its mixing coefficient is selected on the validation sample from the original fixed grid `[0.0, 0.25, 0.50, 0.75, 1.0]`, separately for each indicator regime, using only the corresponding retrained `N0_node_only` predictions.

The graph heads are trained under target-only surface-context weakening. Training and validation each contain one deterministically selected target per exact-repeat group. Target selection does not use the label.

## Fixed seeds and data rules

- optimization seeds: `17, 29, 43, 59, 71, 89, 101, 113, 127, 149`;
- test target-selection realizations: `101, 211, 307, 401, 503`;
- training target-selection seed: `314159`;
- validation target-selection seed: `161803`;
- five document-level folds for out-of-fold local probabilities;
- unchanged official NER-UK 2.0 test split;
- unchanged frozen XLM-RoBERTa-base embedding caches;
- unchanged target-only and whole-component surface-context weakening procedures;
- unchanged early stopping, optimizer settings and class weighting.

## Evaluation conditions

Every retrained head is evaluated on the same 800 target mentions under three conditions:

- `clean_targets`: no artificial weakening; evaluation is restricted to the selected targets;
- `target_surface`: only the selected target is weakened, while its repeat peers remain clean;
- `component_surface`: the selected target and all mentions in its exact-repeat component are weakened.

The whole-component condition remains a diagnostic intervention because it applies multiple weakened neighbors although training weakens one target per component.

## Fixed estimands

The primary family contains three paired contrasts, all measured in target accuracy with a smallest effect size of interest of `0.005`:

1. `q_off_G_repeat_minus_N0_target_surface`: recovery without an explicit degradation cue;
2. `indicator_dependence_of_graph_gain`: `(G_repeat - N0)_{q_on} - (G_repeat - N0)_{q_off}` under target-only weakening;
3. `q_off_context_borrowing_target_minus_component`: `(G_repeat - N0)_{target_surface,q_off} - (G_repeat - N0)_{component_surface,q_off}`.

Holm correction is applied within this three-contrast primary family.

The following reference contrasts are reported with paired 95% confidence intervals and are explicitly treated as secondary/descriptive:

- `q_on_G_repeat_minus_N0_target_surface`, to replicate the original flagged result with the current code and environment;
- `q_off_AVG_repeat_minus_N0_target_surface`;
- `q_off_G_repeat_minus_AVG_repeat_target_surface`;
- `q_off_G_repeat_minus_N0_clean_targets`.

## Uncertainty analysis

The analysis uses the same crossed bootstrap design as the original diagnostic experiment:

- 20,000 bootstrap replicates;
- optimization seeds and source-stratified test documents resampled with replacement;
- target-selection realizations resampled within seed;
- identical resampled units used for every term of a contrast;
- two-sided centered-bootstrap p-values;
- 95% percentile confidence intervals.

## Required outputs

- machine-readable configuration and manifest with source/config checksums;
- per-seed model checkpoints for all four trained heads;
- validation alpha-selection records for `AVG_repeat`;
- test predictions for every seed, condition, realization and indicator regime;
- raw and summarized metrics;
- crossed-bootstrap table for all fixed estimands;
- compact revision table suitable for insertion into the manuscript;
- run log containing Python, NumPy, PyTorch, Transformers, CUDA and GPU versions.

## Interpretation boundary

Persistence of recovery in `q_off` supports context transfer under unflagged artificial weakening. It does not establish robustness to naturally occurring missing or corrupted context. Failure to preserve the effect would narrow the claim to recovery when degradation is explicitly signaled. Neither outcome is interpreted as end-to-end NER evidence because mention boundaries remain known.
