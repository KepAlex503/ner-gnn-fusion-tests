# Document-level inter-mention links for Ukrainian named-entity type classification

Code for the controlled experiments in the article *Controlled Evaluation of
Document-Level Inter-Mention Links for Named Entity Type Classification in
Ukrainian Texts* (O. Hryshyn, V. Shymkovych, D. Grela; under review).

The study asks whether links between repeated mentions within a document
improve the classification of entity types when mention boundaries are
known. It compares a graph-refinement head with a baseline that has the same
head parameterization but exchanges no messages, and it adds
degree-preserving topology shuffling, simple averaging, gold-label oracle
diagnostics, and controlled weakening of local context.

Scope:

- **Task:** entity *type* classification of gold NER-UK 2.0 spans (all 13
  classes, nested spans included). Boundary detection is not evaluated, so
  results are not end-to-end NER scores.
- **Encoder:** frozen `FacebookAI/xlm-roberta-base`. All compared heads
  receive identical local representations; the encoder is never fine-tuned.

The tag [`revision-1`](https://github.com/KepAlex503/ner-gnn-fusion-tests/tree/revision-1) marks the code state used for
the revised manuscript.

## Mapping from the paper to the code

| Paper | What it covers | Command | Configuration |
|---|---|---|---|
| Exploratory stage (E1–E5) | Unmatched local vs. graph comparison, 3 seeds | `run` | `configs/neruk_xlmr.json` |
| Exploratory stage (E6–E10) | Relation-aware follow-up | `run-followup` | `configs/neruk_xlmr_followup.json` |
| Second stage E11–E15 | Oracle diagnostics, intervention-free comparison, context weakening, training-data volume, mention groups | `run-diagnostics` | `configs/neruk_xlmr_diagnostics.json` |
| E13-R (post-review) | Degradation indicator `q_i` enabled vs. held at zero, matched retraining of `N0` and `G_rep` | `run-reviewer1-control` | `configs/neruk_xlmr_reviewer1_q_control.json` |
| E12-M (post-review) | Exact-repeat vs. token-level lemma edges | `run-reviewer2-morphology` | `configs/neruk_xlmr_reviewer2_morphology.json` |
| Table 4, coverage audit sample | Exact-repeat coverage by type; deterministic audit sample | `scripts/reviewer1_coverage_audit.py` | `configs/neruk_xlmr_diagnostics.json` |
| Table 6 | Layer-wise parameter breakdown | `scripts/reviewer1_parameter_breakdown.py` | — |

The exploratory stage generated hypotheses; the paper's inferential results
come from the second stage and the two post-review controls. The protocols
for the post-review analyses, fixed before the new models were trained, are
in [`docs/protocols/`](docs/protocols/).

Model names in the code differ slightly from the paper: `N0_node_only` is
`N0`, `G_repeat` is `G_rep`, `G_lemma` is `G_lem`, `AVG_repeat` / `AVG_lemma`
are `A_rep` / `A_lem`, `G_semantic_union` is `G_comb`, `O_pruned_union` is
`G_clean`, `O_label_sparse` is `G_gold`, and `GoldVote_label_sparse` is `V_gold`.

## Installation

Python 3.11 or newer is required; the reported runs used Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,download,morphology]"
```

For GPU runs, install the PyTorch build that matches your CUDA driver
(the reported runs used PyTorch 2.13.0 with CUDA 13.0 on an NVIDIA RTX 4070
SUPER). Exact versions of the direct dependencies are listed in
`requirements-lock.txt`. The `morphology` extra is needed only for E12-M.

## Data and model

```bash
python scripts/download_neruk.py    # NER-UK 2.0 at commit 7772b458...
python scripts/download_xlmr.py     # XLM-R base at commit e73636d4...
```

Both scripts pin the revisions used in the paper. NER-UK 2.0
(<https://github.com/lang-uk/ner-uk>) is distributed by its authors under
CC BY-NC-SA 4.0 and is not included in this repository. Files are placed
under `data/external/ner-uk/` and `data/models/xlm-roberta-base/`; encoder
outputs are cached under `data/cache/`.

## Running

Every command writes to `artifacts/<config name>/` unless `--output` is
given, and the post-review runners refuse to overwrite a non-empty output
directory.

```bash
# Quick checks without the transformer
python -m mention_graph.cli run --config configs/synthetic_smoke.json
python -m mention_graph.cli run-diagnostics --config configs/neruk_hashing_diagnostics_smoke.json
python -m mention_graph.cli run-reviewer1-control --config configs/reviewer1_q_control_smoke.json

# Reduced E12-M run (uses the transformer encoder)
python -m mention_graph.cli run-reviewer2-morphology --config configs/neruk_xlmr_reviewer2_morphology_smoke.json

# Paper experiments
python -m mention_graph.cli run-diagnostics --config configs/neruk_xlmr_diagnostics.json
python -m mention_graph.cli run-reviewer1-control --config configs/neruk_xlmr_reviewer1_q_control.json
python -m mention_graph.cli run-reviewer2-morphology --config configs/neruk_xlmr_reviewer2_morphology.json

# Corpus statistics without training
python -m mention_graph.cli inspect-data --config configs/neruk_xlmr.json
```

Auxiliary analyses:

```bash
python scripts/reviewer1_coverage_audit.py \
  --config configs/neruk_xlmr_diagnostics.json \
  --project-root . --output-directory artifacts/coverage-audit
python scripts/reviewer1_parameter_breakdown.py --output artifacts/parameter_breakdown.csv
```

`scripts/summarize_reviewer1_audit.py` summarizes the manual audit decisions;
the annotation file itself is not distributed (see below).

Tests:

```bash
pytest
```

## Outputs

Each run directory contains a `run_manifest.json` (configuration, software
versions, GPU, and SHA-256 checksums of the configuration and source code),
raw and summarized metrics, per-seed checkpoints, per-mention test
predictions, the validation-only selection of the averaging coefficient, and
crossed-bootstrap contrasts (optimization seeds crossed with
source-stratified test documents, 20,000 replicates). The diagnostic run also
writes the tables used in the paper (`table_g1_*.csv` to `table_g9_*.csv`) and
frozen cohort masks.

The trained checkpoints, predictions, and bootstrap outputs reported in the
paper, as well as the manual coverage-audit annotations, are available from
the corresponding author on reasonable request.

## Reproducibility notes

- Seeds, target selections, topology realizations, and bootstrap seeds are
  fixed in the configuration files.
- The original second-stage run (E11–E15) was executed on Windows; the
  post-review controls were executed on Linux with the versions in
  `requirements-lock.txt`. GPU non-determinism and platform differences can
  change absolute values in the last digits; the paper compares the two
  indicator regimes only within the same run.

## Repository layout

```text
configs/                  experiment configurations (full runs and smoke tests)
scripts/                  data/model download and auxiliary analyses
src/mention_graph/        data loading, encoding, graphs, models, training, statistics
tests/                    automated checks (leakage, graphs, metrics, controls)
docs/protocols/           protocols for the post-review analyses
docs/working-notes-uk/    internal working notes in Ukrainian (not the reviewed analysis)
```

## Citation

If you use this code, please cite the article above; full bibliographic
details will be added after publication.

## License

The code is released under the [MIT License](LICENSE). The NER-UK 2.0 corpus
and the XLM-RoBERTa model are not part of this repository and remain under
their own licenses (CC BY-NC-SA 4.0 and MIT, respectively).
