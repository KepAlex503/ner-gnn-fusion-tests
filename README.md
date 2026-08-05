# Графове уточнення типів українських іменованих сутностей

Цей репозиторій реалізує експерименти E1-E5 зі статті про уточнення типів
сутностей за відомих меж згадок. Основний режим використовує NER-UK 2.0,
заморожений `FacebookAI/xlm-roberta-base`, локальний MLP та двошарову
GraphSAGE. Окремий синтетичний режим перевіряє весь протокол без зовнішніх
даних або моделей.

Уже виконані прогони лежать у:

- `artifacts/neruk-xlmr` — основний XLM-R експеримент;
- `artifacts/neruk-xlmr-followup` — причинно чистіші relation-aware
  експерименти E6–E10 із matched controls;
- `artifacts/neruk-hashing-pilot` — швидкий пілот на реальному корпусі;
- `artifacts/synthetic-smoke` — контрольний синтетичний експеримент.

Інтерпретація результатів і готові значення для таблиць статті наведені в
`EXPERIMENT_REPORT_UK.md`.

Декомпозиція внеску голови, node statistics і власне adjacency, а також
typed/random/type-shuffle controls наведені в
`FOLLOWUP_EXPERIMENT_REPORT_UK.md`.

## Що саме реалізовано

- документне розділення без перетину ідентифікаторів;
- label-aware validation, виділений тільки з офіційної DEV-частини;
- читання всіх 21 993 Brat-згадок, включно з вкладеними та 49 згадками через
  перенос рядка;
- перевірка точних і майже тотожних документів між поділами;
- локальне span-pooling представлень XLM-R;
- п'ятискладкові позаскладкові локальні ймовірності для навчальних графів;
- ребра спільного речення, повторної нормалізованої форми та близькості;
- контроль без ребер і випадковий контроль із точним збереженням топології
  графа через перестановку вузлів у межах документа;
- E1: локальна модель, узгодження повторів, GraphSAGE;
- E2: `none`, `sent`, `repeat`, `near`, `all`, `random`;
- E3: precision/recall/F1 за 13 класами та матриці помилок;
- E4: групи частоти `1`, `2`, `3+`;
- E5: нижній регістр, видалення контекстних слів та ASR-подібні заміни;
- 3 парні початкові значення, середнє, стандартне відхилення і документний
  paired bootstrap;
- збереження ймовірностей кожної згадки, моделей, конфігурацій, графової
  діагностики та SHA-256 конфігурації/коду.

## Структура

```text
configs/                    конфігурації трьох режимів
scripts/                    завантаження NER-UK і XLM-R
src/mention_graph/          код даних, графа, моделей і звітування
tests/                      автоматичні перевірки leakage, графів і метрик
artifacts/                  результати прогонів
data/external/ner-uk/       офіційний корпус, не додається до Git
data/models/xlm-roberta-base/  ваги моделі, не додаються до Git
```

## Встановлення

Команди для Windows PowerShell:

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install torch transformers
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

Для NVIDIA GPU варто встановити офіційну CUDA-збірку PyTorch, яка відповідає
драйверу. Поточний прогін виконано з PyTorch `2.13.0+cu130`.

Завантаження ресурсів:

```powershell
.\.venv\Scripts\python.exe scripts\download_neruk.py
.\.venv\Scripts\python.exe scripts\download_xlmr.py
```

NER-UK 2.0 поширюється авторами за CC BY-NC-SA 4.0. Репозиторій:
<https://github.com/lang-uk/ner-uk>.

## Запуск

Спочатку швидка автономна перевірка:

```powershell
.\.venv\Scripts\python.exe -m mention_graph.cli run `
  --config configs\synthetic_smoke.json
```

Пілот на NER-UK без трансформера:

```powershell
.\.venv\Scripts\python.exe -m mention_graph.cli run `
  --config configs\neruk_hashing_pilot.json
```

Основний експеримент:

```powershell
.\.venv\Scripts\python.exe -m mention_graph.cli run `
  --config configs\neruk_xlmr.json
```

Relation-aware follow-up E6–E10:

```powershell
.\.venv\Scripts\python.exe -m mention_graph.cli run-followup `
  --config configs\neruk_xlmr_followup.json
```

Швидка перевірка follow-up pipeline без трансформера:

```powershell
.\.venv\Scripts\python.exe -m mention_graph.cli run-followup `
  --config configs\neruk_hashing_followup_smoke.json
```

Перевірка складу даних без навчання:

```powershell
.\.venv\Scripts\python.exe -m mention_graph.cli inspect-data `
  --config configs\neruk_xlmr.json
```

Тести:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Основні артефакти одного прогону

- `table_dataset.csv` - склад train/validation/test;
- `table_e1.csv` ... `table_e5.csv` - таблиці для статті;
- `metrics_by_seed.csv` і `metrics_detailed.json` - усі метрики;
- `calibration_by_seed.csv` - NLL, Brier score та ECE;
- `paired_bootstrap.json` - парні документні bootstrap-порівняння;
- `confusion_matrices/` - матриці помилок за моделями й seed;
- `qualitative_examples.json` - чотири категорії характерних випадків;
- `graph_diagnostics.json` - кількість ребер та ізольованих вузлів;
- `seed-*/test_predictions.npz` - індивідуальні ймовірності згадок;
- `run_manifest.json` — версії, GPU, конфігурація і контрольні суми.

Follow-up додатково створює:

- `table_f1_alias_selection.csv` — validation-only вибір alias-порога;
- `table_f2_edge_signal.csv` — coverage, label purity, Wilson CI та
  relation-specific permutation null;
- `table_f3_model_comparison.csv` — feature/architecture-matched arms;
- `table_f4_subgroups.csv` і `table_f5_correction_harm.csv` — умовні
  ефекти, виправлення та шкода;
- `table_f6_counterfactual.csv` — same-checkpoint edge interventions;
- `table_f7b_nested_bootstrap.csv` — crossed seed×document bootstrap;
- `table_f9_control_strength.csv` — фактична сила random/type-shuffle
  controls.

## Важлива межа інтерпретації

Це класифікація типу за правильних меж згадок, а не повний end-to-end NER.
XLM-R у зафіксованому основному протоколі заморожена; навчаються локальна
голова та GraphSAGE. Наскрізне донавчання кодувальника має бути окремою
конфігурацією, бо воно змінює і обчислювальну вартість, і предмет порівняння.

## Діагностичні експерименти E11–E15

Окремий pipeline перевіряє, чи попередній слабкий graph effect пояснюється
невдалою задачею, поганими ребрами або архітектурою:

- sparse gold-label oracle як явно недеплойна верхня межа;
- exact-repeat як silver identity relation;
- target surface-only та train-prior masking із clean repeat peer;
- whole-component masking як causal negative control;
- reduced-supervision curve на 10%, 25%, 50% і 100% train-документів;
- train-only lexical ambiguity та validation-derived uncertainty cohorts;
- SimpleProp, no-edge і degree-preserving-random controls;
- 10 optimization seeds, 5 target realizations і crossed seed×document
  bootstrap.

Повний запуск:

```powershell
.\.venv\Scripts\python.exe -m mention_graph.cli run-diagnostics `
  --config configs\neruk_xlmr_diagnostics.json
```

Швидкий CPU smoke-test:

```powershell
.\.venv\Scripts\python.exe -m mention_graph.cli run-diagnostics `
  --config configs\neruk_hashing_diagnostics_smoke.json
```

Протокол до запуску з технічними поправками:
`DIAGNOSTIC_PROTOCOL_UK.md`.

Підсумковий науковий звіт:
`DECISIVE_DIAGNOSTIC_REPORT_UK.md`.

Основні нові артефакти:

- `table_g1_oracle_clean.csv` — natural task, oracle headroom і SimpleProp;
- `table_g2_target_recovery.csv` — clean/target/component masking;
- `table_g3_low_resource.csv` — reduced-supervision curve;
- `table_g4_ambiguity.csv` — repeated, uncertain і train-ambiguous cohorts;
- `table_g5_counterfactual.csv` — same-checkpoint interventions;
- `table_g6_crossed_bootstrap.csv` — primary contrasts та interactions;
- `table_g7_graph_diagnostics.csv` — параметри, ребра, homophily та
  validation checkpoints;
- `table_g8_control_strength.csv` — фактична сила degree-preserving rewiring;
- `table_g9_decision_summary.csv` — компактна decision matrix усіх 17
  bootstrap-контрастів;
- `cohort_masks.npz` — frozen cohort masks із вирівняними mention IDs;
- `diagnostic_manifest.json` — конфігурація, середовище та checksums.
