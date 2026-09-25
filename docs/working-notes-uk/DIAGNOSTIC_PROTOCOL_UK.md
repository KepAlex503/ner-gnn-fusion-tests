# Діагностичний протокол: графове уточнення типів згадок

**Статус:** preregistration до запуску.  
**Дані:** NER-UK 2.0, відомі межі згадок, незмінний document-level test split.  
**Мета:** розрізнити відсутність корисного реляційного сигналу, невдалі ребра та нездатність архітектури використати корисні ребра.

## 1. Обмеження та інтерпретація

NER-UK не містить entity/coreference IDs. Тому:

- точний повтор нормалізованої поверхневої форми є лише **silver identity**, а не gold entity relation;
- нечіткі alias-ребра є евристикою;
- gold-label oracle використовує NER labels, включно з test labels, і є **явно недеплойним діагностичним ceiling**;
- жодний oracle-result не подається як якість практичної системи.

## 2. Незмінні умови

- Ознаки primary arms: frozen XLM-R span embedding, OOF `p_local` та intrinsic mention features. Degree і surface-frequency features не використовуються.
- Fair control `N0`: та сама residual head, глибина, hidden size, dropout, optimizer і validation criterion, але без neighbor messages.
- Optimization seeds фіксуються наперед:
  `17, 29, 43, 59, 71, 89, 101, 113, 127, 149`.
- Hyperparameters і всі thresholds вибираються лише на train/validation. Test використовується один раз після фіксації protocol.
- Primary metric для повного test set: macro-F1. Accuracy, micro-F1, NLL, Brier та ECE є secondary.

## 3. Графи та predictors

### 3.1 Графи

- `E0`: без ребер.
- `E_repeat`: exact normalized-surface repeats; silver identity.
- `E_real`: наперед зафіксований deployable typed graph із `sent`, `repeat`, `near` та validation-selected `alias`.
- `E_random`: degree-preserving within-document rewiring `E_real`.
- `E_type_shuffle`: та сама union topology і relation-signature counts, але type signatures переставлені в межах документа.
- `E_gold`: within-document same-gold-label edges. Граф максимально узгоджується з `E_real` за per-document edge count і degree sequence; невідтворений degree mass звітується. Цей граф недеплойний.

### 3.2 Predictors

- `N0`: fair node-only model.
- `SimpleProp`: validation-tuned convex average власних і mean-neighbor local probabilities; однакова наперед задана сітка ваг для всіх graph regimes.
- `GNN`: поточна gated relational architecture.
- `GoldVote`: leave-one-out majority gold label сусідів у `E_gold`; для вузла без сусідів використовується `N0`. Це лише абсолютна coverage upper bound.

## 4. Primary contrast families

### F1. Diagnostic opportunity

1. `GoldVote(E_gold) - N0`: максимальна теоретична можливість.
2. `SimpleProp(E_gold) - N0`: чи доступний сигнал простому propagation.
3. `GNN(E_gold) - N0`: чи може GNN використати ідеально homophilous edges.
4. Non-inferiority `GNN(E_gold)` до `SimpleProp(E_gold)` із margin `0.01` macro-F1.

### F2. Silver/deployable edges

1. `GNN(E_repeat) - N0`.
2. `GNN(E_real) - N0`.
3. `GNN(E_real) - GNN(E_random)`.
4. `GNN(E_real) - GNN(E_type_shuffle)`.
5. `SimpleProp(E_repeat) - GNN(E_repeat)` як architecture diagnostic на тих самих silver edges.

### F3. Stress interactions

1. Target-context masking interaction, визначений у розділі 5.
2. Monotonic corruption trend, визначений у розділі 6.
3. Low-resource interaction, визначений у розділі 7.

### F4. Pre-specified subgroup interactions

1. `[GNN(E_repeat)-N0]repeated - [GNN(E_repeat)-N0]singleton`.
2. `[GNN(E_real)-N0]ambiguous - [GNN(E_real)-N0]non-ambiguous`.
3. `[GNN(E_repeat)-N0]uncertain - [GNN(E_repeat)-N0]confident`.

## 5. Controlled target-context masking

У кожному exact-repeat component розміру `>=2` одна target mention вибирається мінімальним SHA-256 hash її `mention_id`. Вибір не використовує labels або predictions.

- `target-mask`: у encoder window target mention зберігається, а символи токенів поза її span замінюються пробілами; encoder фактично отримує surface-only sequence. Інші mentions компонента мають повний контекст.
- `component-mask`: маскується контекст target і всіх mentions того самого exact-repeat component.
- `p_local` та embeddings завжди перераховуються з відповідного masked input; clean probabilities не потрапляють у features.
- На train/validation застосовується той самий deterministic target-mask protocol. Один і той самий checkpoint на test оцінюється у `target-mask` і `component-mask`; scoring виконується лише на заздалегідь вибраних targets.

Primary interaction:

`[GNN(E_repeat)-N0]target-mask - [GNN(E_repeat)-N0]component-mask`.

Позитивний ефект означає використання незамаскованого контексту сусідів. Якщо gain зберігається після whole-component masking, contextual-borrowing explanation не підтримується; це negative-control failure або інший механізм.

## 6. Controlled edge corruption

Для `E_repeat`, `E_real` та `E_gold` використовуються рівні
`rho = 0, 0.25, 0.50, 0.75, 1.00`.

Corruption виконується within-document double-edge swaps без self-loops і duplicate edges. Зберігаються edge count і degree sequence; для typed graph додатково зберігаються relation counts. Для кожного `optimization seed × rho` генеруються п'ять незалежних realizations.

Primary test — знак slope macro-F1 від фактично виміряної label homophily; додатково перевіряється `rho=0 - rho=1`. Відсутність degradation для `E_gold` означає, що predictor не використовує правильність сусідів.

## 7. Low-resource protocol

Train fractions фіксуються як `0.10, 0.25, 0.50, 1.00`. Для кожного optimization seed створюються п'ять nested document-sampling chains зі стратифікацією за source і class coverage. Validation і test не змінюються. Local model, OOF predictions та graph heads навчаються заново на кожній fraction.

Primary interaction:

`mean graph gain at {0.10, 0.25} - graph gain at 1.00`,

окремо для `E_repeat` та `E_real`. Повна fraction curve є secondary.

## 8. Ambiguous та repeated subsets

- `repeated`: within-document exact normalized frequency `>=2`.
- `singleton`: frequency `=1`.
- `train-ambiguous`: surface forms із full-train support `>=5` та train label entropy `>=0.5 bits`. Список заморожується до test evaluation і не використовується як model feature.
- `uncertain`: predictive entropy `N0`, вища за threshold, що дорівнює 75-му percentile entropy на validation. Threshold визначається окремо для кожної resource fraction за mean validation probabilities десяти `N0` seeds і потім застосовується до test.
- Test-gold conflicting surface groups є лише exploratory і не входять у confirmatory inference.

Якщо subgroup має менше 200 mentions або 10 документів, подаються лише counts та descriptive estimates.

## 9. Статистичний аналіз і multiplicity

- Point estimate: середня paired різниця за optimization seeds.
- 20,000 crossed bootstrap replicates: optimization seeds і test documents resample незалежно з поверненням; documents стратифікуються за `bruk/ng`.
- Corruption realizations і low-resource sampling chains resample як вкладені в optimization seed.
- Для macro-F1 у кожній replicate агрегуються document confusion matrices і метрика перераховується, а mentions не resample незалежно.
- Ті самі bootstrap draws використовуються для всіх arms одного contrast.
- Holm correction застосовується окремо в F1-F4. Інші subgroup, class і intervention analyses мають BH-FDR `q=0.05` і позначаються exploratory.
- Не робляться окремі висновки про «значущі seeds».

Decision thresholds:

- global SESOI: `0.01` macro-F1;
- subgroup SESOI: `0.005` accuracy;
- supported benefit: Holm-adjusted `p<0.05`, 95% CI lower bound `>0` і point estimate не менше SESOI;
- equivalent/no meaningful effect: multiplicity-adjusted TOST і 90% CI повністю всередині `[-SESOI,+SESOI]`;
- усі інші випадки: **inconclusive**.

## 10. Primary decision matrix

| Наперед визначений патерн | Рішення |
|---|---|
| `SimpleProp(E_gold)` допомагає, але `GNN(E_gold)` не є non-inferior | bad architecture/training |
| Навіть `SimpleProp(E_gold)` еквівалентний нулю, включно з masking/low-resource | немає доступного relational headroom або помилки сусідів повністю корельовані |
| `E_gold` допомагає, але `E_repeat` еквівалентний нулю | task-relation mismatch: surface identity не дає потрібного type signal |
| `E_repeat` допомагає, але `E_real` не допомагає або програє random | bad edge composition/induction |
| `E_real` не кращий за random або type-shuffle | generic smoothing/regularization, а не semantic-edge effect |
| Full-context gain відсутній, але target-mask або low-resource interaction позитивний | conditional rescue under weak local evidence |
| Target-mask gain зникає при component-mask | підтверджено borrowing контексту від сусідів |
| Target-mask gain не зникає при component-mask | negative control не пройдено; contextual-borrowing claim відхиляється |
| `E_real > N0`, `E_real > random/type-shuffle`, corruption дає monotonic degradation | meaningful deployable graph effect підтримано |

Ця матриця використовується без post-hoc зміни thresholds, моделей, subsets або primary contrasts.

## 11. Попередньо зафіксована технічна поправка до запуску

Цей розділ додано після CPU smoke-test, але **до першого запуску на test із XLM-R**.

- Основний gold-label oracle реалізується як sparse within-document граф: кожна згадка з'єднується з найближчими згадками того самого gold-класу на відстанях 1 і 2 у впорядкованій групі. Максимальний degree дорівнює 4. Це зменшує різницю в щільності з практичними графами; oracle залишається явно недеплойним.
- Усі однореляційні arms (`repeat`, semantic union, pruned union, sparse oracle) подають ребра через той самий активний channel. Degree-preserving randomization використовується як same-checkpoint intervention.
- Повна крива часткового edge corruption `rho` не входить до поточного confirmatory run. Фіксуються крайні інтервенції `full`, `no_edges` та повний degree-preserving rewire; проміжна крива залишається окремим розширенням.
- Low-resource curve використовує одну наперед зафіксовану source-stratified nested document chain. Тому цей блок є conditional reduced-supervision proxy, а не оцінкою невизначеності за всіма можливими train-subsamples.
- `target-surface` та `target-prior` мають окремі adapted graph checkpoints. Для causal negative control кожен із цих checkpoint без перенавчання оцінюється також при masking усієї repeat-компоненти.
- `G_semantic_union` і gold-pruned semantic union додаються у full-supervision clean block. Попередній follow-up уже окремо перевірив typed, random і type-shuffle graphs.
- Uncertainty threshold обчислюється один раз з entropy середніх `N0` validation probabilities за всіма optimization seeds; та сама frozen cohort mask використовується для кожного seed на test.
- Додатково фіксуються crossed-bootstrap subgroup interactions: repeated проти singleton та uncertain проти confident; strict train-ambiguous спочатку перевіряється за мінімальним support.
- Формальний multiplicity-adjusted TOST у цьому запуску не виконується. 90% CI всередині SESOI bounds позначається лише як descriptive; без TOST рішення залишається `inconclusive`.
- Bootstrap p-value обчислюється з нуль-центрованого розподілу `theta* - theta_hat`; окремо зберігається descriptive sign probability.
- Exact frozen cohort masks і вирівняні mention IDs зберігаються в `cohort_masks.npz`, щоб ensemble-derived uncertainty cohort можна було перевірити без повторного inference.

Після label-free підрахунку support перед запуском strict `train-ambiguous` cohort має менше 200 test mentions. Відповідно до розділу 8 її interaction вилучається з confirmatory family і подається лише descriptively. Явний comparator у таблиці — `train-seen-unambiguous`; unseen surfaces не включаються до нього. Confirmatory subgroup family містить лише repeated-vs-singleton та uncertain-vs-confident.
