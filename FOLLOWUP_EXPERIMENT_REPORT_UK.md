# Додатковий звіт: причинна перевірка внеску графових ребер

## Короткий висновок

Додаткові експерименти E6–E10 виправляють головний недолік першого
протоколу: тепер локальну нелінійну голову, скалярні структурні ознаки,
message passing, розташування ребер і назви типів ребер перевірено
окремими контролями.

Результат є змішаним, але змістовним:

- node-only reclassifier підвищує macro-F1 із `0.7191` до `0.8144`;
- найвище спостережене середнє має `G_alias`: `0.8263 ± 0.0052`;
- різниця `G_alias − N0` дорівнює `+0.0119`, але це exploratory ranking
  серед багатьох arms, а не підтверджений причинний ефект alias-ребер;
- жоден із семи наперед визначених основних контрастів не пройшов
  crossed seed×document bootstrap із Holm-корекцією;
- same-checkpoint interventions показують невеликий прямий ефект
  повідомлень: приблизно `+0.004–0.006` macro-F1;
- repeat-ребра мають `99.70%` узгодження gold-типів і дають найбільший
  цільовий ефект саме для повторних згадок;
- правильні relation labels не кращі за shuffled labels на незмінній
  топології;
- alias-ребра перспективні, але покривають лише `3.25%` test-згадок, а їх
  безпосередній same-checkpoint внесок дуже малий.

Отже, чесний headline результату:

> Repeat та alias relations мають сильну узгодженість типів і невеликі
> цільові переваги, але розширені контроли не встановлюють стійкої
> загальної переваги змістовної топології або правильних назв типів ребер.

## Що змінено відносно E1–E5

### Однакові первинні ознаки

Усі primary arms отримують той самий `X_base`:

1. заморожене XLM-R span representation;
2. OOF `p_local` для навчальних вузлів;
3. topology-free ознаки: довжина згадки, позиція речення і регістр.

Частота форми та degrees не входять у primary comparison. Вони додані
лише в окремий `N1_node_stats`.

### Чесний no-edge control

`N0_node_only` — той самий residual relation-aware classifier, але всі
adjacency matrices порожні. Усі base arms мають однакову кількість
задекларованих параметрів: `185 619`.

Водночас кількість реально активних relation branches різна:

- `N0`: 0;
- untyped/repeat: 1;
- context/alias-repeat: 2;
- typed/type-shuffle/typed-random: 3;
- alias-all: 4.

Тому порівняння моделей із різною кількістю активних branches не
трактується як чистий причинний ефект ребер. Найнадійніші контрасти:

- `G_untyped` проти `G_untyped_random`;
- `G_typed` проти `G_typed_random`;
- `G_typed` проти `G_type_shuffle`;
- test-time interventions того самого checkpoint.

### Нові controls

- `G_untyped_random`: degree-preserving rewiring union-графа;
- `G_typed_random`: окреме degree-preserving rewiring кожного relation;
- `G_type_shuffle`: незмінний union, але relation signatures переставлені;
- `no_edges`: усі повідомлення вимкнено в уже навченому checkpoint;
- `drop_relation`: по одному вимкнено `sent`, `repeat`, `near`, `alias`.

Random interventions повторено для п’яти topology seeds усередині кожного
з п’яти optimization seeds.

### Статистика

- optimization seeds: `17`, `29`, `43`, `59`, `71`;
- test: 169 документів, 6 931 згадка;
- primary metric: macro-F1;
- 10 000 crossed bootstrap samples;
- в кожній повторній вибірці документи ресемплюються зі стратифікацією
  `bruk/ng`, а optimization seeds — окремим crossed фактором;
- сім primary contrasts мають Holm-корекцію;
- subgroup і correction/harm аналізи позначено exploratory.

## E6. Чи несуть ребра потенційний сигнал?

Null expectation обчислено окремо для кожного relation:

\[
\frac{
  \sum_d |E_{r,d}| P_d(y_u=y_v)
}{
  \sum_d |E_{r,d}|
}.
\]

Це очікувана частка однакових типів після перестановки labels у межах
кожного документа, зважена кількістю ребер relation.

| Relation | Ребра | Покриття | Same-label rate, 95% Wilson CI | Null | Lift |
|---|---:|---:|---:|---:|---:|
| `sent` | 10 442 | 80.94% | 0.3609 [0.3517; 0.3701] | 0.2741 | +0.0867 |
| `repeat` | 5 008 | 34.41% | 0.9970 [0.9951; 0.9982] | 0.6364 | +0.3606 |
| `near` | 32 728 | 98.17% | 0.3165 [0.3115; 0.3216] | 0.2785 | +0.0381 |
| `alias` | 121 | 3.25% | 0.9752 [0.9296; 0.9915] | 0.4257 | +0.5495 |
| semantic union | 37 336 | 99.11% | 0.3962 [0.3913; 0.4012] | 0.3231 | +0.0732 |
| union з alias | 37 396 | 99.13% | 0.3972 [0.3922; 0.4021] | 0.3233 | +0.0739 |

Усі relations мають позитивний descriptive lift. Найчистішим є exact
repeat. `sent` і `near` не є homophilic relations у простому сенсі:
вони часто зв'язують `JOB–PERS`, `ORG–PERS`, `DATE–ORG` тощо. Тому їх не
слід обробляти як звичайне згладжування однакових labels.

## E7. Feature- і architecture-matched порівняння

Середнє ± SD за п'ятьма seeds:

| Модель | Активні branches | Accuracy | Macro-F1 | Δ до N0 |
|---|---:|---:|---:|---:|
| Local | — | 0.8914 | 0.7191 ± 0.0242 | −0.0953 |
| `N0_node_only` | 0 | 0.9131 | 0.8144 ± 0.0098 | 0 |
| `N1_node_stats` | 0 | 0.9083 | 0.8099 ± 0.0106 | −0.0045 |
| `G_untyped_random` | 1 | 0.9092 | 0.8091 ± 0.0158 | −0.0053 |
| `G_untyped` | 1 | 0.9100 | 0.8103 ± 0.0147 | −0.0041 |
| `G_repeat` | 1 | 0.9165 | 0.8196 ± 0.0059 | +0.0052 |
| `G_context` | 2 | 0.9152 | 0.8218 ± 0.0024 | +0.0074 |
| `G_alias_repeat` | 2 | 0.9187 | 0.8246 ± 0.0057 | +0.0102 |
| `G_typed_random` | 3 | 0.9131 | 0.8161 ± 0.0176 | +0.0017 |
| `G_type_shuffle` | 3 | 0.9155 | 0.8218 ± 0.0088 | +0.0074 |
| `G_typed` | 3 | 0.9159 | 0.8207 ± 0.0033 | +0.0063 |
| `G_alias` | 4 | 0.9185 | **0.8263 ± 0.0052** | **+0.0119** |

### Що пояснює попередній великий приріст

На тих самих п'яти seeds:

- `local → N0_node_only`: `+0.0953` macro-F1;
- `N0_node_only → highest observed mean`: `+0.0119`;
- direct same-checkpoint message effect: приблизно `+0.004–0.006`.

Отже, основна частина попереднього `local → graph` приросту походить від
нелінійного residual reclassifier, OOF probabilities і оптимізації голови,
а не від adjacency.

`N1_node_stats` не покращив `N0`, тому прості frequency/degree features
також не пояснюють цей приріст.

### Primary crossed bootstrap

| Контраст | Mean Δ macro-F1 | 95% CI | Holm p | Висновок |
|---|---:|---:|---:|---|
| `G_untyped − N0` | −0.0041 | [−0.0174; 0.0065] | 1.000 | не підтримано |
| `G_untyped − G_untyped_random` | +0.0012 | [−0.0049; 0.0068] | 1.000 | не підтримано |
| `G_typed − G_untyped` | +0.0104 | [−0.0014; 0.0256] | 0.629 | suggestive |
| `G_typed − G_type_shuffle` | −0.0010 | [−0.0127; 0.0097] | 1.000 | не підтримано |
| `G_typed − G_typed_random` | +0.0046 | [−0.0097; 0.0225] | 1.000 | не підтримано |
| `G_repeat − N0` | +0.0052 | [−0.0041; 0.0156] | 1.000 | не підтримано глобально |
| `G_alias − G_typed` | +0.0056 | [−0.0004; 0.0124] | 0.476 | позитивний trend |

Жоден primary contrast не є confirmatory significant.

Exploratory `G_alias_repeat − N0` має `+0.0102`,
95% CI `[0.0023; 0.0200]`, Holm `p=0.0416` у своїй exploratory family.
Але цей контраст активує дві relation branches замість нуля, тому не є
чистим доказом причинної користі alias/repeat messages.

## E8. Де саме виникає ефект?

### Повторні згадки

Для 2 385 згадок із частотою форми `2+`:

| Модель | Macro-F1 | Accuracy |
|---|---:|---:|
| `N0_node_only` | 0.7708 | 0.9189 |
| `G_repeat` | 0.7854 | 0.9270 |
| `G_alias_repeat` | 0.7911 | 0.9295 |

`G_repeat − N0`:

- macro-F1: `+0.0146`;
- accuracy: `+0.0081`;
- виправлено в середньому 14.24% помилок baseline;
- harm rate серед правильних baseline-прогнозів: 0.37%.

Для singleton-згадок той самий contrast дає лише:

- macro-F1: `+0.0033`;
- accuracy: `+0.0008`.

Це підтримує локальний механізм повторів набагато краще, ніж стара E4,
яка порівнювала `graph_all` із `local` і змішувала head та edge effects.
Subgroup interaction формально не тестувався, тому результат exploratory.

### Alias-connected згадки

На 225 alias-connected згадках:

| Модель | Macro-F1 | Accuracy |
|---|---:|---:|
| `N0_node_only` | 0.7264 | 0.9156 |
| `G_typed` | 0.7459 | 0.9253 |
| `G_alias` | 0.7567 | 0.9369 |
| `G_alias_repeat` | 0.7788 | 0.9404 |

`G_alias − N0` має `+0.0303` macro-F1 і `+0.0213` accuracy в цій
підгрупі. Проте підгрупа мала, модель має додаткову активну relation
branch, а аналіз є exploratory.

## E9. Same-checkpoint interventions

Це найчистіша перевірка використання messages: weights та node features
не змінюються, змінюється лише test adjacency.

| Навчена модель | Intervention | Full − intervention macro-F1 |
|---|---|---:|
| `G_untyped` | без усіх ребер | +0.0059 ± 0.0042 |
| `G_untyped` | untyped random, 5 permutations | +0.0040 ± 0.0023 |
| `G_typed` | без усіх ребер | +0.0037 ± 0.0039 |
| `G_typed` | typed random, 5 permutations | +0.0011 ± 0.0039 |
| `G_typed` | shuffled types, 5 permutations | −0.0020 ± 0.0015 |
| `G_typed` | без `repeat` | +0.0019 ± 0.0008 |
| `G_typed` | без `near` | +0.0071 ± 0.0041 |
| `G_typed` | без `sent` | +0.0005 ± 0.0046 |
| `G_alias` | без `alias` | +0.0005 ± 0.0004 |

Що це означає:

1. Модель справді використовує messages, але прямий ефект невеликий.
2. Exact-repeat messages дають малий, дуже послідовний позитивний внесок.
3. `near` найбільше впливає на macro-F1, але майже не покращує accuracy;
   імовірно, це ефект на рідкісних класах.
4. Правильні relation labels не дали переваги над shuffled signatures.
5. Безпосередній внесок alias messages (`+0.0005`) значно менший за
   retrained різницю `G_alias − G_typed` (`+0.0056`). Отже останню не можна
   повністю приписувати alias-ребрам.

Для ізольованих вузлів probabilities до і після видалення adjacency
збігаються; це перевіряється автоматичним assertion.

## E10. Розширені alias-ребра

### Правило

Без використання gold label під час побудови ребра:

- строгі абревіатури;
- однакова кількість токенів;
- велика літера;
- відсутність numeric/date-like форм;
- SequenceMatcher similarity;
- один найближчий bridge між surface groups;
- заборона overlap spans і cap на кількість alias-сусідів.

Threshold вибрано лише на validation:

| Threshold | Validation edges | Coverage | Same-label rate |
|---:|---:|---:|---:|
| 0.88 | 53 | 4.24% | 0.9811 |
| **0.90** | **44** | **3.67%** | **1.0000** |
| 0.92 | 26 | 2.30% | 1.0000 |
| 0.94 | 21 | 1.86% | 1.0000 |
| 0.96 | 8 | 0.71% | 1.0000 |

На test після заморожування threshold:

- 121 alias-ребро;
- 225 згадок, або 3.25%;
- label agreement: `118/121 = 97.52%`;
- 95% Wilson CI: `[92.96%; 99.15%]`;
- 140 вузлів раніше не мали exact-repeat;
- 60 ребер не належать жодному з `sent/repeat/near`.

`97.52%` потрібно називати **label agreement/purity**, а не coreference
precision. Однаковий NER-тип не гарантує того самого референта.

Серед трьох gold-label disagreements:

- `Парус [ORG] ↔ Парусі [ART]` схоже на annotation inconsistency;
- `Віснику державних закупівель [ART] ↔ ВДЗ [ORG]` також схоже на
  annotation inconsistency;
- `Молодняк [MISC] ↔ Молодняка [ORG]` є реальною омонімічною помилкою.

Є й same-label, але non-coreferent випадки, наприклад
`Павлов ↔ Павлова`. Для твердження про справжню coreference precision
потрібна окрема ручна розмітка всіх 121 пар.

## Сила контрольних перетворень

Controls не є косметичними:

- untyped random змінює в середньому 68.0% union-ребер;
- typed random змінює 78.6% `sent`, 56.3% `repeat`, 70.3% `near`;
- type shuffle змінює приблизно 66.1% `sent` signatures,
  49.2% `repeat` і 7.2% `near`;
- type shuffle зберігає union topology точно.

Малий відсоток змін `near` signatures пояснюється тим, що `near`
домінує в union і часто є єдиною signature пари.

## Оновлені відповіді на дослідницькі питання

### Чи є сенс у графовому уточненні?

Так, але ефект значно менший, ніж показувало порівняння з простою
локальною головою. Same-checkpoint оцінка вказує приблизно на
`+0.004–0.006` macro-F1.

### Чи кращі змістовні ребра за випадкові?

Не підтверджено на рівні retrained primary contrasts. Same-checkpoint
untyped comparison показує невелику descriptive перевагу змістовного
union над random (`+0.0040`), але цього недостатньо для сильного
загального твердження.

### Чи потрібні правильні назви типів ребер?

Наразі ні: `G_type_shuffle` не гірша за `G_typed`, а test-time shuffle
навіть має трохи вище macro-F1.

### Чи корисні повтори?

Так, це найкраще локалізований механізм. Repeat edges майже ідеально
узгоджені за типом, дають більший приріст на `2+` згадках і послідовно
погіршують результат після їх видалення з того самого checkpoint.

### Чи корисні alias-ребра?

Вони перспективні, але доказ поки слабкий. `G_alias` має найвище
спостережене середнє, проте primary CI включає нуль, coverage мале, а
same-checkpoint `drop_alias` ефект становить лише близько `0.0005`.

## Як змінити формулювання статті

Не варто стверджувати:

> GraphSAGE з усіма змістовними ребрами дає близько +10 пунктів
> macro-F1 завдяки графовому поширенню інформації.

Підтримуване формулювання:

> Документний residual reclassifier із локальними OOF-ймовірностями
> суттєво покращує типізацію відомих згадок. Message passing дає
> додатковий малий ефект. Найчіткіше цей ефект локалізується на
> високоточних repeat-зв'язках, тоді як загальна перевага змістовної
> топології та правильних relation labels не підтверджена.

Це є сильнішим науковим результатом, оскільки він:

- відділяє representational gain від edge effect;
- містить topology- і type-matched random controls;
- показує як виправлення, так і шкоду;
- не приховує негативні абляції;
- пропонує конкретний механізм, який справді має сигнал.

## Обмеження

1. Використовуються gold mention spans, а не повний end-to-end NER.
2. XLM-R заморожена.
3. П'ять optimization seeds краще за три, але не замінюють ширшої
   перевірки на іншому корпусі.
4. Highest observed mean серед багатьох arms має winner's curse.
5. Subgroup і correction/harm таблиці exploratory та не мають окремої
   multiplicity correction.
6. Same-label alias purity не є coreference precision.
7. `sent` і `near` потребують моделі heterophilous relations, а не лише
   relation-separated mean aggregation.

## Рекомендований наступний крок

Найдоцільніший наступний експеримент — не ще більша GNN, а простіша
цільова модель repeat-компонентів:

1. confidence-weighted product/CRF для equality constraint;
2. порівняння з `G_repeat` на тих самих згадках `2+`;
3. ручна `same_referent` розмітка 121 alias-пари;
4. лише після цього — gated combination repeat/alias із contextual
   heterophilous relations.

## Артефакти

- `artifacts/neruk-xlmr-followup/table_f1_alias_selection.csv`;
- `table_f2_edge_signal.csv`;
- `table_f3_model_comparison.csv`;
- `table_f4_subgroups.csv`;
- `table_f5_correction_harm.csv`;
- `table_f6_counterfactual.csv`;
- `table_f7b_nested_bootstrap.csv`;
- `table_f9_control_strength.csv`;
- `nested_seed_document_bootstrap.json`;
- `alias_validation_examples.json`;
- `alias_test_examples.json`;
- `seed-*/test_predictions.npz`;
- `seed-*/*_model.pt`.

Контрольні суми фінального запуску:

- config SHA-256:
  `7220f6a4b915732343c03236c41d2e6c2a252b407ad07b44f77395806e007478`;
- source SHA-256:
  `ece9c26eb13e9be3fb6d0c6e00afae41483c42fa05395df5cf319ab7c18d990e`.
