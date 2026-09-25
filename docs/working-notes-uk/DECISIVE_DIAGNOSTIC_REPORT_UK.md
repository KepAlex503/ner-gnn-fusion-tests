# Додатковий діагностичний звіт E11–E15: коли граф справді допомагає NER

**Дата прогону:** 27 липня 2026 року  
**Дані:** NER-UK 2.0, незмінний document-level test split  
**Статус:** повний XLM-R/CUDA-прогін завершено; 200 моделей, 10 optimization seeds, 20 000 crossed-bootstrap повторів  
**Протокол:** `DIAGNOSTIC_PROTOCOL_UK.md`

## Коротка відповідь

Граф не є марним. Але стандартна оцінка на чистому NER майже не перевіряє його
сильну сторону.

На звичайному test set exact-repeat граф дає лише `+0.0041` macro-F1 над
архітектурно узгодженим node-only baseline. Довірчий інтервал перетинає нуль,
а ефект менший за наперед заданий practically meaningful threshold `0.01`.
Тому твердження «граф покращує clean NER загалом» не підтверджено.

Водночас, коли зовнішній контекст лише цільової згадки контрольовано прибрано,
але clean-контекст її exact-repeat сусідів збережено, граф підвищує accuracy
на `+0.0879`; 95% CI `[0.0639; 0.1131]`, Holm-adjusted
`p = 0.00025`. Якщо прибрати контекст усієї repeat-компоненти, ця перевага
не зберігається. Target-versus-component interaction дорівнює `+0.2469`;
95% CI `[0.2037; 0.2918]`, Holm-adjusted `p = 0.00025`.

Отже, найточніший висновок такий:

> Графове уточнення забезпечує відтворюване cross-mention recovery, коли
> локальний доказ цільової згадки слабкий, а пов'язана згадка має якісний
> контекст. Його загальний clean-ефект обмежують сильний node-only baseline,
> неповне покриття repeat-зв'язками та шум ширших semantic edges.

| Наукове твердження | Результат |
|---|---|
| Граф покращує звичайний clean NER | Не підтверджено |
| Поточна GNN здатна використати інформативні ребра | Підтверджено |
| Repeat-граф допомагає за слабкого локального контексту target | Сильно підтверджено |
| Саме складна GNN краща за просте усереднення сусідів | Не підтверджено |
| Зменшення обсягу train data саме по собі посилює користь графа | Не підтверджено |
| Якість складу ребер є bottleneck | Підтверджено oracle-pruning контролем |

## 1. Що саме було змінено

Попередні експерименти переважно відповідали на запитання, чи покращує граф
середню якість на повному clean test set. Новий блок розділяє чотири можливі
причини малого результату:

1. у даних немає relational headroom;
2. корисний сигнал є, але граф побудований із невдалих ребер;
3. ребра інформативні, але архітектура не вміє ними скористатися;
4. граф корисний лише за слабкого локального evidence, а clean aggregate
   приховує цей режим.

Для цього додано:

- sparse gold-label oracle і gold-pruned semantic union;
- просте validation-tuned усереднення й GoldVote ceilings;
- target-surface і target-prior recovery tasks;
- whole-component negative controls із тим самим checkpoint;
- same-checkpoint `no_edges` і degree-preserving rewiring;
- low-resource fractions `0.10`, `0.25`, `0.50`, `1.00`;
- repeated/singleton, uncertain/confident і train-only ambiguity cohorts;
- 20 000 crossed bootstrap samples із незалежним resampling optimization
  seeds та source-stratified test documents.

Усі learned node-only і graph heads мають однакові `185 747` параметрів.
Однореляційні arms використовують той самий активний relation channel.
Clean primary metric — macro-F1; target-recovery primary metric — accuracy.
SESOI зафіксовано як `0.01` macro-F1 і `0.005` accuracy.

## 2. Дані та контрольовані target-задачі

Test split містить 169 документів, 6 931 згадку та всі 13 класів.
Exact normalized surface repeats охоплюють 2 385 згадок у 800
within-document групах. У кожній групі label-free SHA-256 правилом вибиралася
одна target-згадка; використано п'ять наперед заданих реалізацій вибору.

`target_surface` зберігає текст самої згадки, але замінює пробілами зовнішній
контекст у її encoder window. Це не `[MASK]`-corruption. Інші згадки
repeat-компоненти залишаються clean.

`component_surface` оцінює той самий target-surface checkpoint без
перенавчання, але прибирає зовнішній контекст у всіх згадок тієї самої
repeat-компоненти. Тому різниця між target- і component-режимами ізолює
наявність clean neighbor evidence.

`target_prior` є суворішим stress test: embedding target зануляється, а
`p_local` замінюється train-only class prior. Він використовується як
механістична діагностика, а не як deployment estimate.

## 3. E11 — чи існує relational headroom

| Модель на clean test | Macro-F1, mean ± SD | Δ до `N0` |
|---|---:|---:|
| `N0_node_only` | `0.8179 ± 0.0124` | — |
| `AVG_label_sparse` | `0.8514 ± 0.0235` | `+0.0335` |
| `O_label_sparse` | `0.8660 ± 0.0338` | `+0.0481` |
| `GoldVote_label_sparse` | `0.9669 ± 0.0033` | `+0.1490` |

Confirmatory contrasts:

| Контраст | Estimate | 95% CI | Holm p | Рішення |
|---|---:|---:|---:|---|
| `O_label_sparse − N0` | `+0.0481` | `[0.0266; 0.0714]` | `0.00020` | supported benefit |
| `AVG_label_sparse − N0` | `+0.0335` | `[0.0211; 0.0484]` | `0.00040` | supported benefit |
| `GoldVote_label_sparse − N0` | `+0.1490` | `[0.1242; 0.1749]` | `0.00020` | non-deployable ceiling |

Це відкидає пояснення «у задачі взагалі немає relational signal». Навіть
простий predictor використовує ідеальні зв'язки, а поточна GNN також отримує
значний приріст.

Однак non-inferiority `O_label_sparse` відносно `AVG_label_sparse` не
встановлено: різниця `+0.0146`, 95% CI `[-0.0113; 0.0419]`; нижня межа
трохи переходить наперед заданий margin `−0.01`. Отже, не можна стверджувати,
що learned GNN стабільно не гірша або краща за просту propagation rule.

Усі label-sparse oracle, GoldVote та gold-pruned результати використовують
test gold labels для побудови графа або голосування. Це діагностичні
non-deployable ceilings, а не якість практичної системи.

## 4. E12 — clean NER і якість ребер

| Модель | Clean macro-F1 | Δ до `N0` |
|---|---:|---:|
| `local` | `0.7143` | `−0.1036` |
| `N0_node_only` | `0.8179` | — |
| `AVG_repeat` | `0.8210` | `+0.0031` |
| `G_repeat` | `0.8220` | `+0.0041` |
| `G_semantic_union` | `0.8119` | `−0.0060` |
| `O_pruned_union` | `0.8430` | `+0.0251` |

Для deployable exact-repeat графа:

`G_repeat − N0 = +0.0041` macro-F1, 95% CI
`[-0.0042; 0.0127]`, Holm-adjusted `p = 0.6513`.

Це inconclusive і нижче SESOI. Великий `local → graph` приріст знову
пояснюється переважно nonlinear residual reclassifier:
`local = 0.7143`, `N0 = 0.8179`, тоді як додавання repeat messages змінює
середній macro-F1 лише на `+0.0041`.

`AVG_repeat − G_repeat = −0.0010`, 95% CI
`[-0.0100; 0.0076]`. 90% CI лежить усередині `±0.01`, але формального
TOST не виконувалося, тому це не доказ еквівалентності. Практично результати
дуже схожі, і перевага складної архітектури над усередненням не встановлена.

Найважливіший edge-quality control:

`O_pruned_union − G_semantic_union = +0.0311` macro-F1,
95% CI `[0.0203; 0.0429]`, Holm-adjusted `p = 0.00020`.

Semantic union містить 37 336 test edges із label homophily `0.3962`.
Після gold-pruning залишається 14 794 edges із homophily `1.0`, і якість
суттєво зростає. Це сильний доказ, що для ширшого графа bottleneck полягає
у складі та точності ребер, а не у принциповій нездатності message passing.
Водночас pruning разом змінює purity, кількість ребер, degree і topology,
тому цей контраст ідентифікує ефект усього складу графа, а не чистий
причинний ефект лише homophily.

## 5. E13 — контрольоване відновлення слабкої згадки

### 5.1 Абсолютні результати

Accuracy рахується лише на 800 наперед вибраних targets і усереднюється за
10 optimization seeds та 5 target realizations.

| Evaluation condition | `N0` | `AVG_repeat` | `G_repeat` | `G_repeat − N0` |
|---|---:|---:|---:|---:|
| clean targets | `0.9034` | `0.9109` | `0.9117` | `+0.0083` |
| target surface-only | `0.8220` | `0.9163` | `0.9099` | `+0.0879` |
| whole component surface-only | `0.8220` | `0.8220` | `0.6630` | `−0.1590` |

Коли target втрачає контекст, node-only accuracy знижується приблизно на
8.1 percentage points, але repeat-граф майже повністю повертає clean-рівень.
Просте усереднення навіть трохи вище за GNN (`0.9163` проти `0.9099`), тому
позитивний механізм не слід подавати як унікальну перевагу GNN.

### 5.2 Confirmatory causal controls

| Контраст | Estimate accuracy | 95% CI | Holm p |
|---|---:|---:|---:|
| `G_repeat − N0` на target-surface | `+0.0879` | `[0.0639; 0.1131]` | `0.00025` |
| `G_repeat(full) − G_repeat(no_edges)`, той самий checkpoint | `+0.3100` | `[0.2588; 0.3629]` | `0.00025` |
| target-vs-component borrowing interaction | `+0.2469` | `[0.2037; 0.2918]` | `0.00025` |
| приріст graph gain: target-surface проти clean targets | `+0.0796` | `[0.0556; 0.1043]` | `0.00025` |
| oracle graph `− N0` на target-surface | `+0.1154` | `[0.0894; 0.1433]` | `0.00025` |

Target-versus-component interaction є головним механістичним результатом.
Це той самий checkpoint і ті самі target nodes; змінюється лише наявність
clean evidence у пов'язаних згадок. Позитивна interaction підтримує
пояснення, що target справді запозичує контекст від сусідів.

Same-checkpoint `no_edges` ablation підтверджує залежність checkpoint від
messages, але його не слід використовувати як самостійний deployment
baseline: checkpoint навчався з edges. Для порівняння практичної якості
справедливішим baseline залишається окремо навчений `N0`.

### 5.3 Топологічний контроль

Degree-preserving within-document rewiring змінює в середньому `56.31%`
repeat edges і зберігає degree кожного node. Водночас homophily знижується
лише з `0.9970` до приблизно `0.8510`. Отже, це консервативний
degree/document-conditioned control, а не label-neutral random null.

На clean test правильний repeat graph перевищує rewired лише на:

- `+0.00044` accuracy;
- `+0.00114` macro-F1.

На target-surface правильна topology перевищує rewired на `+0.16547`
accuracy. Це показує, що значення конкретних зв'язків проявляється саме
в режимі weak-target evidence, хоча навіть rewired control зберігає
високу homophily.

Для oracle graph rewiring сильніший: змінюється приблизно `90.81%` edges,
а homophily падає з `1.0` до `0.4361`. Clean macro-F1 знижується на
`0.0741`; вилучення всіх edges — на `0.1269`. Архітектура явно реагує
на правильну інформативну topology.

### 5.4 Суворий target-prior stress test

| Condition | `N0` accuracy | `G_repeat` accuracy |
|---|---:|---:|
| only target replaced by zero + train prior | `0.5300` | `0.9074` |
| whole repeat component replaced | `0.5300` | `0.0690` |

Екстремальний stress test повторює surface-only патерн: GNN відновлює target,
коли clean peers залишаються, і провалюється, коли relational evidence також
знищено. Через штучність zero+prior corruption ці числа є механістичними,
а не оцінкою очікуваної deployment quality.

## 6. E14 — low-resource regime

Одна fixed source-stratified nested chain містить:

| Train fraction | Документи | Згадки | Класи |
|---:|---:|---:|---:|
| `0.10` | 34 | 1 406 | 13 |
| `0.25` | 84 | 3 298 | 13 |
| `0.50` | 167 | 6 406 | 13 |
| `1.00` | 333 | 12 799 | 13 |

| Train fraction | `G_repeat − N0`, macro-F1 |
|---:|---:|
| `0.10` | `−0.0032` |
| `0.25` | `−0.0051` |
| `0.50` | `+0.0137` |
| `1.00` | `+0.0041` |

На 10% supervision confirmatory contrast дорівнює `−0.0032`,
95% CI `[-0.0092; 0.0025]`. Interaction
`mean(10%,25%) gain − full gain = −0.0082`,
95% CI `[-0.0174; 0.0005]`, Holm-adjusted `p = 0.1403`.

Отже, гіпотеза «менше labeled train data автоматично робить граф
кориснішим» не підтримана. Це важливе розрізнення: weak local evidence
конкретного target і загальне зменшення training supervision — різні режими.
На малому train learned GNN може сама потребувати більше даних, тоді як
просте `AVG_repeat` має невеликі позитивні descriptive deltas.

## 7. E15 — де концентрується deployable ефект

| Cohort | Support | `G_repeat − N0`, accuracy |
|---|---:|---:|
| repeated | 2 385 | `+0.00826` |
| singleton | 4 546 | `+0.00026` |
| uncertain | 1 771 | `+0.01175` |
| confident | 5 160 | `+0.00002` |
| uncertain + repeated | 491 | `+0.03992` |
| train-ambiguous | 82 | `+0.00366` |

Repeat-vs-singleton interaction:

`+0.0080` accuracy, 95% CI `[0.0029; 0.0129]`,
Holm-adjusted `p = 0.0045`. Ефект закономірно концентрується там, де
repeat relation справді існує.

Uncertain-vs-confident interaction має очікуваний знак:
`+0.0117`, але 95% CI `[-0.0026; 0.0274]`,
Holm-adjusted `p = 0.1218`. Тому selective routing за uncertainty є
перспективною наступною гіпотезою, але ще не confirmatory result.

Strict train-ambiguous cohort має лише 82 mentions, а його repeated subset —
21 mention. Вони подаються лише descriptively і не входять до confirmatory
family. Їх недостатньо для висновку про lexical ambiguity.

## 8. Що змінилося відносно попереднього звіту

Попередній чесний висновок був таким: основний `local → graph` приріст
походить від node-only residual head, а direct edge effect на clean data
малий і статистично нестійкий.

Новий прогін не скасовує цей висновок для clean NER. Він уточнює причину:

1. **Граф може використовувати relations.** Oracle GNN і oracle averaging
   мають значний приріст; правильний oracle graph різко деградує після
   rewiring/no-edge intervention.
2. **Repeat relations мають сильну умовну користь.** Target recovery,
   direct-message ablation і component negative control усі підтримані.
3. **Широкі semantic edges занадто шумні.** Gold-pruning дає значне
   `+0.0311` macro-F1 відносно того самого semantic union.
4. **Clean aggregate має ceiling/masking problem.** Більшість targets уже
   правильно розв'язуються локально, а repeat edges покривають лише 34.41%
   test mentions.
5. **Складність GNN ще не виправдана.** Simple averaging є конкурентним;
   non-inferiority learned oracle до averaging не встановлена.
6. **Low-resource explanation не спрацювало.** Користь з'являється не
   просто через менше train labels, а коли relational neighbors мають
   інформацію, якої бракує конкретному target.

Тому проблема була не в тому, що «ми повністю тестували граф на неправильній
задачі». Clean NER є правильною задачею для загальної deployment claim —
і ця claim залишається непідтвердженою. Але clean aggregate був недостатньою
задачею для перевірки механізму cross-mention recovery. Саме для цієї
механістичної гіпотези попередній дизайн дивився не туди.

## 9. Рекомендоване формулювання для статті

Українською:

> Стандартна оцінка на чистому NER не встановила надійної загальної переваги
> графового уточнення над архітектурно узгодженим node-only classifier.
> Водночас контрольоване послаблення локального контексту цільової згадки
> виявило великий і відтворюваний recovery effect, коли згадка залишалася
> пов'язаною з clean exact-repeat сусідами. Ефект зникав або змінював знак
> після послаблення всієї компоненти. Gold-defined і gold-pruned graphs
> додатково показали значний structural headroom. Результати вказують на
> conditional utility за локальної нестачі evidence та bottleneck якості
> ребер, а не на загальну нездатність message passing використовувати
> relational information.

Manuscript-ready English:

> Standard clean NER evaluation did not establish a reliable general benefit
> of graph refinement beyond an architecture-matched node-only classifier.
> However, controlled target-level evidence degradation produced a large and
> reproducible recovery effect when an affected mention remained connected
> to clean exact-repeat peers. The effect disappeared or reversed when the
> entire connected component was degraded. Gold-defined and gold-pruned
> diagnostic graphs further demonstrated substantial structural headroom.
> These findings indicate conditional utility under local-evidence failure
> and an edge-quality bottleneck, rather than a general inability of message
> passing to exploit relational information.

Рекомендована headline claim:

> **Graph refinement is a conditional cross-mention recovery mechanism, not a
> uniformly better replacement for local NER.**

## 10. Що варто робити далі

Найкращі наступні експерименти тепер не повинні просто додавати ще один
тип ребер до повного графа.

1. **Selective graph routing.** Активувати propagation лише для repeated
   targets із високою local uncertainty; окремо звітувати coverage,
   selective accuracy, calibration і global macro-F1.
2. **Edge precision–coverage curve.** Навчити або відкалібрувати edge scorer,
   а потім перевірити кілька validation-fixed thresholds. Gold-pruning
   показує, що саме edge composition має найбільший резерв.
3. **Simple propagation as a first-class baseline.** Порівнювати learned
   GNN з averaging/label propagation за однакового graph і routing policy.
4. **Natural weak-evidence benchmarks.** Перевірити truncated context,
   OCR/ASR degradation, заголовки, таблиці та cross-sentence mentions, а не
   лише штучне masking.
5. **Dataset із gold entity/coreference IDs.** NER-UK не дозволяє відділити
   справжню entity identity від exact surface equality. Такий benchmark
   потрібен для перевірки більш широкої relational claim.

## 11. Обмеження

- Exact repeat є silver identity relation, а не gold coreference.
- Oracle і gold-pruned graphs використовують test gold labels і не є
  deployable.
- Target-surface залишає surface text та прибирає зовнішній контекст; це
  контрольований intervention, а не природний distribution shift.
- Whole-component evaluation використовує target-adapted checkpoint без
  retraining; це навмисний causal negative control, не окрема production
  model.
- `no_edges` у вже навченому checkpoint ізолює message dependence, але не
  замінює незалежно навчений `N0`.
- Repeat rewiring зберігає degree і document constraints, але залишається
  сильно homophilous; це не label-neutral null.
- Low-resource curve умовна на одну наперед зафіксовану nested subset chain.
- Strict train-ambiguous cohort недостатньо великий для confirmatory test.
- 90% CI всередині SESOI подається лише descriptively; формального TOST не
  виконано.
- Перевагу GNN над простим averaging не встановлено.

## 12. Відтворюваність і артефакти

- optimization seeds:
  `17, 29, 43, 59, 71, 89, 101, 113, 127, 149`;
- topology/target realizations:
  `101, 211, 307, 401, 503`;
- 200 learned models, 3 020 metric records, 17 crossed-bootstrap contrasts;
- test: 169 documents, 6 931 mentions;
- config checksum:
  `143e9085c27157fdc901835488339152ff1ab9726ab64a511d303c9d408c781c`;
- source checksum:
  `aa4df43f7fc11c6c59ff78a17b1f91c4dbefe84ca1d8c77758ae1e19aa2f3d35`;
- NER-UK checkout:
  `7772b45807453854883a8e6d23e4d145014a1a42`;
- GPU: NVIDIA GeForce RTX 4070 SUPER;
- повний test suite після прогону: 36/36.

Основні таблиці:

- `table_g1_oracle_clean.csv` — clean oracle і propagation ceilings;
- `table_g2_target_recovery.csv` — target/component recovery;
- `table_g3_low_resource.csv` — resource curve;
- `table_g4_ambiguity.csv` — cohort results;
- `table_g5_counterfactual.csv` — no-edge і rewiring interventions;
- `table_g6_crossed_bootstrap.csv` — усі confirmatory estimates, CI та Holm p;
- `table_g7_graph_diagnostics.csv` — topology/model diagnostics;
- `table_g8_control_strength.csv` — фактична сила rewiring;
- `table_g9_decision_summary.csv` — компактна decision matrix усіх 17
  bootstrap-контрастів.

Згенеровані predictions, checkpoints, frozen cohort masks, exact target
selection, manifests і raw metrics збережено в
`artifacts/neruk-xlmr-diagnostics/`.
