# Reproducing the paper results

The release regenerates the camera-ready main results from the original Ruixuan v5 evaluation, using the supplied group and topic partitions. The later evaluation-protocol variants are not used. The `reproduction/reference` files contain manuscript values for checking outputs; they are never used as inputs to metric computation or figure rendering.

After installing the package and downloading the data, run:

```bash
debate download --data-root data
debate reproduce all --data-root data --output-dir outputs/paper
```

This performs cached evaluation without an API key or GPU. It fits the DeGroot coefficients on training data, recomputes the test metrics and confidence intervals, renders figures, rebuilds the quantitative tables and demographics, and compares numeric outputs with the manuscript. The default uses 1,000 bootstrap resamples and the original random seeds. It takes several minutes and reads many CSV files. Downloading the dataset is the only network step.

Individual stages are available for a smaller rerun:

```bash
debate reproduce primary --data-root data --output-dir outputs/paper
debate reproduce tables --data-root data --output-dir outputs/paper
debate reproduce demographics --data-root data --output-dir outputs/paper
debate reproduce historical --data-root data --output-dir outputs/paper
debate reproduce verify --data-root data --output-dir outputs/paper
```

`tables` reads the primary stage's computed Llama results, so run `primary` first. `--reuse-summary` explicitly reuses metric summaries already produced in the same output directory; without that option the primary stage recomputes metrics from cached label rows. `--include-train` additionally computes training-partition summaries, which are not plotted in the main figures. Changing `--bootstrap-reps` changes confidence intervals and is not expected to pass the manuscript check.

## Figure and table map

Identifiers below are stable release identifiers. The exact manuscript asset names are in [`figure_mapping.json`](../reproduction/reference/figure_mapping.json), including legacy filenames whose wording does not describe the current evaluation population.

| Release output | Paper result | Computation |
|---|---|---|
| `primary/group_test_set/M1` | Six stance-error panels, group test | Cached per-message stance/validity labels; both simulation modes |
| `M2` | Public/private error correlations, group test | Recomputed model-level errors |
| `M3` | Convergence trajectories and changes, group test | Cached per-group stance trajectories |
| `M4` | Stance transition matrices, group test | Matched initial/final stance labels |
| `M5` | Partner gap, opinion change, Spearman correlations | Cached paired conversation trajectories and bootstrap intervals |
| `primary/topic_test_set/A3`–`A5`, `A8`, `A9` | Corresponding topic-test panels | Same calculations on the topic-test partition |
| `primary/group_test_set/A1`, `primary/topic_test_set/A2` | Text metrics beyond stance | Released original scored text-metric aggregate caches |
| `A6_public`, `A6_private`, `A7_public`, `A7_private` | All public/private opinion-distribution panels | Cached stance labels; human, seven base models, SFT, DeGroot |
| `tables/dataset_inventory.csv` and `.tex` | Table 1 human inventory | Canonical human message records and validity sidecars |
| `tables/dataset_totals.json` | Table 1 total row | Union of the topic partitions |
| `tables/dataset_inventory.csv`, `dataset_coverage.csv` | Appendix human-data and SFT-coverage tables | Original masks and actual available human/model files |
| `tables/cross_model_convergence.csv` and `.tex` | Cross-model convergence table | Llama, Qwen, Mistral cached trajectories |
| `tables/cross_model_private_errors.csv` and `.tex` | Cross-model private-opinion error table | Cached stance labels for each model family |
| `demographics/aggregates.json`, `demo_*_fig.pdf` | Demographic text and all 15 figure panels | Original exit-survey responses in the reconstructed 708-session split CSVs |
| `historical/conversation_quality.csv` | 120-conversation quality-audit table | Paired independent human annotations |
| `historical/memory_ablation.csv` | GPT-4o-mini memory-removal comparison | Original cached historical experiment labels |

Each figure is exported as PDF, PNG, and numeric CSV. The CSVs also preserve useful intermediate quantities and counts. Table exports prioritize usable numeric data and simple LaTeX tabular output; they do not insert themselves into a manuscript checkout. Study diagrams, UI screenshots, prompt examples, training configuration tables, and comparison metadata about other datasets are not empirical outputs of this evaluator.

## Evaluation details that affect the results

- The partitions contain 573/135 train/test groups for group generalization and 511/197 for topic generalization. The canonical human inventory contains 697 available groups among the 708 sessions. Missing cached outputs remain missing; no labels are synthesized to fill them.
- `v0` means Full Dynamics Simulation; `v1_5` means Next Message Prediction. These are retained in cache filenames to keep joins unambiguous.
- The seven base-model series use the original base evaluation cache. Llama SFT uses its own delivered evaluation cache. Message validity and available examples can differ between them. The code preserves the original metric-specific inclusion rules and reports per-prefix availability in `primary/csv/per_prefix_status.csv`.
- Table 1 uses the canonical human records and message-type-specific validity sidecars. The SFT result figures use the human/model validity records associated with that evaluation. Thus Table 1 counts are an inventory, not the denominator of every plotted metric.
- DeGroot is refit separately for each split and simulation mode using the training partition, never the test labels. Fitted coefficients and fit counts are written to `primary/csv/degroot_fit_alphas.csv`.
- The Qwen private-error calculation retains the original common successful-group intersection across the Qwen and accompanying Llama comparisons. Its released Llama cache is an eligibility dependency, even though the cross-model table reports only Qwen in that section. Mistral uses its own base/SFT comparison. Interpret changes within a model family, not differences between families' absolute errors.
- The text-metric panels use the original scored aggregate caches. Their base and SFT populations differ, as in the paper; these panels are descriptive and are not a paired comparison on a common test subset. The release does not relabel these aggregates as a new rescore of the stance-evaluation records.
- Histograms retain the original count denominators and stance-bin handling. Records outside the displayed bins are exposed through `unbinned_n`; changing the denominator to plotted bins would change the paper's percentages.
- Demographic percentages use item-specific response denominators. Multi-select percentages use respondents, so their sum can exceed 100%. The aggregation reports exclusions and missing values and does not export participant IDs.

## Numeric verification

`verification.json` checks all 16 quantitative figure families against the manuscript CSVs, including the numeric plotting inputs, counts, matrices, correlations, and confidence intervals. The separate fitted-alpha file is also checked. The old manuscript error-panel CSVs contain stale alpha metadata in otherwise valid result rows; those two unused metadata columns are excluded from row comparison, and the actual refitted coefficients are checked directly.

The adapter preserves Python 3.10 population-SD rounding (`sqrt(pvariance)`). Python 3.11 changed `statistics.pstdev` to an exact-fraction square root; differences of approximately one floating-point unit can split continuous DeGroot SD ties and change rank diagnostics. `numerics.py` preserves the original calculation on both supported Python versions, and a regression test checks the resulting rank tie. No tolerance is widened for plotted values or rank metrics.

The other stages write `tables/dataset_verification.json`, `tables/cross_model_verification.json`, `demographics/verification.json`, `historical/memory_ablation_verification.json`, and `historical/conversation_quality_verification.json`. Cross-model values are checked at the paper's displayed precision. PDF and PNG bytes may differ with fonts or plotting-library versions; numeric agreement is the reproducibility criterion. The plotting dependencies are pinned in the package configuration, and figure geometry and model order follow the publication renderers.

## Evaluating new generations

New simulations can be evaluated without importing them into the fixed paper-cache tree. Annotate the human session CSV with `--source human` and its generated simulation CSV with `--source llm`, using both stance and validity tasks. Then run:

```bash
debate evaluate --human runs/human_labels.csv --llm runs/model_labels.csv \
  --session SESSION_PREFIX --model MODEL_NAME --simulation-mode v0 \
  --output-dir outputs/new_run
```

Use `--simulation-mode v1_5` for Next Message Prediction. Any CSV filenames are accepted. The command reads the actual source-specific `human_validity` and `llm_validity` columns, so an inherited generic `validity` column cannot accidentally substitute for a model judgment. It preserves participant IDs, normalizes numeric event keys such as `1` and `1.0`, and requires keyed row alignment; it rejects the legacy positional fallback. Output `summary_metrics.csv` contains the same six public/private stance prediction errors, counts, associated correlations, and group-bootstrap intervals as the paper's scientific metric code. It does not compute embedding-based text scores.

For multiple sessions, supply `--pairs pairs.json` instead of `--human`, `--llm`, and `--session`. Paths are relative to the JSON file, for example:

```json
[
  {"session":"session_1","human":"human/session_1.csv","llm":"model/session_1.csv","model":"my-model"},
  {"session":"session_2","human":"human/session_2.csv","llm":"model/session_2.csv","model":"my-model"}
]
```

`evaluation_manifest.json` records usable and excluded rows, join coverage, judge identifiers, validity protocol, and bootstrap configuration. When annotation metadata exists, its output checksum is checked. Mixed LLM validity protocols are rejected so they cannot silently be pooled. The default `--validity-policy valid-only` excludes missing or UNKNOWN validity. `--validity-policy paper-inline` instead preserves the historical rule of excluding only explicit INVALID labels. This choice concerns how existing labels are used; the separate annotation option `--validity-protocol` controls which text is judged. The selected choices remain visible in the new evaluation outputs.

Offline integration tests exercise both real simulation modes with a deterministic fake model, annotate their outputs with a fake judge, and verify all six resulting metrics against known values. They also test that arbitrary filenames retain validity filtering and that missing model-specific or UNKNOWN labels cannot inherit a human validity flag.

## Historical validity-label limitation

The archived Llama SFT Post Opinion validity requests used the paired human text when no separate `post_opinion` field was present, even when generated `llm_text` was available. An audit recovered the same validity flags in 255 matched opinion-memory rows. The paper evaluator reads these stored inline `llm_validity` labels and blanks the stance label on INVALID rows; it does not replace them with a new judgment of the generated text. This can affect private-opinion inclusion and the metrics that depend on it.

Cached replay preserves those labels and the reported results. Passing the numerical checks establishes faithful replay of the historical evaluation; it does not independently establish generated-text validity or resolve this limitation. Fresh annotation exposes `--validity-protocol generated-text` (the default, which judges generated Post Opinion text) and `--validity-protocol paper-v5` (the historical selector). The choice is recorded in request-cache identities and metadata. See [stance and validity annotation](pipelines.md#stance-and-validity-annotation). New judgments or a protocol change should be treated as a new evaluation, not silently substituted into the paper caches.

## Historical supporting analyses and remaining inputs

The main evaluation and several appendix checks come from different experiments. The release keeps that provenance visible instead of substituting the main evaluation for a missing historical experiment.

The original GPT-4o-mini memory-ablation caches reproduce the four reported convergence changes: with memory, public −0.28 and private −0.32; without memory, −0.57 and −0.59. This calculation uses the cached labels directly, population SD (`ddof=0`), and the original available-group means. There are 38 public groups in the ablation and 39 private groups. The two conditions retain their own available data.

The original conversation-quality annotation workbook was also recovered. Its sanitized paired annotations reproduce all three per-annotator rates and 358 agreements among 360 criterion judgments (99.4%).

The following supporting claims currently have reported aggregate values but not the exact underlying annotations or experiment outputs needed for independent replay:

| Supporting claim | Missing input |
|---|---|
| Historical 39-group Llama pairwise distance/range/mean shift and topic Wilcoxon results | Exact Llama experiment labels and original aggregation selection; the available similarly named sample CSVs do not reproduce these results |
| Llama temperature sweep | Cached outputs/labels for all three decoding conditions |
| Human-judged convergence on 20 groups | Paired human stance annotations and selected group list |
| Second-judge robustness ranges | Paired labels from the named second judge, source/mode selection, and group list |
| 10% manual classifier validation (11,161 tweets and 7,423 private texts) | Selected examples and human validation labels |
| 400-message human–human agreement | Both independent annotators' labels and selected message list |
| Separate illustrative conversation | Its original illustrative checkpoint and trajectory are separate from the v5 experiment; it is not represented as a v5 generation |

[`coverage_manifest.json`](../reproduction/coverage_manifest.json) records these distinctions in machine-readable form. `historical/historical_coverage.json` preserves the reported values as audit references with a reproduction status. A successful main numeric check does **not** certify the unavailable studies. Additional original records can be added later without changing the main evaluation population.

## Provenance and implementation

The numerical routines and publication plotters are vendored under `reproduction/vendor`, preserving their original directory layout and author attribution. [`vendor_manifest.json`](../reproduction/vendor_manifest.json) records source and release SHA-256 hashes. Changes to those files are limited to removing machine-specific default paths. Numerical reference data have their own [dataset license](../reproduction/reference/LICENSE), separate from the MIT-licensed code. The release adapter supplies every required data path explicitly; presentation fixes are isolated in `presentation.py`, and the complete histogram compositor is in `histograms.py`. The cached run neither imports API clients nor submits requests.

Input component paths are declared in `coverage_manifest.json` and in the dataset release manifest. File checksums are verified by `debate download` and `debate verify`; numeric manuscript checks are a separate step performed by `debate reproduce`.
