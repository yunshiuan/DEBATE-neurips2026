# Release validation — 2026-11-06

The complete `debate reproduce all` command passed from freshly extracted release archives with 1,000 bootstrap resamples and no summary reuse. The dataset is pinned to [commit 39ae48ab](https://huggingface.co/datasets/seantw/DEBATE_LLM/tree/39ae48ab86048a4c480e224c9a6badfabde79a39). All 86,261 extracted files passed SHA-256 verification; a separate unauthenticated public download of the historical component verified its 7 files.

| Check | Result |
|---|---|
| Quantitative figure families | 16 passed; 19,104 CSV rows and 47,060 numeric cells; zero mismatches |
| Training-fit DeGroot coefficients | All 8 matched |
| Dataset inventory/coverage | All 68 field checks passed |
| Cross-model tables | All 72 displayed values matched |
| Demographics | All 18 aggregate fields matched; all 15 figure panels rendered |
| Historical memory ablation | All 4 values matched |
| Conversation-quality audit | All criterion rates matched; 358/360 agreement (99.4%) on 120 conversations |
| Full automated test suite | 30 tests and 9 subtests passed |
| Installed wheel outside the checkout | 9 module imports and 9 CLI help checks passed; historical stage passed on publicly downloaded data |

Validation used Python 3.11.3, NumPy 1.26.4, pandas 2.2.3, SciPy 1.10.1, and Matplotlib 3.10.9. The retained simulator emits LangChain deprecation warnings; these did not fail the tests. The five reproduction-specific tests include a complete offline simulation→annotation→metric round trip in both simulation modes, source-validity guards, and a numerical rank-tie regression. Model and judge providers were deterministic test doubles; no paid model requests or GPU training were performed.

The adapter preserves the original Python 3.10 population-SD calculation, `sqrt(pvariance)`. Python 3.11's exact-fraction square-root implementation differs by approximately one floating-point unit on some continuous DeGroot values. That can split SD ties and alter rank diagnostics. Preserving the original calculation restores exact diagnostic agreement without changing reference values or widening numeric tolerances.

These checks establish faithful reproduction of the released experiment caches. They do not resolve the disclosed historical post-opinion validity source issue, recover missing auxiliary annotations, or establish exact SFT retraining equivalence without the original formatted training files and adapters. See [coverage and limitations](../docs/paper_reproduction.md), [pipeline provenance](../docs/pipelines.md), and the [machine-readable validation report](validation_20261106.json).
