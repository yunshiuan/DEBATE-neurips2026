# Release validation — 2026-11-06

For the initial release, the complete `debate reproduce all` command passed from freshly extracted release archives with 1,000 bootstrap resamples and no summary reuse. The checks below record that initial release; the camera-ready follow-up is recorded at the end of this document. The dataset is pinned to [commit 39ae48ab](https://huggingface.co/datasets/seantw/DEBATE_LLM/tree/39ae48ab86048a4c480e224c9a6badfabde79a39). All 86,261 extracted files passed SHA-256 verification; a separate unauthenticated public download of the historical component verified its 7 files.

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


## Camera-ready tables and SFT settings — 2026-11-06

The updated table stage matches all **100 displayed values** in the camera-ready appendix: 28 convergence values, 36 public-opinion errors, and 36 private-opinion errors. The three LaTeX exports match the manuscript values in row order. The original 36 model-family convergence values remain available in a separate provenance CSV. Verification also rejects missing, duplicate, or extra table cells.

The table stage recomputed the Qwen/Mistral metrics from the released labels and reused the four previously verified primary Llama CSVs. All 68 inventory checks passed. The original 16 figure outputs were reverified with zero mismatches; their calculations and assets did not change, so they were not rerendered in this follow-up. All 86,261 extracted data files again passed checksum verification. The published data revision remains unchanged.

The full offline suite passed **45 tests and 9 subtests**, including six training profiles through a mocked training entry point. These tests check that the recovered settings reach the model loader and trainer. A newly built wheel was installed outside the checkout: module imports, all six profile resources, all 100 numeric references, and three CLI help commands passed. The installed Qwen group profile completed a CPU dry run over 20,020 reconstructed training examples and 4,520 validation examples with batch size 4, accumulation 4, no packing, and NF4 double quantization.

No model weights were downloaded and no GPU training or live LLM calls were performed. Missing original training JSONLs and adapter weights still prevent an exact-retraining claim. The detailed follow-up record is [camera_ready_sync_20261106.json](camera_ready_sync_20261106.json); the earlier [validation report](validation_20261106.json) remains the record of the initial full reproduction run.
