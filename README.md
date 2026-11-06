# DEBATE: NeurIPS 2026 reproducibility code

Code for **DEBATE: A Large-Scale Benchmark for Evaluating Opinion Dynamics in Role-Playing LLM Agents**.

The associated [dataset](https://huggingface.co/datasets/seantw/DEBATE_LLM) contains human conversation records, the evaluation partitions, cached model generations, and stance and validity annotations. This repository contains the simulation, annotation, training, evaluation, and figure-generation code.

## Reproduce from cached data

Use Python 3.10 or 3.11 on Linux or macOS. Cached evaluation runs on a CPU and does not require an API key or model weights.

```bash
git clone https://github.com/yunshiuan/DEBATE-neurips2026.git
cd DEBATE-neurips2026
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
debate download --data-root ./data
debate verify --data-root ./data
debate reproduce all --data-root ./data --output-dir ./outputs/paper
```

The code pins a dataset revision. Downloads and individual extracted files are checked with SHA-256 manifests. To use a cloned Hugging Face dataset without network access:

```bash
debate download --source /path/to/DEBATE_LLM --data-root ./data
```

The complete release downloads about 1.69 GB and extracts to about 15.1 GB. Allow additional space for the download cache, temporary extraction files, and analysis outputs. `debate download --list` shows component sizes. `--components` selects components; the default downloads all. Use `--data-root` to choose a data location outside the code checkout.

The reproduction command writes PDF/PNG figures, numeric CSVs, LaTeX tables, and verification reports. See [paper reproduction](docs/paper_reproduction.md) for individual stages and the figure/table mapping.

## Run new experiments

```bash
python -m pip install -e '.[simulation]'
debate pipeline --help
```

See [pipeline instructions](docs/pipelines.md) for simulation modes, Likert and validity annotation, SFT preparation, training, and model serving. API inference requires the relevant credentials; local inference and training require suitable model access and hardware. The cached outputs provide the reproducible record of the paper's experiments. New stochastic runs and changing hosted model versions can produce different generations.

Use `debate evaluate --help` to compute the benchmark metrics from newly annotated human/model records. This command checks event alignment and applies the supplied validity labels explicitly.

SFT data preparation and training scripts are included for Llama, Qwen, and Mistral. The original formatted training JSONLs and trained adapters were not recovered; the reconstructed corpus has different example counts. The training commands support new experiments and do not claim exact retraining of the published checkpoints. See the [SFT provenance and counts](docs/pipelines.md#sft-preparation-and-provenance).

## Evaluation protocol and coverage

The camera-ready evaluation uses one set of 107 discussion topics and two partition strategies: held-out participant groups and held-out topics. It uses the valid observations required by each metric, without requiring all participants in a session to complete every stage. The split masks contain 708 sessions; 697 have canonical human-evaluation records.

The main Llama results, unchanged base-model evaluations, and Qwen/Mistral replications retain their respective evaluation references. Comparisons between Base and SFT are made within each model family. Training-partition data are used for gradient updates; the reported SFT test partitions were also used for checkpoint selection. These partitions are not an untouched model-selection holdout.

See [paper reproduction](docs/paper_reproduction.md) for the figure/table mapping and the evidence available for each analysis. The release distinguishes recomputation from individual cached labels, rerendering from retained aggregate values, and analyses whose original annotation records are unavailable. It does not substitute a different evaluation subset to fill a missing source.

## Validation and development

Install `pip install -e '.[simulation,dev]'` and run `python -m pytest -q` for offline tests. They exercise download integrity, actual simulation with a fake provider, annotation-cache replay, SFT preprocessing, and the simulation-to-metrics path. `requirements-tested-cpu.txt` records the Python 3.11 CPU validation environment; it is not a GPU-training dependency lock. Full training and paid API inference were not executed during release validation.

## Licenses

Code is distributed under the [MIT license](LICENSE), with upstream attribution in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md). The dataset and its derivatives remain subject to the separate **DEBATE Dataset Research-Only License (Non-Commercial, v1.0)** in the [data repository](https://huggingface.co/datasets/seantw/DEBATE_LLM/blob/main/LICENSE). Model access and model weights are governed by their respective providers' terms.
