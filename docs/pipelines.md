# Generation, annotation and supervised fine-tuning

The downloaded data contain the processed event records, fixed split masks, cached generations, stance labels and validity labels used by the paper reproduction commands. Reproducing results from those caches requires no model API or GPU. The commands below support new simulations and training experiments. New model calls and training runs are not expected to reproduce the cached outputs bit for bit.

All commands also work as `python -m pipelines.run ...` before installing the console entry point. Set `DATA_ROOT` to the extracted release directory and use a separate output directory:

```bash
export DATA_ROOT="$PWD/data"
export RUN_ROOT="$PWD/runs/example"
```

## Simulation

Install the simulation dependencies with `pip install -e '.[simulation]'`. A local model server additionally requires its own CUDA/vLLM environment. API clients read credentials from environment variables; they never read key files from the repository. Use `OPENAI_API_KEY` for OpenAI or install the `gemini` extra and set `GEMINI_API_KEY` for Gemini. A local vLLM server needs no provider API key.

Validate every released group-test input without loading a model or making requests:

```bash
debate pipeline simulate --data-root "$DATA_ROOT" \
  --mode next --output-dir "$RUN_ROOT/simulation" --dry-run
```

Launch a base model in one terminal:

```bash
debate pipeline serve --model meta-llama/Llama-3.1-8B-Instruct
```

Then generate in another terminal:

```bash
debate pipeline simulate --data-root "$DATA_ROOT" \
  --mode full --api huggingface \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --endpoint http://localhost:8000/v1 \
  --output-dir "$RUN_ROOT/simulation"
```

`--input-dir` selects another directory of session CSVs; `--mask` selects session prefixes from a text file. Use the released `group_split_data_bd` or `topic_split_data_bd` train/test directories under `reproduction/inputs/llama/data/sft_data`. `--limit 1` runs a single session for a smoke test. Removing `--dry-run` performs model requests.

| Mode | Chat history used for the next prediction | Public tweets |
|---|---|---|
| `full` / `v0` | Previously generated messages | Round 1 is conditioned; rounds 2 and 3 are generated |
| `next` / `v1_5` | Human messages, including after each predicted message | Round 1 is conditioned; rounds 2 and 3 are generated |

Both modes follow the recorded dyadic partner schedule, initialize personas and private opinions from the human records, and generate final private-opinion text and agreement levels. The retained research simulator excludes participants lacking a tweet in any of the three rounds, an exit survey, or a sent message. This simulation eligibility rule is distinct from the training formatter's eligibility rule.

The default SFT-inference parameters are temperature 0.7, top-p 0.9, repetition penalty 1.2, minimum 5 and maximum 258 new tokens, and seed 1. Qwen model identifiers containing `Qwen` disable thinking. For an adapter, serve it with `--adapter /path/to/adapter --adapter-name Qwen-debate-sft` (or another model-appropriate name), then pass that name to `simulate --model` and set `--model-family llama` for Qwen/Llama or `--model-family mistral` for Mistral. Keeping `Qwen` in a Qwen adapter's served name preserves the thinking-mode setting. `serve --dry-run` prints the command without loading weights. HF identifiers and local model directories are accepted by the server; obtain any model access permissions before downloading gated weights.

Outputs are written under `simulation/<session>/<model>/simulation-v0.csv` or `simulation-v1_5.csv`, with generation metadata, input hashes and prompt/history logs. An existing output is skipped only when its recorded pipeline configuration and input hash match. Use a new directory for changed configurations, or explicitly request `--overwrite`.

## Stance and validity annotation

Annotate one event CSV at a time. The judge defaults to `gpt-5-mini`; `--judge` can pin another compatible identifier and `--endpoint` can select an OpenAI-compatible endpoint.

```bash
debate pipeline annotate --input /path/to/simulation-v0.csv \
  --output "$RUN_ROOT/labels.csv" --cache "$RUN_ROOT/judge_requests.jsonl" \
  --source llm --topic 'The discussion statement' --dry-run
```

Remove `--dry-run` to make requests. `--source human` labels `text`; `--source llm` labels generated `llm_text`, except the conditioned Initial Opinion uses human text. `--task stance` or `--task validity` runs one annotation type. The topic can be inferred when the input path contains the full session prefix.

Choose the validity protocol explicitly when comparing a new run with historical results:

| `--validity-protocol` | Post Opinion validity input | Purpose |
|---|---|---|
| `generated-text` (default) | Generated `llm_text` when `--source llm` | New experiments assessing generated responses |
| `paper-v5` | `post_opinion` when present, otherwise human `text` | Reproduce the historical paper's input-selection rule |

Other validity events and all stance judgments retain their usual text source in both modes. A blank selected validity input is skipped. The protocol is recorded in output metadata and the request-cache identity; cache entries cannot be reused across protocols. `paper-v5` reproduces the historical input rule, not the nondeterministic API responses. Use the released caches for exact historical-label replay.

After annotation, use the [new-generation evaluation command](paper_reproduction.md#evaluating-new-generations) to compare human/model files with the same six stance metrics. It validates matching event keys and annotation provenance, and applies validity filtering explicitly.

The output preserves event columns and adds `likert_pred`, `label_pred`, `human_validity` or `llm_validity`, and judge metadata. Stance labels map from Certainly disagree = 1 to Certainly agree = 6. Unparseable outputs remain missing/UNKNOWN. Requests and responses are cached by a hash of the judge identifier, messages, reasoning settings and validity protocol. Repeat with `--cache-only` to replay without initializing an API client; a missing request raises an error.

Initial/post validity system prompts were recovered from recorded eval_v5 requests and checked for consistency across 307 initial and 306 post requests. Tweet/chat templates come from the retained research prompts. `pipelines/SOURCE_MANIFEST.json` records source hashes and adaptations. Historical cache replay uses the original stored labels; new annotation uses the selected protocol explicitly and does not assert equality with historical labels.

The historical selector is visible in the retained `src/eval/validate_response_update.py`, `apply_validation()` Post Opinion branch (lines 679–684): it selects `post_opinion`, falling back to `text`, independently of `isLLM`. A deterministic audit of 83 Llama SFT v0 validity files found 334 post rows. Of these, 281 had both nonempty generated text and a recorded validation prompt; all 281 prompts contained human text and none contained the generated text. All 255 matching opinion-memory rows retained the corresponding validity label while their stance-scored message was generated text. The other 53 post rows had empty generated text (48) or no recorded prompt (5). Thus historical post-opinion validity filtering cannot be interpreted as an independently verified assessment of generated final responses. No historical labels were changed for this release.

## SFT preparation and provenance

The release includes reconstructed split CSVs and the retained mixed human validity caches. The latter recover the June initial/post validity labels. Original formatted `bd_*.jsonl` training files and trained adapter weights were not recovered. Applying the retained formatter and recovered labels does **not** match the historical example counts:

| Split | Partition | Historical recorded examples | Current reconstruction |
|---|---|---:|---:|
| Group | Train | 17,057 | 20,020 |
| Group | Test | 3,804 | 4,520 |
| Topic | Train | 14,662 | 17,500 |
| Topic | Test | 6,199 | 7,040 |

Group counts were obtained by running the formatter on all 708 sessions. Topic counts sum those same session examples under the released topic masks; the formatter's group/topic branches are identical. The exact June formatter revision or original JSONLs are still required to resolve this difference. These examples are labeled `reconstructed`, including when all post-opinion validity records are available.

Prepare the two group partitions:

```bash
debate pipeline prepare-sft --data-root "$DATA_ROOT" --split group \
  --partition train --output-dir "$RUN_ROOT/sft-data"
debate pipeline prepare-sft --data-root "$DATA_ROOT" --split group \
  --partition test --output-dir "$RUN_ROOT/sft-data"
```

Replace `group` with `topic` for the topic split. Add `--dry-run` to inspect masks, hashes and post-label coverage. The default post-label source is `reproduction/inputs/llama/validity/<session>/simulation-human.csv`. `--post-validity-root` also accepts original per-session opinion-memory sidecars if recovered. Missing post labels raise an error unless `--allow-reconstructed` explicitly permits the original formatter's keep-by-default fallback; the manifest records every such fallback. Unknown labels on present rows are not treated as VALID.

Each JSONL example has system/user/assistant messages plus session, participant, split, partition and corpus-provenance fields. Training targets include later-round tweets, valid sent messages, and valid final-opinion text and Likert responses. Round-1 tweets are context only. Invalid sent messages can appear in human history but are excluded as targets. The retained formatter excludes participants with fewer than two tweet rows, no exit survey or no sent messages. It selects the first post-opinion record for each participant. Session-mask overlap is rejected, and files are never silently overwritten.

## SFT training

Training requires a separate CUDA environment with PyTorch, Transformers, TRL, PEFT, Datasets, Accelerate and bitsandbytes. Qwen/Mistral profiles additionally use Unsloth. GPU training and serving were not executed during release validation. The CPU environment lock is **not** a tested GPU-training lock.

For a new CUDA environment, use the installation guidance matching your hardware. The implementation uses the [TRL completion-only SFT API](https://huggingface.co/docs/trl/v0.21.0/sft_trainer). Qwen3.5 requires current architecture support: [Unsloth's loader](https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py) specifies Transformers >=5.2.0 and supports `text_only=True`, which this pipeline requests. The [official Qwen3.5 model card](https://huggingface.co/Qwen/Qwen3.5-4B) and [Unsloth notebook](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)_Vision.ipynb) document framework compatibility; the latter currently uses PyTorch 2.8.0 and TRL 0.22.2. Treat these as upstream installation references, not as validation of this release's full training run.

Inspect training inputs and configuration without importing GPU libraries or downloading weights:

```bash
debate pipeline train --profile llama-group \
  --train "$RUN_ROOT/sft-data/group_train.jsonl" \
  --validation "$RUN_ROOT/sft-data/group_test.jsonl" \
  --output-dir "$RUN_ROOT/adapter" --dry-run
```

To train a new experiment, remove `--dry-run` and add `--accept-reconstructed`. That flag acknowledges the corpus mismatch above. `--model` accepts an HF identifier or local model path; `--revision` pins a model revision. The CLI rejects overlapping session IDs and applies completion-only loss. Only train examples receive gradient updates. As reported in the paper, the corresponding test partition supplies validation loss for checkpoint selection.

Six profiles are in `pipelines/configs/sft_profiles.json`:

| Profiles | Epochs | Batch / accumulation | Maximum length | Packing |
|---|---:|---|---:|---|
| `llama-group`, `llama-topic` | 5 | 8 / 32 | No explicit truncation limit | No |
| `qwen-group` | 3 | 8 / 2 | 4,096 | Yes |
| `qwen-topic` | 3 | 16 / 1 | 4,096 | No |
| `mistral-group`, `mistral-topic` | 3 | 16 / 1 | 4,096 | Yes |

All profiles use learning rate 5e-5, LoRA rank/alpha 16, dropout 0, paged AdamW 8-bit and cosine scheduling with 10% warmup. Llama uses all linear modules and seed 42; cross-model profiles use seed 3407 and weight decay 0.001. The paper does not establish the precise historical Qwen/Mistral target-module list, quantization setting, dependency versions or checkpoint-selection policy. Those profiles supply explicit new-run defaults and record the unverified settings. Training records its actual runtime versions and saves the adapter, tokenizer and trainer state. The portable Llama default uses SDPA; `--attention-implementation` can select another supported backend, and `--no-compile` disables the paper's compilation setting.

## Offline validation

```bash
python -m pytest pipelines/tests -q
```

The tests run the actual research simulator with a deterministic fake provider, check both history protocols, verify round-1 conditioning and generated later tweets/final opinions, replay annotation caches without an API, check generated-text selection for post-opinion annotation, validate masked SFT examples and completion boundaries, and reject training/validation overlap. A full 135-session group-test simulation dry run and full 708-session SFT formatting run were also completed on Python 3.11 with pandas 2.2.3, NumPy 1.26.4, langchain-core 1.7.6, langchain-openai 1.7.4 and openai 2.54.0. No paid generation, model-weight download or GPU training was performed.
