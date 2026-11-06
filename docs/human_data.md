# Human data: self-reports, text annotations, and opinion pairs

Use **`self_reported_opinion`** to study human self-reported opinion change. It preserves the original `sliderValue` on a 1–6 scale. Use **`annotated_stance`** to study stance classified from human-written text by the annotation model. These are different measurements and can disagree.

The legacy **`agreement_level` field contains model-generated opinion metadata**, even in files that also contain human records. It is not a human self-report or a verbal encoding of `sliderValue`. The new human-data views exclude it, along with model generations and prompts.

## Choose a view

| File | Size | Contents |
|---|---|---|
| [`human_events.csv`](https://huggingface.co/datasets/seantw/DEBATE_LLM/blob/main/human_views/human_events.csv) | 101,396 events in 708 sessions | Human event records across the recorded sessions, including distinct repeated opinion submissions and their automatic-submission flags |
| [`paper_human_evaluation.csv`](https://huggingface.co/datasets/seantw/DEBATE_LLM/blob/main/human_views/paper_human_evaluation.csv) | 28,286 records in 697 sessions | The canonical human-evaluation records used for the paper's inventory, with cached text-based stance and validity annotations |
| [`human_opinion_pairs.csv`](https://huggingface.co/datasets/seantw/DEBATE_LLM/blob/main/human_views/human_opinion_pairs.csv) | 2,832 participant pairs | One row per participant and session, with selected initial/final self-reports, their texts, selection reasons, and all candidate record IDs |

The release masks contain 708 sessions. The paper's canonical human-evaluation inventory contains 697 sessions and 2,788 participants. These counts do not guarantee a complete or unique initial/final opinion pair for each participant. The older `golden/` directory contains 264 curated sessions and is a different subset; it is not the camera-ready evaluation selection.

With the Hugging Face `datasets` library installed:

```python
from datasets import load_dataset

events = load_dataset("seantw/DEBATE_LLM", "human_events", split="data")
paper = load_dataset("seantw/DEBATE_LLM", "paper_human_evaluation", split="data")
pairs = load_dataset("seantw/DEBATE_LLM", "human_opinion_pairs", split="data")
```

The new configurations use the neutral split name `data`. They treat only empty CSV cells as missing and preserve literal strings such as `NA`, including the cached `NA` stance label. Use `group_split` or `topic_split` to select the scientific training/test partition. The unchanged legacy `default/train` configuration is a file-loading view; its `train` label does not identify a scientific training partition.

## Regenerate the views

Install the code using the [README instructions](../README.md#reproduce-from-cached-data), then download the three required components and export:

```bash
debate download --data-root ./data --components human_inputs base_labels validity_labels
debate export-human --data-root ./data --output-dir ./human_views
```

After downloading, export runs locally without API calls, model weights, or private data. To use an existing Hugging Face dataset checkout with its Git LFS contents already downloaded, add `--source /path/to/DEBATE_LLM` to the download command.

The exporter writes the three CSVs, [`schema.json`](https://huggingface.co/datasets/seantw/DEBATE_LLM/blob/main/human_views/schema.json), and [`manifest.json`](https://huggingface.co/datasets/seantw/DEBATE_LLM/blob/main/human_views/manifest.json). The schema specifies column types and missing-value conventions. The manifest records source and output checksums. Export refuses to overwrite a nonempty output directory unless `--overwrite` is explicitly supplied for a previous export.

These are analysis views generated from the released inputs. They do not replace the archived scientific records, change the code's pinned dataset revision, or alter the paper's evaluation rules. Use `debate reproduce` for paper figures and tables.

## Event fields

Both event tables have the following fields. The machine-readable schema is the complete column reference.

| Field | Meaning |
|---|---|
| `record_id` | Stable row identifier within its view; the pairs view refers to IDs in `human_events` |
| `session_id`, `session_prefix`, `topic` | Session identity, source filename prefix used by the release, and the topic statement |
| `participant_id`, `sender_id`, `recipient_id` | Original participant keys; keep them as strings and join participants within a session |
| `event_type`, `event_order`, `chat_round_order` | Stage, event position, and chat round |
| `field`, `human_text` | Source field and its human-written response; survey rows may contain a field response rather than dialogue |
| `self_reported_opinion` | Original human rating on the 1–6 scale, when present |
| `self_reported_opinion_label` | Label derived only from `self_reported_opinion` |
| `is_auto_submitted` | Whether the source submission was automatic; missing means unavailable, not false |
| `group_split`, `topic_split` | `train` or `test` membership in each released partition |
| `source_file`, `source_row` | Source path relative to the extracted data root and 1-based data-row number, excluding the CSV header |
| `duplicate_count` | Number of identical source rows represented by the exported row, including the retained copy; normally 1 |

Automatic submission means the experiment submitted the record automatically; the flag alone does not establish whether a person actively selected the recorded slider position. The simplified legacy `raw/` CSVs omit this flag. The new views recover it from the public archived inputs without inferring it from row order or text.

The scale runs from **1 = Certainly disagree** to **6 = Certainly agree**. Missing ratings stay missing; neither text nor model-generated metadata is used to fill them. The JSON schema gives the complete label mapping and data types. When reading CSVs directly, preserve nullable types and do not interpret a missing Boolean as `false`.

The paper view additionally includes:

| Field | Meaning |
|---|---|
| `annotated_stance`, `annotated_stance_label` | Cached LLM classification of the human text; separate from the self-reported rating |
| `human_validity` | Effective human-text validity annotation used by the paper inventory: `VALID`, `INVALID`, or missing |
| `annotation_model` | Model used for the cached stance annotation (`gpt-5-mini`) |
| `stance_in_paper_inventory` | Whether an effective validity-filtered cached stance is present for the inventory calculation |

`human_validity` records the validity verdict on the text. `stance_in_paper_inventory` records whether a cached stance remains present after validity filtering; it is not simply a test for `human_validity == "VALID"`.

An `annotated_stance` of 0 preserves an unassigned `NA` annotation from the cache; it is not a seventh opinion category. The paper view preserves canonical human-evaluation texts, including messages concatenated during the original preprocessing. It is therefore not a row-for-row copy of the complete event view. Cached annotations and missing values are retained rather than regenerated.

## Repeated submissions and opinion-pair selection

The event view removes exact duplicate survey rows. It keeps distinct initial/final opinion submissions, including records with conflicting slider values. Repeated opinion rows must not be treated as independent participants.

The pairs view groups opinion submissions by `(session_id, participant_id)` and stage. It applies the following rule independently to the initial and final stage:

1. If both manual and automatic submissions exist, prefer manual over automatic candidates. Submissions with unknown flags remain candidates.
2. If only automatic submissions exist, retain them as candidates and preserve the automatic-submission flag.
3. Select a single candidate, or one representative of candidates with identical text, rating, and automatic-submission flag. If candidates remain unresolved, leave the selected fields missing and retain their IDs for inspection.

The selected text and rating always come from the same event row. The exporter does not average conflicting ratings, choose the last row by default, or infer a rating from `agreement_level` or `annotated_stance`. This is a selection policy for the new convenience view; the historical simulation, training, and evaluation code did not use a single common pair-selection rule.

Pair fields use `pre_` and `post_` prefixes:

| Suffix | Meaning |
|---|---|
| `record_id`, `text`, `self_reported_opinion`, `self_reported_opinion_label`, `is_auto_submitted` | Values from the selected initial/final event, or missing if none is selected |
| `submission_count` | Number of source submissions for that participant and stage |
| `conflicting_ratings` | Whether observed numeric ratings differ, including automatic records excluded by manual preference |
| `selection_reason` | `missing_submission`, `manual_preferred`, `single_submission`, `identical_submissions`, or `unresolved_multiple_submissions` |
| `candidate_record_ids` | JSON array of all event-record IDs for that stage, including candidates excluded by manual preference |

`self_reported_opinion_change` is final minus initial self-report and is missing unless both selected ratings are available. Positive values mean a shift toward agreement with the topic statement. Check the selection reasons, conflicts, and automatic-submission flags when choosing the sample for a new study; the export does not apply a universal completeness or quality threshold.

## Source lineage and compatibility

The event view comes from the released raw human-event CSVs. Automatic-submission flags are matched from the archived group-partition inputs using the session, participant, stage, and event order; the export checks that the opinion text and rating agree. Training and test inputs together cover the sessions once; the alternative topic-partition copies are not added as extra observations. Opinion-pair values point back to the event records. The paper view comes from the canonical human-evaluation caches and their validity inputs. Relative source paths, row numbers, and checksums make these transformations inspectable.

The legacy folders and scientific archives keep their original column names for compatibility. In those files, `sliderValue` is the human self-report, `likert_pred`/`label_pred` are text-based stance annotations, and `agreement_level` is model-generated metadata. Some source files use `message_recieved`; the new views normalize the spelling to `message_received`.

The dataset's [research-only license](https://huggingface.co/datasets/seantw/DEBATE_LLM/blob/main/LICENSE) applies to these views as well. Participant keys support linkage within the dataset and must not be used to identify participants.
