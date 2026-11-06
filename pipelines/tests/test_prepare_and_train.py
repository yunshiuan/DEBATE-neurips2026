import json

import pytest

from pipelines import prepare_sft, train_sft
from .conftest import PREFIX, synthetic_rows, write_session


def test_prepare_respects_masks_validity_and_target_format(tmp_path):
    split = tmp_path / "data/reproduction/inputs/llama/data/sft_data"
    events = synthetic_rows()
    invalid = next(row for row in events if row["event_type"] == "message_sent")
    invalid["validity"] = "INVALID"
    write_session(split / f"group_split_data_bd/train/{PREFIX}.csv", events)
    other = PREFIX.replace("120000", "120001")
    write_session(split / f"group_split_data_bd/train/{other}.csv")
    (split / "masks").mkdir()
    (split / "masks/group_train_prefixes.txt").write_text(PREFIX + "\n")
    (split / "masks/group_test_prefixes.txt").write_text(other + "\n")
    out = tmp_path / "examples"
    prepare_sft.main(["--data-root", str(tmp_path / "data"), "--split", "group", "--partition", "train",
                      "--output-dir", str(out), "--allow-reconstructed"])
    examples = [json.loads(line) for line in (out / "group_train.jsonl").read_text().splitlines()]
    assert len(examples) == 39
    assert {item["session_id"] for item in examples} == {PREFIX}
    assert {item["partition"] for item in examples} == {"train"}
    assert all([m["role"] for m in item["messages"]] == ["system", "user", "assistant"] for item in examples)
    targets = [item["messages"][-1]["content"] for item in examples]
    assert not any("HUMAN_TWEET_1_" in target for target in targets)
    assert not any(invalid["text"] in target for target in targets)
    assert sum("HUMAN_TWEET_" in target for target in targets) == 8
    assert sum("HUMAN_POST_" in target for target in targets) == 4
    assert sum(target == "Probably agree" for target in targets) == 4
    assert any(invalid["text"] in item["messages"][1]["content"] for item in examples)
    with pytest.raises(FileExistsError):
        prepare_sft.main(["--data-root", str(tmp_path / "data"), "--split", "group", "--partition", "train",
                          "--output-dir", str(out), "--allow-reconstructed"])
    (split / "masks/group_test_prefixes.txt").write_text(PREFIX + "\n")
    with pytest.raises(ValueError, match="overlap"):
        prepare_sft.main(["--data-root", str(tmp_path / "data"), "--split", "group", "--partition", "train",
                          "--output-dir", str(tmp_path / "overlap"), "--allow-reconstructed", "--dry-run"])


def test_completion_only_boundary():
    class Tokenizer:
        def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
            result = "".join(f"<{m['role']}>{m['content']}" for m in messages)
            return result + ("<assistant>" if add_generation_prompt else "")
    row = {"messages": [{"role": "system", "content": "persona"}, {"role": "user", "content": "history"},
                        {"role": "assistant", "content": "target"}]}
    output = train_sft.completion_example(Tokenizer(), row)
    assert output == {"prompt": "<system>persona<user>history<assistant>", "completion": "target"}


def test_train_plan_excludes_validation_from_gradient_data(tmp_path):
    def write(name, session, partition):
        path = tmp_path / name
        path.write_text(json.dumps({"session_id": session, "partition": partition, "training_corpus_status": "reconstructed",
                                    "messages": [{"role": "user", "content": "input"}, {"role": "assistant", "content": "target"}]}) + "\n")
        return path
    train = write("train.jsonl", "train_session", "train")
    test = write("test.jsonl", "test_session", "test")
    out = tmp_path / "adapter"
    args = ["--profile", "llama-group", "--train", str(train), "--validation", str(test), "--output-dir", str(out), "--dry-run"]
    train_sft.main(args)
    plan = json.loads((out / "training_plan.json").read_text())
    assert plan["gradient_data"] == "train only"
    assert plan["config"]["completion_only_loss"] is True
    write("test.jsonl", "train_session", "test")
    with pytest.raises(ValueError, match="overlap"):
        train_sft.main(args)


@pytest.mark.parametrize("profile", [
    "llama-group", "llama-topic", "qwen-group", "qwen-topic", "mistral-group", "mistral-topic",
])
def test_recovered_profiles_reach_trainer_and_model_loader(tmp_path, monkeypatch, profile):
    """Exercise the training entry point offline; catch hardcoded overrides of saved settings."""
    import sys
    from types import SimpleNamespace

    calls = {}
    family = profile.split("-")[0]
    assistant_header = "<|start_header_id|>assistant<|end_header_id|>" if family == "llama" else "<assistant>"

    class Tokenizer:
        eos_token = "<eos>"

        def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
            result = "".join((assistant_header if m["role"] == "assistant" else f"<{m['role']}>") + m["content"]
                             for m in messages)
            return result + (assistant_header if add_generation_prompt else "")

        def save_pretrained(self, output):
            calls["tokenizer_saved"] = True

    tokenizer = Tokenizer()
    model = object()

    class Dataset:
        column_names = ["messages"]

        def map(self, formatter, remove_columns):
            example = formatter({"messages": [{"role": "user", "content": "history"},
                                               {"role": "assistant", "content": "target"}]})
            assert example["completion"] == "target"
            assert example["prompt"].endswith(assistant_header)
            return self

    def load_model(*args, **kwargs):
        calls["model_loader"] = kwargs
        return model

    def load_unsloth(**kwargs):
        load_model(**kwargs)
        return model, tokenizer

    def configure_adapter(model, **kwargs):
        calls["adapter"] = kwargs
        return model

    def prepare_model(model, **kwargs):
        calls["model_preparation"] = kwargs
        return model

    def lora_config(**kwargs):
        calls["adapter"] = kwargs
        return kwargs

    class Trainer:
        def __init__(self, **kwargs):
            calls["trainer"] = kwargs

        def train(self, **kwargs):
            calls["train"] = kwargs

        def save_model(self):
            calls["model_saved"] = True

        def save_state(self):
            pass

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True),
                                                            bfloat16="bfloat16"))
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **kw: Dataset()))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
        BitsAndBytesConfig=lambda **kw: kw, set_seed=lambda seed: None))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(LoraConfig=lora_config,
                                                           prepare_model_for_kbit_training=prepare_model))
    monkeypatch.setitem(sys.modules, "trl", SimpleNamespace(SFTConfig=lambda **kw: kw, SFTTrainer=Trainer))
    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastLanguageModel=SimpleNamespace(
        from_pretrained=load_unsloth, get_peft_model=configure_adapter)))
    monkeypatch.setattr(train_sft, "version", lambda name: "offline-test")
    for partition in ["train", "test"]:
        (tmp_path / f"{partition}.jsonl").write_text(json.dumps({
            "session_id": partition, "partition": partition, "training_corpus_status": "reconstructed",
            "messages": [{"role": "user", "content": "input"}, {"role": "assistant", "content": "target"}],
        }) + "\n")
    output = tmp_path / "adapter"
    assert train_sft.main(["--profile", profile, "--train", str(tmp_path / "train.jsonl"),
                           "--validation", str(tmp_path / "test.jsonl"), "--output-dir", str(output),
                           "--accept-reconstructed"]) == 0
    arguments = calls["trainer"]["args"]
    plan = json.loads((output / "training_plan.json").read_text())
    assert arguments == plan["trainer_arguments"]
    assert calls["model_saved"] and calls["tokenizer_saved"]
    assert arguments["load_best_model_at_end"] and arguments["metric_for_best_model"] == "eval_loss"
    assert arguments["num_train_epochs"] == (5 if family == "llama" else 3)
    assert plan["historical_exact_retraining_claim"] is False
    assert plan["config"]["historical_evidence"]["selected_epoch_is_descriptive_not_forced"] is True
    quantization = calls["model_loader"]["quantization_config"]
    assert quantization["load_in_4bit"] and quantization["bnb_4bit_quant_type"] == "nf4"
    if family == "llama":
        assert arguments["gradient_checkpointing"] is False
        assert calls["model_preparation"]["use_gradient_checkpointing"] is True
        assert arguments["group_by_length"] is True
        assert arguments["per_device_train_batch_size"] == 8
        assert arguments["gradient_accumulation_steps"] == 32
        assert arguments["packing"] is False
        assert arguments["torch_compile"] is True
        assert set(calls["adapter"]["target_modules"]) == {
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        assert "20260608-bd-real" in plan["config"]["historical_evidence"]["run_id"]
    elif family == "qwen":
        import re
        assert arguments["per_device_train_batch_size"] == arguments["per_device_eval_batch_size"] == 4
        assert arguments["gradient_accumulation_steps"] == 4
        assert arguments["packing"] is False
        assert arguments["completion_only_loss"] is None
        assert arguments["torch_compile"] is False
        assert quantization["bnb_4bit_use_double_quant"] is True
        assert calls["model_loader"]["text_only"] is True
        targets = calls["adapter"]["target_modules"]
        assert re.fullmatch(targets, "model.layers.0.self_attn.q_proj")
        assert not re.fullmatch(targets, "visual.layers.0.self_attn.q_proj")
    else:
        assert arguments["per_device_train_batch_size"] == 16
        assert arguments["gradient_accumulation_steps"] == 1
        assert arguments["packing"] and arguments["eval_packing"] and arguments["padding_free"]
        assert arguments["loss_type"] == "chunked_nll"
        assert arguments["torch_compile"] is True


def test_no_compile_override_preserves_other_training_settings():
    profile = json.loads(train_sft.PROFILES.read_text())["mistral-group"]
    arguments = train_sft.trainer_arguments(profile, "output", compile_enabled=False)
    assert arguments["torch_compile"] is False
    assert "torch_compile_mode" not in arguments
    assert "torch_compile_backend" not in arguments
    assert arguments["padding_free"] is True
    assert arguments["loss_type"] == "chunked_nll"
