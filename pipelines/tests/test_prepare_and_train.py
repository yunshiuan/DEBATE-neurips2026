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
