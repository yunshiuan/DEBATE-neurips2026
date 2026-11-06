"""Train completion-only LoRA adapters from explicit train and validation JSONL files."""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

from .common import digest, write_json

PROFILES = Path(__file__).resolve().parent / "configs/sft_profiles.json"


def inspect_examples(path, expected_partition):
    count, sessions, statuses = 0, set(), set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            example = json.loads(line)
            messages = example.get("messages", [])
            if not messages or messages[-1].get("role") != "assistant" or not messages[-1].get("content"):
                raise ValueError(f"{path}:{count + 1}: last message must be a nonempty assistant target")
            if example.get("partition", expected_partition) != expected_partition:
                raise ValueError(f"Unexpected partition in {path}:{count + 1}")
            if example.get("session_id"):
                sessions.add(example["session_id"])
            statuses.add(example.get("training_corpus_status", "user_supplied_unverified"))
            count += 1
    if count == 0:
        raise ValueError(f"Empty training file: {path}")
    return {"count": count, "sessions": sorted(sessions), "corpus_statuses": sorted(statuses), "sha256": digest(path)}


def completion_example(tokenizer, row, llama=False, enable_thinking=False):
    messages = row["messages"]
    kwargs = {"tokenize": False, "enable_thinking": enable_thinking}
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if llama:
        header = "<|start_header_id|>assistant<|end_header_id|>"
        position = rendered.find(header)
        if position < 0:
            raise ValueError("Expected Llama assistant header is absent from the tokenizer template")
        boundary = position + len(header)
        return {"prompt": rendered[:boundary].strip(), "completion": rendered[boundary:].strip()}
    prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **kwargs)
    if not rendered.startswith(prompt):
        raise ValueError("Chat template does not preserve the prompt prefix; provide a compatible tokenizer rather than silently changing the target mask")
    return {"prompt": prompt, "completion": rendered[len(prompt):]}


def main(argv=None):
    profiles = json.loads(PROFILES.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=profiles, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", help="Override base HF identifier or use a local model directory")
    parser.add_argument("--revision", help="Pin a Hugging Face model revision")
    parser.add_argument("--attention-implementation", default="sdpa", help="Portable default; original Llama implementation requested flash_attention_3")
    parser.add_argument("--accept-reconstructed", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate data, split separation and config without importing GPU libraries or downloading weights")
    args = parser.parse_args(argv)
    config = profiles[args.profile].copy()
    if args.model:
        config["model"] = args.model
    train_info = inspect_examples(args.train, "train")
    validation_info = inspect_examples(args.validation, "test")
    overlap = set(train_info["sessions"]) & set(validation_info["sessions"])
    if overlap:
        raise ValueError(f"Training and validation sessions overlap: {sorted(overlap)[:3]}")
    reconstructed = "reconstructed" in train_info["corpus_statuses"] + validation_info["corpus_statuses"]
    if reconstructed and not args.accept_reconstructed and not args.dry_run:
        parser.error("These examples are reconstructed, not the original training corpus. Use --accept-reconstructed to run a new experiment with the documented limitation.")
    plan = {"profile": args.profile, "config": config, "train": train_info, "validation": validation_info,
            "gradient_data": "train only", "model_selection_data": "validation (paper test partition)",
            "model_revision": args.revision, "attention_implementation": args.attention_implementation,
            "torch_compile": config["torch_compile"] and not args.no_compile,
            "historical_exact_retraining_claim": False}
    write_json(args.output_dir / "training_plan.json", plan)
    if args.dry_run:
        print(f"Validated {train_info['count']} train and {validation_info['count']} validation examples; no model loaded.")
        return 0
    if config["backend"] == "unsloth":
        # Unsloth must be imported before transformers/peft for its patches.
        from unsloth import FastLanguageModel
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer
    if not torch.cuda.is_available():
        raise RuntimeError("The training profiles require a CUDA GPU; cached reproduction does not.")
    set_seed(config["seed"])
    revision = {"revision": args.revision} if args.revision else {}
    peft_config = None
    if config["backend"] == "unsloth":
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["model"], max_seq_length=config["max_length"],
            load_in_4bit=config["load_in_4bit"], dtype=torch.bfloat16,
            **({"text_only": True} if args.profile.startswith("qwen") else {}), **revision)
        model = FastLanguageModel.get_peft_model(
            model, r=config["lora_rank"], lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"], bias="none", target_modules=config["target_modules"],
            use_gradient_checkpointing="unsloth", random_state=config["seed"], use_rslora=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config["model"], **revision)
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(config["model"], quantization_config=quantization,
                                                    device_map="auto", torch_dtype=torch.bfloat16,
                                                    attn_implementation=args.attention_implementation, **revision)
        model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": True})
        peft_config = LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"],
                                 lora_dropout=config["lora_dropout"], bias="none", task_type="CAUSAL_LM",
                                 target_modules=config["target_modules"])
    tokenizer.pad_token = tokenizer.eos_token
    datasets = {}
    for name, path in (("train", args.train), ("validation", args.validation)):
        dataset = load_dataset("json", data_files=str(path), split="train")
        datasets[name] = dataset.map(lambda row: completion_example(tokenizer, row, llama=args.profile.startswith("llama"),
                                                                    enable_thinking=config.get("enable_thinking", False)),
                                     remove_columns=dataset.column_names)
    training_args = SFTConfig(
        output_dir=str(args.output_dir), num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"], per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"], gradient_accumulation_steps=config["gradient_accumulation_steps"],
        completion_only_loss=True, max_length=config["max_length"], packing=config["packing"],
        optim=config["optim"], lr_scheduler_type=config["lr_scheduler_type"], warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"], adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, max_grad_norm=1.0,
        eval_strategy="epoch", save_strategy="epoch", eval_on_start=True, do_eval=True,
        metric_for_best_model="eval_loss", greater_is_better=False, load_best_model_at_end=True, save_total_limit=2,
        bf16=True, tf32=True, gradient_checkpointing=True, seed=config["seed"],
        torch_compile=plan["torch_compile"], torch_compile_backend="inductor", torch_compile_mode="max-autotune",
        report_to="none", logging_steps=1,
    )
    trainer = SFTTrainer(model=model, processing_class=tokenizer, peft_config=peft_config,
                         train_dataset=datasets["train"], eval_dataset=datasets["validation"], args=training_args)
    dependencies = {}
    for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "bitsandbytes", "unsloth"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            pass
    write_json(args.output_dir / "runtime_versions.json", dependencies)
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    trainer.save_model()
    trainer.save_state()
    tokenizer.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
