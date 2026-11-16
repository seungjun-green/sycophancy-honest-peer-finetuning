"""PEFT causal-LM SFT using Transformers Trainer and explicit assistant-only labels."""

import json
import math
import os
from pathlib import Path

from .artifacts import Run, atomic_text, file_digest, read_jsonl, seed_everything, write_json
from .chat import (
    AssistantCollator,
    encode_assistant_only,
    get_tokenizer,
    mask_report,
    messages_for_model,
    template_info,
)


def cosine_multiplier(step, warmup_steps, total_steps, minimum=0.1):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def train(config):
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.optim.lr_scheduler import LambdaLR
    from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    settings = config["training"]
    tiny = settings["backend"] == "tiny"
    if tiny and not config["smoke"]:
        raise ValueError("Tiny training is synthetic; set smoke=true.")
    if not tiny and not torch.cuda.is_available():
        raise RuntimeError("Real SFT requires a CUDA GPU. Use `honest-peer smoke` for local CPU validation.")
    seed_everything(config["seed"])
    root = Path(config["paths"]["data"])
    paths = {split: root / f"{split}.{settings['data_variant']}.jsonl" for split in ("train", "validation")}
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"Training data missing: {path}; run build-data first.")
    inputs = {str(path): file_digest(path) for path in paths.values()}
    out = Path(config["paths"]["training"])
    # A training variant config can also name its future adapter path; loading the base is explicit below.
    with Run(out, config, inputs):
        if (out / "training-summary.json").exists():
            return json.loads((out / "training-summary.json").read_text())
        if settings["checkpoint_every_epochs"] <= 0:
            raise ValueError("training.checkpoint_every_epochs must be positive")
        if tiny:
            from transformers import LlamaConfig, LlamaForCausalLM

            from .smoke import TinyTokenizer

            torch.set_num_threads(1)
            tokenizer = TinyTokenizer()
            model = LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=len(tokenizer),
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=512,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            )
        else:
            tokenizer = get_tokenizer(config)
            model = AutoModelForCausalLM.from_pretrained(
                config["model"]["name"],
                revision=config["model"]["revision"],
                token=os.environ.get("HF_TOKEN") or False,
                torch_dtype=torch.bfloat16 if settings["bf16"] else torch.float32,
            )
        model.config.use_cache = False
        model = get_peft_model(
            model,
            LoraConfig(
                r=settings["rank"],
                lora_alpha=settings["alpha"],
                lora_dropout=settings["dropout"],
                target_modules=settings["target_modules"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        prepared, stats, reports, groups = {}, {}, {}, {}
        # training.validation: false trains on train + validation files together and skips evaluation.
        use_validation = settings.get("validation", True)
        sources = (
            {split: [path] for split, path in paths.items()}
            if use_validation
            else {"train": [paths["train"], paths["validation"]]}
        )
        for split, files in sources.items():
            examples, truncated, dropped = [], 0, 0
            rows = [row for path in files for row in read_jsonl(path)]
            groups[split] = {row["group_id"] for row in rows}
            for row in rows:
                encoded = encode_assistant_only(
                    tokenizer,
                    messages_for_model(row["messages"], config["model"]),
                    settings["max_seq_length"],
                )
                truncated += encoded["original_length"] > settings["max_seq_length"]
                # Labels are shifted by causal LM loss; require at least one predictable assistant token.
                if all(label == -100 for label in encoded["labels"][1:]):
                    dropped += 1
                    continue
                reports.setdefault(split, mask_report(tokenizer, encoded))
                examples.append({key: encoded[key] for key in ("input_ids", "attention_mask", "labels")})
            if not examples:
                raise ValueError(f"All {split} examples have no assistant loss after truncation.")
            prepared[split] = examples
            stats[split] = {
                "source_examples": len(rows),
                "used_examples": len(examples),
                "truncated": truncated,
                "dropped_no_assistant_tokens": dropped,
            }
        if use_validation and groups["train"] & groups["validation"]:
            raise ValueError("Training and validation contain overlapping conversation groups")
        template = template_info(tokenizer)
        stats["template"] = template
        stats["model_revision"] = getattr(model.config, "_commit_hash", None)
        write_json(out / "data-stats.json", stats)
        write_json(out / "mask-report.json", reports)
        tokenizer.save_pretrained(out / "tokenizer")

        class ResearchTrainer(Trainer):
            def create_scheduler(self, num_training_steps, optimizer=None):
                if self.lr_scheduler is None:
                    warmup = self.args.get_warmup_steps(num_training_steps)
                    self.lr_scheduler = LambdaLR(
                        optimizer or self.optimizer,
                        lambda step: cosine_multiplier(
                            step, warmup, num_training_steps, settings["min_lr_ratio"]
                        ),
                    )
                return self.lr_scheduler

        class ResearchCallback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                interval = settings["checkpoint_every_epochs"]
                self.next_checkpoint = (math.floor(((state.epoch or 0) + 1e-8) / interval) + 1) * interval

            def on_step_end(self, args, state, control, **kwargs):
                interval = settings["checkpoint_every_epochs"]
                if (state.epoch or 0) + 1e-8 >= self.next_checkpoint or state.global_step == state.max_steps:
                    control.should_save = True
                    control.should_evaluate = use_validation
                    while self.next_checkpoint <= (state.epoch or 0) + 1e-8:
                        self.next_checkpoint += interval
                return control

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    with (out / "loss.jsonl").open("a") as f:
                        f.write(json.dumps({"step": state.global_step, "epoch": state.epoch, **logs}) + "\n")
                        f.flush()
                        os.fsync(f.fileno())

            def on_save(self, args, state, control, **kwargs):
                checkpoint = out / f"checkpoint-{state.global_step}"
                atomic_text(checkpoint / "manifest.json", (out / "manifest.json").read_text())
                write_json(checkpoint / "chat-template.json", template)

        if settings["wandb"]:
            os.environ["WANDB_PROJECT"] = settings["wandb_project"]
        arguments = TrainingArguments(
            output_dir=str(out),
            num_train_epochs=settings["epochs"],
            per_device_train_batch_size=settings["per_device_batch_size"],
            per_device_eval_batch_size=settings["per_device_batch_size"],
            gradient_accumulation_steps=settings["gradient_accumulation_steps"],
            learning_rate=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
            adam_beta1=settings["adam_betas"][0],
            adam_beta2=settings["adam_betas"][1],
            optim="adamw_torch",
            max_grad_norm=settings["max_grad_norm"],
            warmup_ratio=settings["warmup_ratio"],
            bf16=settings["bf16"],
            gradient_checkpointing=settings["gradient_checkpointing"],
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=settings["logging_steps"],
            report_to=["wandb"] if settings["wandb"] else [],
            seed=config["seed"],
            data_seed=config["seed"],
            remove_unused_columns=False,
            dataloader_num_workers=0,
            use_cpu=tiny,
            label_names=["labels"],
        )
        trainer = ResearchTrainer(
            model=model,
            args=arguments,
            train_dataset=prepared["train"],
            eval_dataset=prepared.get("validation"),
            processing_class=tokenizer,
            data_collator=AssistantCollator(tokenizer.pad_token_id),
            callbacks=[ResearchCallback()],
        )
        last = get_last_checkpoint(str(out)) if settings["resume"] else None
        result = trainer.train(resume_from_checkpoint=last)
        validation = trainer.evaluate() if use_validation else None
        adapter_path = out / "adapter"
        model.save_pretrained(adapter_path, safe_serialization=True)
        tokenizer.save_pretrained(adapter_path)
        atomic_text(adapter_path / "manifest.json", (out / "manifest.json").read_text())
        write_json(adapter_path / "chat-template.json", template)
        summary = {
            "synthetic": tiny,
            "train": result.metrics,
            "validation": validation,
            "adapter": str(adapter_path),
            "resumed_from": last,
            "effective_batch_size": settings["per_device_batch_size"]
            * settings["gradient_accumulation_steps"]
            * arguments.world_size,
        }
        if settings["merge_adapter"]:
            merged = model.merge_and_unload()
            merged.save_pretrained(out / "merged", safe_serialization=True)
            tokenizer.save_pretrained(out / "merged")
            atomic_text(out / "merged" / "manifest.json", (out / "manifest.json").read_text())
        if config["hub"]["push_adapter"]:
            repo = config["hub"]["adapter_repo"]
            if not repo:
                raise ValueError("Set hub.adapter_repo before pushing")
            from huggingface_hub import HfApi

            api = HfApi(token=os.environ.get("HF_TOKEN") or False)
            api.create_repo(repo, private=config["hub"]["private"], exist_ok=True)
            api.upload_folder(repo_id=repo, folder_path=adapter_path)
        write_json(out / "training-summary.json", summary)
        return summary
