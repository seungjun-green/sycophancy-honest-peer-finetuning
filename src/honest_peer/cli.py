"""Thin CLI shared by Colab notebooks and local invocations."""

import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path

from .config import load_config


def small_run(config):
    old_training = Path(config["paths"]["training"])
    old_adapter = Path(config["model"]["adapter"]) if config["model"]["adapter"] else None
    config["smoke"] = True
    config["data"]["sampling"].update(total=10, validation=2)
    config["data"]["max_source_records"] = 200
    config["training"]["epochs"] = 0.5
    config["evaluation"]["samples"] = {name: 20 for name in config["evaluation"]["samples"]}
    # Keep live tiny runs separate from full research runs.
    config["paths"] = {
        key: str(Path("outputs/live-smoke") / Path(path).relative_to("outputs"))
        if str(path).startswith("outputs/")
        else str(Path(path) / "live-smoke")
        for key, path in config["paths"].items()
    }
    if old_adapter is not None:
        try:
            config["model"]["adapter"] = str(
                Path(config["paths"]["training"]) / old_adapter.relative_to(old_training)
            )
        except ValueError:
            raise ValueError("For --smoke, the adapter must be inside paths.training") from None
    return config


async def dispatch(args, config):
    if args.command == "smoke":
        from .smoke import run_smoke

        return await run_smoke(config, args.output)
    if args.command == "build-data":
        from .data import build_data

        return await build_data(config)
    if args.command == "prepare-data":
        from .data import prepare_data

        return await prepare_data(config)
    if args.command == "rewrite-data":
        from .data import rewrite_data

        if args.per_category is not None:
            config["rewrite"]["preview_per_category"] = args.per_category
        if args.turns_per_conversation is not None:
            config["rewrite"]["preview_turns_per_conversation"] = (
                None if args.turns_per_conversation == 0 else args.turns_per_conversation
            )
        return await rewrite_data(config, preview=args.preview)
    if args.command == "generate-pushback-data":
        from .pushback_data import generate_pushback_data

        return await generate_pushback_data(config)
    if args.command == "train":
        from .training import train

        return train(config)
    if args.command in ("prepare", "evaluate"):
        from .benchmarks import prepare
        from .evaluation import evaluate

        names = config["evaluation"]["benchmarks"] if args.benchmark == "all" else [args.benchmark]
        results = []
        for name in names:
            results.append(
                await (prepare(config, name) if args.command == "prepare" else evaluate(config, name))
            )
        return results
    if args.command == "compare":
        from .results import compare

        return compare(config)
    if args.command == "spot-check":
        from .data import spot_check

        return {"html": str(spot_check(config, args.count))}
    if args.command == "inspect":
        from .sources import load_source

        rows, source = load_source(config["sources"][args.source], config["paths"]["cache"])
        first = next(iter(rows), None)
        return {"source": source, "first_row": first}
    if args.command == "tune-concurrency":
        from .api import plan_concurrency, probe_rate_limits
        from .data import estimate_rewrite_tokens

        settings = config["rewrite"]
        workload = estimate_rewrite_tokens(config)
        probe = await probe_rate_limits(settings)
        # Rewrite latency ~ time to first token + output length at a conservative 50 tokens/s.
        latency = probe["probe_latency_seconds"] + workload["output_tokens_per_call"] / 50
        concurrency, binding = plan_concurrency(
            probe["limits"],
            workload["input_tokens_per_call"],
            workload["output_tokens_per_call"],
            latency,
            safety=args.safety,
            cap=args.max,
        )
        return {
            "model": settings["model"],
            **probe,
            **workload,
            "estimated_latency_seconds": round(latency, 1),
            "binding_limit": binding,
            "concurrency": concurrency or settings["concurrency"],
            "source": "rate_limit_headers" if concurrency else "config_default_no_headers",
        }
    if args.command == "doctor":
        report = {
            "environment": {key: bool(os.environ.get(key)) for key in ("HF_TOKEN", "ANTHROPIC_API_KEY")},
            "dependencies": {
                name: importlib.util.find_spec(name) is not None
                for name in ("torch", "peft", "vllm", "detoxify", "lm_eval")
            },
        }
        if report["dependencies"]["torch"]:
            import torch

            report["cuda"] = torch.cuda.is_available()
            report["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        if args.api:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic()
            try:
                models = await client.models.list(limit=100)
                names = [model.id for model in models.data]
                report["available_anthropic_models"] = names
                report["configured_models_available"] = {
                    config[key]["model"]: config[key]["model"] in names for key in ("rewrite", "judge")
                }
            finally:
                await client.close()
        return report
    raise ValueError(args.command)


def main():
    # Colab's inline backend belongs to its kernel, not the isolated project environment.
    # Set this before imports in dispatch can load Matplotlib (e.g. through Trainer).
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser(description="Honest-peer fine-tuning experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "build-data",
        "prepare-data",
        "rewrite-data",
        "generate-pushback-data",
        "train",
        "prepare",
        "evaluate",
        "compare",
        "spot-check",
        "inspect",
        "doctor",
        "smoke",
        "tune-concurrency",
    ):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/default.yaml")
        command.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        command.add_argument(
            "--smoke",
            action="store_true",
            help="Small live run using real configured backends; synthetic offline smoke is a separate command.",
        )
        if name in ("prepare", "evaluate"):
            command.add_argument("--benchmark", default="all")
        if name == "spot-check":
            command.add_argument("--count", type=int, default=50)
        if name == "rewrite-data":
            command.add_argument(
                "--preview",
                action="store_true",
                help="Live Claude prompt review: 5 training conversations per category, 1 turn each by default.",
            )
            command.add_argument(
                "--per-category", type=int, help="Override the preview conversations per category."
            )
            command.add_argument(
                "--turns-per-conversation",
                type=int,
                help="Preview: assistant turns to rewrite per conversation (0 = all).",
            )
        if name == "inspect":
            command.add_argument("--source", default="sharegpt")
        if name == "doctor":
            command.add_argument(
                "--api",
                action="store_true",
                help="List account-visible Anthropic models without generating content",
            )
        if name == "tune-concurrency":
            command.add_argument("--max", type=int, default=32, help="Upper bound on concurrency")
            command.add_argument(
                "--safety", type=float, default=0.7, help="Fraction of the rate limit to plan for"
            )
        if name == "smoke":
            command.add_argument("--output", default="outputs/smoke")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    if args.smoke:
        config = small_run(config)
    try:
        result = asyncio.run(dispatch(args, config))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
