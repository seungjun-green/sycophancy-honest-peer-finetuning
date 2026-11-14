"""Paired comparisons, exact McNemar tests, and item-bootstrap confidence intervals."""

import csv
import io
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .artifacts import Run, atomic_text, file_digest, read_jsonl, write_json
from .evaluation import aggregate_metrics


def mcnemar_exact(first, second):
    if len(first) != len(second):
        raise ValueError("Paired outcomes must have identical lengths")
    lost = sum(a and not b for a, b in zip(first, second, strict=True))
    gained = sum(not a and b for a, b in zip(first, second, strict=True))
    discordant = lost + gained
    return {
        "base_true_variant_false": lost,
        "base_false_variant_true": gained,
        "n_pairs": len(first),
        "p_value": float(binomtest(lost, discordant, 0.5).pvalue) if discordant else 1.0,
    }


def bootstrap_metrics(records, repeats, confidence, seed):
    rng = np.random.default_rng(seed)
    values = {key: [] for key in aggregate_metrics(records)}
    for _ in range(repeats):
        sample = [records[index] for index in rng.integers(0, len(records), size=len(records))]
        for key, result in aggregate_metrics(sample).items():
            if result["value"] is not None:
                values[key].append(result["value"])
    alpha = (1 - confidence) / 2
    return {
        key: [float(np.quantile(scores, alpha)), float(np.quantile(scores, 1 - alpha))] if scores else None
        for key, scores in values.items()
    }


def compare(config):
    cache_dir = Path(config["paths"]["cache"]).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(config["paths"]["evaluation"])
    files = sorted(root.glob("*/*/summary.json"))
    if not files:
        raise ValueError(f"No summaries found in {root}")
    inputs = {str(path): file_digest(path) for path in files}
    for path in files:
        inputs[str(path.parent / "records.jsonl")] = file_digest(path.parent / "records.jsonl")
    output = Path(config["paths"]["comparison"])
    with Run(output, config, inputs):
        experiments = {}
        synthetic = set()
        for path in files:
            summary = json.loads(path.read_text())
            records = list(read_jsonl(path.parent / "records.jsonl"))
            if len(records) != summary["n_items"] or len({r["id"] for r in records}) != len(records):
                raise ValueError(f"Missing/duplicate per-item records in {path}")
            synthetic.add(summary["synthetic"])
            experiments[summary["benchmark"], summary["variant"]] = (summary, records)
        if len(synthetic) > 1:
            raise ValueError("Cannot compare synthetic smoke outputs with real research outputs")
        variants = sorted({variant for _, variant in experiments})
        baseline = config["comparison"]["baseline"]
        if baseline not in variants:
            raise ValueError(f"Baseline {baseline} not found")
        variants = [baseline] + [variant for variant in variants if variant != baseline]
        table, tests, intervals, deltas = {}, {}, {}, {}
        for (benchmark, variant), (summary, records) in experiments.items():
            if (benchmark, baseline) not in experiments:
                raise ValueError(f"Missing baseline for {benchmark}")
            base_summary, base_records = experiments[benchmark, baseline]
            if (
                summary["selection_sha256"] != base_summary["selection_sha256"]
                or summary["protocol_sha256"] != base_summary["protocol_sha256"]
            ):
                raise ValueError(f"Unpaired items or different prompts/settings for {benchmark}/{variant}")
            base_map, current_map = {r["id"]: r for r in base_records}, {r["id"]: r for r in records}
            if set(base_map) != set(current_map):
                raise ValueError(f"Item IDs differ for {benchmark}/{variant}")
            ci = bootstrap_metrics(
                records,
                config["comparison"]["bootstrap_samples"],
                config["comparison"]["confidence"],
                config["seed"],
            )
            for metric, value in summary["metrics"].items():
                key = f"{benchmark}/{metric}"
                table.setdefault(key, {})[variant] = value["value"]
                intervals.setdefault(key, {})[variant] = ci[metric]
                base_value = base_summary["metrics"].get(metric, {}).get("value")
                deltas.setdefault(key, {})[variant] = (
                    value["value"] - base_value
                    if value["value"] is not None and base_value is not None
                    else None
                )
                if variant != baseline:
                    pairs = [
                        (
                            base_map[identifier]["outcomes"].get(metric),
                            current_map[identifier]["outcomes"].get(metric),
                        )
                        for identifier in sorted(base_map)
                    ]
                    # Binary exact tests apply only to individual scalar outcomes, on jointly eligible items.
                    eligible = [(a, b) for a, b in pairs if type(a) is bool and type(b) is bool]
                    if eligible:
                        tests[f"{key}/{variant}"] = {
                            **mcnemar_exact([a for a, _ in eligible], [b for _, b in eligible]),
                            "jointly_eligible_only": len(eligible) != len(pairs),
                        }
        result = {
            "synthetic": synthetic.pop(),
            "table": table,
            "deltas_vs_base": deltas,
            "confidence_intervals": intervals,
            "paired_tests": tests,
            "ci_method": "percentile bootstrap resampling items; instructions stay grouped by prompt",
            "p_values_adjusted_for_multiple_comparisons": False,
        }
        write_json(output / "comparison.json", result)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            ["metric", *variants, *[f"delta_{v}_vs_{baseline}" for v in variants if v != baseline]]
        )
        for metric, row in sorted(table.items()):
            writer.writerow(
                [
                    metric,
                    *[row.get(v) for v in variants],
                    *[deltas[metric].get(v) for v in variants if v != baseline],
                ]
            )
        atomic_text(output / "comparison.csv", buffer.getvalue())
        markdown = "| Metric | " + " | ".join(variants) + " |\n|---|" + "---:|" * len(variants) + "\n"
        for metric, row in sorted(table.items()):
            markdown += (
                "| "
                + metric
                + " | "
                + " | ".join(
                    "—"
                    if row.get(v) is None
                    else f"{row[v]:.3f}"
                    + (
                        f" ({deltas[metric][v]:+.3f})"
                        if v != baseline and deltas[metric].get(v) is not None
                        else ""
                    )
                    for v in variants
                )
                + " |\n"
            )
        atomic_text(output / "comparison.md", markdown)
        for metric, row in table.items():
            valid = [v for v in variants if row.get(v) is not None]
            if not valid:
                continue
            y = [row[v] for v in valid]
            low = [max(0, row[v] - intervals[metric][v][0]) if intervals[metric].get(v) else 0 for v in valid]
            high = [
                max(0, intervals[metric][v][1] - row[v]) if intervals[metric].get(v) else 0 for v in valid
            ]
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(
                valid, y, yerr=[low, high], capsize=5, color=["#42566b", "#328a73", "#cc9254"][: len(valid)]
            )
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Rate / score")
            ax.set_title(("SYNTHETIC • " if result["synthetic"] else "") + metric, fontsize=10)
            fig.tight_layout()
            fig.savefig(output / (metric.replace("/", "__") + ".png"), dpi=160)
            plt.close(fig)
        return result
