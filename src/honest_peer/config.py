"""YAML inheritance, explicit dotted overrides, and resolved prompt snapshots."""

from copy import deepcopy
from pathlib import Path

import yaml


def merge(base, override):
    out = deepcopy(base)
    for key, value in override.items():
        out[key] = (
            merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
        )
    return out


def load_config(path, overrides=(), _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"Config inheritance cycle at {path}")
    seen.add(path)
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    parent = cfg.pop("extends", None)
    if parent:
        cfg = merge(load_config(path.parent / parent, _seen=seen), cfg)
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError(f"Override must be key=value: {override}")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ValueError(f"Unknown config key: {key}")
            node = node[part]
        if parts[-1] not in node:
            raise ValueError(f"Unknown config key: {key}")
        node[parts[-1]] = yaml.safe_load(value)
    return resolve(cfg)


def resolve(cfg):
    cfg = deepcopy(cfg)
    from importlib.resources import files

    for section in ("rewrite", "judge"):
        settings = cfg.get(section, {})
        for key, value in list(settings.items()):
            if key.endswith("_template") and value:
                # Package prompts are portable; custom paths are relative to the working directory.
                source = (
                    files("honest_peer").joinpath("prompts", value[8:])
                    if value.startswith("package:")
                    else Path(value)
                )
                settings[key.removesuffix("_template") + "_text"] = source.read_text()
        if section == "judge":
            import json

            from .vendor import mask_prompts

            defaults = {
                key: value
                for key, value in vars(mask_prompts).items()
                if key.isupper() and isinstance(value, str)
            }
            custom = settings.get("mask_prompts_file")
            replacements = json.loads(Path(custom).read_text()) if custom else {}
            if set(replacements) - set(defaults) or any(
                not isinstance(value, str) for value in replacements.values()
            ):
                raise ValueError("MASK prompt replacements must map official prompt constant names to text")
            settings["mask_texts"] = {**defaults, **replacements}
    return cfg
