import asyncio
from copy import deepcopy
from pathlib import Path

import pytest

from honest_peer.config import load_config
from honest_peer.data import build_data
from honest_peer.smoke import smoke_config
from honest_peer.training import train


def test_interrupted_training_restores_optimizer_rng_and_adapter(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import load_file
    from transformers import Trainer

    config = smoke_config(load_config("configs/default.yaml"), tmp_path / "input")
    asyncio.run(build_data(config))
    complete = deepcopy(config)
    complete["paths"]["training"] = str(tmp_path / "complete")
    train(complete)
    interrupted = deepcopy(config)
    interrupted["paths"]["training"] = str(tmp_path / "interrupted")
    save = Trainer._save_checkpoint

    def disconnect(self, *args, **kwargs):
        save(self, *args, **kwargs)
        raise KeyboardInterrupt("simulate notebook disconnection after a durable checkpoint")

    with monkeypatch.context() as patch:
        patch.setattr(Trainer, "_save_checkpoint", disconnect)
        with pytest.raises(KeyboardInterrupt):
            train(interrupted)
    assert not (Path(interrupted["paths"]["training"]) / "training-summary.json").exists()
    result = train(interrupted)
    assert result["resumed_from"].endswith("checkpoint-3")
    first = load_file(str(tmp_path / "complete/adapter/adapter_model.safetensors"))
    resumed = load_file(str(tmp_path / "interrupted/adapter/adapter_model.safetensors"))
    assert first.keys() == resumed.keys()
    assert all(torch.equal(first[name], resumed[name]) for name in first)
