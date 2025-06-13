import torch
import os
from tft_torch.tft import TemporalFusionTransformer
import pickle
from omegaconf import OmegaConf, DictConfig
import torch
import tft_torch
print(tft_torch.__file__)
from tft_torch.tft import TemporalFusionTransformer


def test_script_and_save(tmp_path):
    # 1) instantiate & eval
    model = TemporalFusionTransformer(...your args...)
    model.eval()

    # 2) script
    scripted = torch.jit.script(model)

    # 3) save & reload
    out_file = tmp_path / "tft_scripted.pt"
    torch.jit.save(scripted, str(out_file))
    loaded = torch.jit.load(str(out_file))

    # 4) sanity-check: inference runs & shapes match
    example = torch.randn(2, 36, 10)  # match your forward signature
    y1 = model(example)
    y2 = loaded(example)
    assert y1.shape == y2.shape
