import pytest
pytest.importorskip('gymnasium')
pytest.importorskip('stable_baselines3')
from gymnasium import spaces
import numpy as np
import torch
from battery_smart_charging.features import SetTransformerExtractor


def test_attention_features_shape():
    obs_space = spaces.Box(low=-1, high=2, shape=(12,), dtype=np.float32)
    extractor = SetTransformerExtractor(obs_space, n_cells=4)
    result = extractor(torch.rand(5, 12))
    assert result.shape == (5, 64)
    assert torch.isfinite(result).all()
