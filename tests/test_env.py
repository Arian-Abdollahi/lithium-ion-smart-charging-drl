import numpy as np
import pytest

pytest.importorskip('gymnasium')
from battery_smart_charging import env as env_module
from battery_smart_charging.physics import LithiumIonPack


def test_env_reset_step_and_corrected_success_label(tmp_path, monkeypatch, synthetic_model_data):
    path = tmp_path / 'dummy.mat'
    path.write_bytes(b'synthetic test fixture; not a MATLAB model')
    def pack_factory(model, n, T, dt, different_params=True):
        return LithiumIonPack(model, n, T, dt, different_params, model_data=synthetic_model_data)
    monkeypatch.setattr(env_module, 'LithiumIonPack', pack_factory)
    env = env_module.LIBPackEnv(model=path, number_of_cells=4)
    observation, _ = env.reset(seed=42)
    assert observation.shape == (12,)
    action = np.array([0.3, 0.4, 0.5, 0.6, 0.35], dtype=np.float32)
    nxt, reward, terminated, truncated, info = env.step(action)
    assert nxt.shape == (12,)
    assert np.isfinite(reward)
    assert info['success'] == (info['is_done'] and not info['safety_violation_ever'])
    assert np.isclose(info['C_smooth'], np.sum(np.abs(action)))
    assert not (terminated and truncated)
    env.close()


def test_missing_mat_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='PANmodel.mat'):
        env_module.LIBPackEnv(model=tmp_path / 'unavailable.mat')
