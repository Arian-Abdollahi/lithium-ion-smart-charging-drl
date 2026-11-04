"""Regression tests for the exact author-uploaded MATLAB battery model.

Tests validate import and one short physics transition. They are NOT evidence
that historical SAC checkpoints or PDF-reported results were reproduced.
"""

import hashlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from battery_smart_charging.physics import ElectricalModel, LithiumIonPack

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'PANmodel.mat'
EXPECTED_SHA256 = 'b90711c02e33bd7a2afbec57202e51f09f7d4ecddaaa2ffe2094736a81b39c2b'
REQUIRED = {
    'temps', 'etaParam', 'QParam', 'GParam', 'M0Param', 'MParam',
    'R0Param', 'RCParam', 'RParam', 'SOC', 'OCV0', 'OCVrel',
    'OCV', 'SOC0', 'SOCrel',
}


def test_author_model_schema_and_integrity():
    assert MODEL_PATH.is_file(), 'Original author-uploaded model is missing'
    assert hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() == EXPECTED_SHA256
    data = loadmat(MODEL_PATH, simplify_cells=True)
    assert 'model' in data
    model = data['model']
    assert REQUIRED.issubset(model)
    temps = np.asarray(model['temps'])
    assert temps.shape == (8,)
    assert np.all(np.diff(temps) > 0)
    for key in ('etaParam', 'QParam', 'GParam', 'M0Param', 'MParam', 'R0Param', 'RCParam', 'RParam'):
        assert np.asarray(model[key]).shape == temps.shape
        assert np.all(np.isfinite(model[key]))
    for key in ('SOC', 'OCV0', 'OCVrel'):
        assert np.asarray(model[key]).shape == (201,)
    assert np.all(np.diff(model['SOC']) > 0)


def test_author_model_loaded_by_repository_code():
    ecm = ElectricalModel(MODEL_PATH, number_of_cells=4, T=25, dt=1)
    assert np.isfinite(np.asarray(ecm.QParam)).all()
    pack = LithiumIonPack(MODEL_PATH, number_of_cells=4, T=25, dt=1, different_params=False)
    pack.ECM.z_k[:] = 0.4
    voltage = pack.updateModel(-5 * np.ones(4), Tf=25)
    assert voltage.shape == (1, 4)
    assert np.all(np.isfinite(voltage))
    assert np.all(np.isfinite(pack.ThM.Tc))
    assert np.all(pack.ECM.z_k > 0.4)
