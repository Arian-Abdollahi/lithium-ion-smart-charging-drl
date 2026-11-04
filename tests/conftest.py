"""Synthetic values for unit tests only: NEVER thesis battery parameters."""

import pytest


@pytest.fixture
def synthetic_model_data():
    temps = [0.0, 25.0, 50.0]
    curve = [3.0, 3.3, 3.6, 3.85, 4.1]
    return {
        'temps': temps,
        'etaParam': [1.0] * 3, 'QParam': [2.3] * 3,
        'GParam': [0.1] * 3, 'M0Param': [0.0] * 3,
        'MParam': [0.01] * 3, 'R0Param': [0.01] * 3,
        'RCParam': [2.0] * 3, 'RParam': [0.02] * 3,
        'SOC': [0.0, .25, .5, .75, 1.0],
        'OCV0': curve, 'OCVrel': [0.0] * len(curve),
        'OCV': curve, 'SOC0': [0.0, .25, .5, .75, 1.0],
        'SOCrel': [0.0] * len(curve),
    }
