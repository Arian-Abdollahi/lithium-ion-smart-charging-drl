import numpy as np

from battery_smart_charging.physics import (
    ElectricalModel, LithiumIonPack, LumpedModel, make_OCVfromSOCtemp,
)


def test_ocv_interpolation(synthetic_model_data):
    f = make_OCVfromSOCtemp(synthetic_model_data, 25)
    assert np.isclose(f(0.5), 3.6)


def test_original_state_equations_work_with_synthetic_parameters(synthetic_model_data):
    ecm = ElectricalModel(None, 4, 25, 1, model_data=synthetic_model_data)
    ecm.z_k = np.full((1, 4), 0.4)
    before = ecm.z_k.copy()
    new_state = ecm.stateEqn(np.full(4, -1.0))
    assert all(a.shape == (1, 4) for a in new_state)
    assert (new_state[2] > before).all()  # Negative original current means charge.
    ecm.updateState(new_state)
    assert np.all(np.isfinite(ecm.outputEqn(np.full(4, -1.0))))


def test_original_thermal_model_has_finite_update():
    thermal = LumpedModel(4, 25, 1)
    thermal.updateTemp(np.ones((1, 4)), 20)
    assert thermal.Tc.shape == (1, 4)
    assert np.all(np.isfinite(thermal.Tc))


def test_notebook_derived_battery_pack(synthetic_model_data):
    pack = LithiumIonPack(None, 4, 25, 1, different_params=False, model_data=synthetic_model_data)
    pack.ECM.z_k[:] = .4
    voltages = pack.updateModel(-1 * np.ones(4), 20)
    assert voltages.shape == (1, 4)
    assert pack.ECM.z_k.min() > .4
