import numpy as np


def test_original_cell_interleaving_convention():
    state_grouped = np.array([[.2, .3, .4, .5], [.6, .7, .8, .9], [1.0, 1.1, 1.2, 1.3]])
    flattened = state_grouped.T.reshape(-1)
    recovered = flattened.reshape(4, 3).T
    np.testing.assert_array_equal(recovered, state_grouped)
