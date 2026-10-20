import numpy as np
import pytest

from grstbx.matchup import matchup_metrics


def test_matchup_metrics():
    x = np.array([1.0, 2.0, 4.0, np.nan])
    metrics = matchup_metrics(x, 2 * x)
    assert metrics['N'] == 3
    assert metrics['slope'] == pytest.approx(2)
    assert metrics['r'] == pytest.approx(1)
    assert metrics['mape'] == pytest.approx(100)
    assert metrics['eps'] == pytest.approx(100)    # y / x = 2 everywhere
    assert metrics['beta'] == pytest.approx(100)
    assert matchup_metrics(x, x)['rmse'] == 0
