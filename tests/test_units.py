import astropy.units as u

from rubix.units import Zsun


def test_zsun_unit():
    assert str(Zsun) == "Zsun"
    assert u.Unit("Zsun") == Zsun
