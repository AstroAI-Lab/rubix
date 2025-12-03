"""
This module contains the supported templates for the SSP grid.

Example:

>>> from rubix.spectra.ssp.templates import BruzualCharlot2003
>>> BruzualCharlot2003
>>> print(BruzualCharlot2003.age)
"""

from .factory import get_ssp_template

BruzualCharlot2003 = get_ssp_template("BruzualCharlot2003")

# having this here forces a dwonload of the Mastar data
# MaStar_CB19_SLOG_1_5 = get_ssp_template("Mastar_CB19_SLOG_1_5")

__all__ = ["BruzualCharlot2003"]  # , "Mastar_CB19_SLOG_1_5"]
