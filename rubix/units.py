from astropy import units as u

# Define custom units here
Zsun = u.def_unit("Zsun", u.dimensionless_unscaled)
u.add_enabled_units(Zsun)
