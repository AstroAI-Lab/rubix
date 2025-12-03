.. rubix documentation master file, created by
   sphinx-quickstart on Thu Oct 10 13:33:35 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RUBIX's documentation!
=================================

RUBIX is a tested and modular Open Source tool developed in JAX, designed to forward model IFU cubes of galaxies from cosmological hydrodynamical simulations.
The code automatically parallelizes computations across multiple GPUs, demonstrating performance
improvements over state-of-the-art codes. For further details see the publications or the documentation of the individual functions.

Currently the following functionalities are provided:

- Generate mock IFU flux cubes for stars from IllustrisTNG50, NIHAO or other cosmological hydrodynamical simulations

- Generate mock photometric images for stars for different filter curves

- Use different stellar population synthesis models (Bruzual & Charlot, Mastar, FSPS, EMILES)

- Use MUSE as telescope instrument (and some other instruments)

- Use different dust attenuation laws

- Calculate gradients of the modelled flux cubes with respect to stellar age and metallicity using JAX automatic differentiation

Currently the code is under development and is not yet all functionality is available.
We are working on adding more features and improving the code, espectially we work on the following features:

- Adding gas emission lines and gas continuum

- Extend gradient calculation to more parameters and scale the gradient calculation to larger data sets

- Sampling from distribution functions

If you are interested in contributing to the code or have ideas for further features, please contact us via a github issue or via email.
If you use the code in your research, please cite the following paper: :ref:`publications`


.. toctree::
   :maxdepth: 1
   :caption: RUBIX documentation:

   self
   installation
   versions
   publications
   license
   acknowledgments

Notebooks
===================

.. toctree::
   :maxdepth: 1
   :caption: Notebooks:

   notebooks/create_rubix_data.ipynb
   notebooks/pipeline_demo.ipynb
   notebooks/rubix_pipeline_single_function_shard_map.ipynb
   notebooks/rubix_pipeline_stepwise.ipynb
   notebooks/dust_extinction.ipynb
   notebooks/gradient_age_metallicity_adamoptimizer_multi.ipynb
   notebooks/gradient_age_metallicity_adamoptimizer_vs_finite_diff.ipynb
   notebooks/cosmology.ipynb
   notebooks/telescope.ipynb
   notebooks/ssp_template.ipynb
   notebooks/ssp_interpolation.ipynb
   notebooks/ssp_template_fsps.ipynb
   notebooks/psf.ipynb
   notebooks/filter_curves.ipynb

Code base documentation
=======================

.. toctree::
   :maxdepth: 3
   :caption: Code documentation:

   rubix.core
   rubix.cosmology
   rubix.galaxy
   rubix.pipeline
   rubix.spectra
   rubix.telescope
   rubix.utils
