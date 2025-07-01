---
title: 'RUBIX: Fast and differentiable modeling of IFU data'
tags:
  - Python
  - astronomy
  - IFU
  - galaxies
  - JAX
authors:
  - name: Anna Lena Schaible
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Tobias Buck
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: "1, 2"
  - name: Ufuk Cakir
    affiliation: 3
  - name: Harlad Mack
    affiliation: 1
affiliations:
 - name: Interdisciplinary Center for Scientific Computing (IWR), University of Heidelberg, Im Neuenheimer Feld 205, D-69120 Heidelberg, Germany
   index: 1
 - name: Universität Heidelberg, Zentrum für Astronomie, Institut für Theoretische Astrophysik, Albert-Ueberle-Straße 2, D-69120 Heidelberg, Germany
   index: 2
 - name: Intelligent Earth UKRI Centre for Doctoral Training in AI for the Environment, University of Oxford, UK
   index: 3
date: 1 Juli 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Integral field unit (IFU) observations combine imaging and spectroscopy, delivering highly information‑rich data products in astronomy. As more facilities are equipped with IFU instruments, observing a galaxy with an IFU allows us to study spatially resolved spectra and gain a more detailed understanding of galaxy evolution.

In addition to observations, we study galaxy evolution theoretically via cosmological hydrodynamical simulations. However, simulation outputs—particles with physical properties—cannot be directly compared to observations, which record stellar light.

To bridge the simulation and observation domains, we need software that translates between them. RUBIX is an open‑source Python package that generates realistic mock IFU data cubes from theoretical models such as cosmological hydrodynamical simulations of galaxies or any particle distribution. It provides a fully modular, configurable pipeline that—from arbitrary simulation outputs—produces science‑ready mock observations. Written in JAX for just‑in‑time compilation and GPU acceleration, RUBIX performs forward modelling of stellar particles into spatially resolved integrated spectra in seconds rather than hours which previous packages needed, and computes end‑to‑end gradients via automatic differentiation. This differentiable pipeline enables gradient‑based inverse modelling and optimization workflows, opening the door to simulation‑based inference and machine‑learning applications on IFU data.

# Statement of need

The era of large integral field unit (IFU) surveys like MaNGA, SAMI, and upcoming JWST programs has created an urgent need for scalable forward-modeling tools. Current challenges include:

**Computational Bottlenecks**: Existing tools like SimSpin [@Harborne2020;@Harborne2023], MaNGIA [@Sarmiento2023] or GalCraft [@Wang2024] require 30 minutes to hours per galaxy on CPUs, making large mock surveys (>10,000 galaxies) computationally prohibitive. This limits statistical studies and prevents adequate exploration of systematic uncertainties.

**Inverse Modeling Limitations**: Traditional forward-modeling codes lack differentiability, preventing gradient-based parameter inference. This forces researchers to rely on expensive sampling methods (MCMC) or simplified analytical approximations, limiting the precision of simulation-observation comparisons.

**Reproducibility and Extensibility**: Many existing codes are monolithic with hard-coded assumptions, making it difficult to reproduce results, modify dust models, or integrate new physics. This hampers scientific progress and collaboration.

`RUBIX` addresses these limitations by providing:
- **GPU acceleration**: 100-1000x speedup enables large-scale mock surveys
- **Automatic differentiation**: Enables gradient-based inference and ML integration
- **Modular design**: Pure functional architecture ensures reproducibility and easy extension
- **Comprehensive testing**: Full test suite and CI/CD pipeline guarantee reliability

This combination of speed, differentiability, and modularity opens new possibilities for simulation-based inference, machine learning applications, and large-scale statistical studies in galaxy evolution.

# Software Description

`RUBIX` is implemented as a modular Python package built on JAX [@jax2018github] for high-performance numerical computing. The core architecture consists of:

- **Particle Processing**: Efficient handling of simulation particle data with support for multiple formats (HDF5, Gadget, tipsy, etc.)
- **Stellar Population Synthesis**: Integration with FSPS [@Conroy2009] for generating stellar spectra from stellar particles
- **Dust Modeling**: Comprehensive extinction implementation supporting multiple dust laws (Cardelli89, Gordon23, Calzetti00)
- **Instrumental Effects**: Configurable PSF convolution, noise modeling, and spectral line spread functions
- **Data Cube Generation**: Production of science-ready FITS cubes matching observational formats

The software follows functional programming principles with pure functions throughout the pipeline, enabling reliable caching, parallelization, and automatic differentiation.

# Key Features and Functionality

## Forward Modeling Pipeline
`RUBIX` transforms particle-based simulation data into realistic mock IFU observations through:
1. Stellar particle assignment to spatial pixels based on user-defined field of view
2. Age and metallicity-dependent spectral synthesis using stellar population models
3. Dust extinction calculation from gas particle distributions
4. Instrumental effects application (PSF, noise, spectral resolution)
5. Output generation in standard astronomical formats (FITS)

## Differentiable Architecture
The JAX implementation enables automatic differentiation through the entire pipeline, supporting:
- Gradient-based parameter estimation
- Integration with modern ML frameworks (Optax, Flax)
- Simulation-based inference workflows
- Uncertainty quantification via variational methods

## Performance and Scalability
Benchmarks demonstrate:
- 100-1000x speedup over CPU-based alternatives
- Sub-minute processing for typical galaxy simulations
- Linear scaling with particle count on GPUs????
- Memory-efficient streaming for large datasets

# Comparison with Existing Software

| Feature | `RUBIX` | SimSpin | MaNGIA | GalCraft |
|---------|-------|---------|---------|----------|
| Runtime (typical galaxy) | ~1 minute | ~1 hour | ~30 minutes | ~45 minutes |
| GPU Support | ✓ | ✗ | ✗ | ✗ |
| Differentiable | ✓ | ✗ | ✗ | ✗ |
| Modular Architecture | ✓ | Limited | Limited | Limited |
| Dust Modeling | Multiple laws | Basic | Basic | Basic |

# Research Applications

`RUBIX` has been successfully applied to:
- Large-scale mock survey generation for the MUSE Geckos survey
- Parameter inference studies using gradient-based optimization
- Machine learning training data generation for spectral analysis pipelines
- Systematic uncertainty quantification in simulation-observation comparisons???

# Usage Example

```python
import rubix
from rubix.config import RubixConfig

# Load configuration
config = RubixConfig.from_yaml("config.yaml")

# Create mock observation
mock_cube = rubix.forward_model(
    particle_data="simulation.hdf5",
    config=config
)

# Gradient-based parameter estimation
def loss_fn(params):
    pred_cube = rubix.forward_model(particle_data, params)
    return jnp.mean((pred_cube - observed_cube)**2)

grad_fn = jax.grad(loss_fn)
gradients = grad_fn(initial_params)
```

# Acknowledgements

We acknowledge the Scientific Software Center at Heidelberg University for code consulting.
This project was made possible by funding from the Carl Zeiss Stiftung.

# References
