---
title: 'RUBIX: Fast and differentiable modeling of IFU data'
tags:
  - python
  - astronomy
  - IFU
  - galaxies
  - JAX
authors:
  - name: Anna Lena Schaible
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Tobias Buck
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
date: 4 Juli 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Integral field unit (IFU) observations combine imaging and spectroscopy, delivering highly information‑rich data products in astronomy. As more facilities are equipped with IFU instruments, observing a galaxy with an IFU allows us to study spatially resolved spectra and gain a more detailed understanding of galaxy evolution.

In addition to observations, we study galaxy evolution theoretically via cosmological hydrodynamical simulations. However, simulation outputs—particles with physical properties—cannot be directly compared to observations, which record stellar light.

To bridge the simulation and observation domains, we need software that translates between them. `RUBIX` is an open‑source python package that generates realistic mock IFU data cubes from theoretical models such as cosmological hydrodynamical simulations of galaxies or any particle distribution. It provides a fully modular, configurable pipeline that—from arbitrary simulation outputs—produces science‑ready mock observations. Written in JAX for just‑in‑time compilation and GPU support, `RUBIX` performs forward modelling of stellar particles into spatially resolved integrated spectra in seconds rather than hours which previous packages needed, and computes end‑to‑end gradients via automatic differentiation. This differentiable pipeline enables gradient‑based inverse modelling and optimization workflows, opening the door to simulation‑based inference and machine‑learning applications on IFU data.

# Statement of need

The era of large integral field unit (IFU) surveys like MaNGA, SAMI, and upcoming JWST programs has created an urgent need for scalable forward-modeling tools. Current challenges include:

**Computational bottlenecks**: Existing tools like SimSpin [@Harborne2020;@Harborne2023], MaNGIA [@Sarmiento2023] or GalCraft [@Wang2024] require 30 minutes to hours per galaxy on CPUs, making large mock surveys (>10,000 galaxies) computationally prohibitive. This limits statistical studies and prevents adequate exploration of systematic uncertainties.

**Inverse modeling limitations**: Traditional forward-modeling codes lack differentiability, preventing gradient-based parameter inference. This forces researchers to rely on expensive sampling methods (MCMC) or simplified analytical approximations, limiting the precision of simulation-observation comparisons.

**Reproducibility and extensibility**: Many existing codes are monolithic with hard-coded assumptions, making it difficult to reproduce results, modify dust models, or integrate new physics. This hampers scientific progress and collaboration.

| Feature | `RUBIX` | SimSpin | MaNGIA | GalCraft |
|---------|-------|---------|---------|----------|
| Runtime (typical galaxy) | ~1 minute | ~1 hour | ~30 minutes | ~45 minutes |
| GPU Support | ✓ | ✗ | ✗ | ✗ |
| Differentiable | ✓ | ✗ | ✗ | ✗ |
| Modular Architecture | ✓ | Limited | Limited | Limited |
| Dust Modeling | Multiple laws | Basic | Basic | Basic |

In this table we summarize the bottlenecks and limitations of current existing software compared to `RUBIX`. `RUBIX` addresses these limitations by providing:
- **Computation acceleration**: 100x speedup and GPU support enables large-scale mock surveys
- **Automatic differentiation**: Enables gradient-based inference and ML integration
- **Modular design**: Pure functional architecture ensures reproducibility and easy extension
- **Comprehensive testing**: Full test suite and CI/CD pipeline guarantee reliability

This combination of speed, differentiability, and modularity opens new possibilities for simulation-based inference, machine learning applications, and large-scale statistical studies in galaxy evolution.

# Software description

![Schematic overview of the `RUBIX`softwarre: We hand into the pipeline particle data and a configuration as input. The pipeline splits the data onto different devices and are then linear structured functions. As output of our software we get an IFU CUBE.\label{fig:overview}](rubix_code_overview.png)

`RUBIX` is implemented as a modular Python package built on JAX [@jax2018github] for high-performance numerical computing. In Figure \autoref{fig:overview} we show a schematic overview of our pipeline. `RUBIX`takes as input particle data i.e. from cosmological hydrodynamical simulations and a configuration. In the beginning of the pipeline the particle data are splitt and distributed to the computation devices via `shard_map`. The software follows functional programming principles with pure functions throughout the pipeline, enabling parallelization, and automatic differentiation. To each subset of particles the pipeline functions are applied in a linear way and in the end the data are pulled together from all devices and combined. The final output is then the computed mock IFU cube.

The key features of `RUBIX` are:

**Forward modeling**: `RUBIX` transforms particle-based simulation data into realistic mock IFU observations through:
1. Stellar particle assignment to spatial pixels based on user-defined field of view
2. Age and metallicity-dependent spectral synthesis using stellar population models, i.e. FSPS [@Conroy2009]
3. Dust extinction calculation from gas particle distributions (multiple dust laws availible [@Cardelli1989;@Gordon2023;@Calzetti2000])
4. Instrumental effects application (LSF, PSF, noise, spectral resolution)
5. Output generation in standard astronomical formats (FITS)

**Inverse modeling**: `RUBIX` has a differentiable architecture. The JAX implementation enables automatic differentiation through the entire pipeline, supporting:
- Gradient-based parameter estimation
- Integration with modern ML frameworks (Optax, Flax)
- Simulation-based inference workflows
- Uncertainty quantification via variational methods

# Research applications

So far we are using the forward modeling mode to generate large-scale mock surveys for the GECKOS survey [Fraser-McKelvie2024]. We are also working on the inverse modeling mode and do parameter inference studies using gradient-based optimization. Further plans are generating machine learning training data for spectral analysis pipelines.

# Usage example

```python
from rubix.core.pipeline import RubixPipeline 
import os

# Define configuration
config = {...}

# Create mock observation
pipe = RubixPipeline(config)
inputdata = pipe.prepare_data()
rubixdata = pipe.run_sharded(inputdata)
```

# Acknowledgements

We acknowledge the Scientific Software Center at Heidelberg University for code consulting.
This project was made possible by funding from the Carl Zeiss Stiftung.

# References
