---
title: 'RUBIX: Fast and differentiable modeling of IFU data'
tags:
  - python
  - JAX
  - astronomy
  - IFU
  - galaxies
authors:
  - name: Anna Lena Schaible
    corresponding: true # (This is how to denote the corresponding author)
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Ufuk Cakir
    equal-contrib: true
    affiliation: 3
  - name: Tobias Buck
    affiliation: "1, 2"
  - name: Harald Mack
    affiliation: 1
affiliations:
 - name: Interdisciplinary Center for Scientific Computing (IWR), University of Heidelberg, Im Neuenheimer Feld 205, D-69120 Heidelberg, Germany
   index: 1
 - name: Universität Heidelberg, Zentrum für Astronomie, Institut für Theoretische Astrophysik, Albert-Ueberle-Straße 2, D-69120 Heidelberg, Germany
   index: 2
 - name: Intelligent Earth UKRI Centre for Doctoral Training in AI for the Environment, University of Oxford, UK
   index: 3
date: 19 December 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Integral field unit (IFU) spectroscopy provides spatially resolved spectral data that are central to modern studies of galaxy formation and evolution. Interpreting such data increasingly relies on forward modeling, in which theoretical models or simulations are translated into the observational domain. However, existing IFU forward-modeling tools are typically computationally expensive, CPU-bound, and non-differentiable, limiting their applicability to large surveys and preventing their use in modern gradient-based inference and machine-learning workflows.

`RUBIX` is an open-source Python package for fast and fully differentiable forward modeling of IFU data from particle-based galaxy models. Implemented entirely in `JAX`, `RUBIX` leverages just-in-time compilation, automatic vectorization, and native GPU/TPU support to generate realistic mock IFU data cubes in seconds. Its end-to-end differentiable architecture enables gradient-based optimization, variational inference, and seamless coupling to machine-learning models, supporting both forward and inverse modeling within a single framework. `RUBIX` is designed as a modular, functional pipeline, promoting reproducibility, extensibility, and integration into simulation-based inference workflows for IFU surveys.

# Statement of need

Large IFU surveys such as CALIFA, MaNGA, SAMI, GECKOS, and current and upcoming programs with instruments like MUSE and JWST/NIRSpec are producing vast, information-rich datasets that demand scalable and flexible analysis methods. Forward modeling enables direct, apples-to-apples comparisons between theoretical models and data, but existing IFU forward-modeling tools are limited in several important ways.

First, **computational performance remains a major bottleneck**: widely used packages for generating mock IFU observations from simulations often require tens of minutes to hours per galaxy on CPUs, making large mock surveys or extensive parameter studies impractical. Second, these tools are generally non-differentiable, which precludes efficient gradient-based optimization and inference. As a result, there are **limitations to inverse modelling** which must rely on expensive sampling methods or simplified approximations, limiting the scope and precision of simulation–observation comparisons. Third, many existing codes are monolithic and difficult to extend, with hard-coded modeling assumptions that hinder **reproducibility and extensibility** through exploration of alternative physical models.

`RUBIX` addresses these limitations by providing a fast, GPU-accelerated, and fully differentiable IFU forward-modeling pipeline. Built on `JAX`, `RUBIX` enables end-to-end automatic differentiation through all stages of the modeling process (from particle-based inputs to science-ready IFU data cubes), allowing gradient-based parameter estimation, variational inference, and integration with modern machine-learning techniques. Its modular, functional architecture facilitates reproducible workflows and straightforward integration of alternative spectral models, dust prescriptions, and instrument effects. Together, these features make `RUBIX` a practical foundation for large-scale mock surveys, simulation-based inference, and machine-learning applications in IFU astronomy. \autoref{comparison} summarizes key differences between `RUBIX` and commonly used IFU forward-modeling tools.

: Comparison between `RUBIX` and existing software. `RUBIX` addresses many limitations of existing software by providing: computation acceleration, automatic differentiation, modular design and comprehensive testing. []{label="comparison"}

| Feature | `RUBIX` | SimSpin | MaNGIA | GalCraft |
|---------|-------|---------|---------|----------|
| Runtime (typical galaxy) | ~ 1 minute | ~ 1 hour | ~ 30 minutes | ~ 1.4 hours |
| GPU Support | yes | no | no | no |
| Differentiable | yes | no | no | no |
| Modular Architecture | yes | Limited | Limited | Limited |
| Dust Modeling | Multiple laws | Basic | Basic | Basic |

# Software description

![Schematic overview of the `RUBIX` software: We hand into the pipeline particle data and a configuration as input. The pipeline splits the data onto different devices. The pipeline itself consists of functiona that are applieds in a linear way. As output of our software we get an IFU cube‚.\label{fig:overview}](rubix_code_overview.png)


`RUBIX` is implemented as a modular Python package built entirely on the `JAX` [@jax2018github] high-performance numerical computing framework. The software follows a functional design paradigm, expressing the forward-modeling pipeline as a composition of pure functions operating on particle-based inputs and user-defined configuration parameters. This design enables just-in-time compilation, automatic vectorization, and parallel execution across CPUs, GPUs, and TPUs without requiring user-managed low-level code.

The **forward-modeling pipeline** transforms particle data from theoretical galaxy models or cosmological hydrodynamical simulations into realistic mock IFU data cubes. Key stages include spatial binning of stellar particles, spectral synthesis based on particle age and metallicity (e.g. via FSPS [@Conroy2009]), dust attenuation derived from gas distributions (with multiple dust laws available [@Cardelli1989;@Gordon2023;@Calzetti2000]), and the application of instrumental effects such as spectral resolution, point-spread functions, and noise models. The final outputs are science-ready IFU cubes in standard astronomical formats (FITS; see \autoref{fig:overview}).

All stages of the pipeline are differentiable to enable **inverse modelling**. `JAX`’s automatic differentiation enables the computation of exact gradients with respect to both physical and nuisance parameters, allowing `RUBIX` to support gradient-based optimization, variational inference, and integration with modern machine-learning libraries. This key feature distinguishes `RUBIX` from traditional IFU forward-modeling tools and allows forward and inverse modeling to be performed consistently within a single framework.Comprehensive documentation, testing, and continuous integration support reproducible and extensible research workflows.

# Related work

Several existing software packages perform forward modeling of IFU observations from simulations, including SimSpin [@Harborne2020;@Harborne2023], MaNGIA [@Sarmiento2023], GalCraft [@Wang2024] or Synthesizer [Lovell2025Synthesizer,Roper2025synthesizer]. These tools enable the generation of realistic mock data but are typically CPU-bound and non-differentiable, limiting their scalability and applicability to modern inference and machine-learning approaches.
More general forward-modeling frameworks, such as Synthesizer, provide flexible and efficient generation of synthetic observables from theoretical galaxy models, with a strong emphasis on modularity and performance. `RUBIX` complements these approaches by focusing specifically on IFU data and by providing a fully differentiable pipeline designed for gradient-based inverse modeling.

`RUBIX` is conceptually closest to recent `JAX`-based modeling frameworks such as scarlet2 [scarlet2], which enable differentiable scene modeling for astronomical imaging. While scarlet2 targets pixel-level modeling of imaging data, `RUBIX` addresses the distinct challenges posed by IFU spectroscopy, including the combination of spatial and spectral information and the forward modeling of particle-based galaxy simulations. By extending differentiable modeling concepts to IFU data, `RUBIX` fills a methodological gap between traditional forward-modeling codes and modern machine-learning–driven inference frameworks.

# Research applications

`RUBIX` has been presented at the Machine Learning and the Physical Sciences workshop at NeurIPS 2024 [@Cakir2024] and at the 1st Workshop on Differentiable Systems and Scientific Machine Learning at EurIPS 2025. The software is currently used to generate large-scale mock IFU surveys for GECKOS [@Fraser-McKelvie2024] and to perform gradient-based inverse modelling of the data cubes.


# Acknowledgements

We acknowledge the Scientific Software Center at Heidelberg University [(SSC)](https://www.ssc.uni-heidelberg.de/en) for code consulting.
This project was made possible by funding from the Carl Zeiss Stiftung.
The authors acknowledge usage of the AI clusters \textit{Tom} and \textit{Jerry}, funded by Field of Focus~2 of Heidelberg University.

# References
