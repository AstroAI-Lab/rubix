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

INtegral field unit (IFU) observations combine imaging and spectroscopy, delivering highly information‑rich data products in astronomy. As more facilities are equipped with IFU instruments, observing a galaxy with an IFU allows us to study spatially resolved spectra and gain a more detailed understanding of galaxy evolution.

In addition to observations, we study galaxy evolution theoretically via cosmological hydrodynamical simulations. However, simulation outputs—particles with physical properties—cannot be directly compared to observations, which record integrated light.

To bridge the simulation and observation domains, we need software that translates between them. RUBIX is an open‑source Python package that generates realistic mock IFU data cubes from cosmological hydrodynamical simulations of galaxies or any particle distribution. It provides a fully modular, configurable pipeline that—from arbitrary simulation outputs—produces science‑ready mock observations. Written in JAX for just‑in‑time compilation and GPU acceleration, RUBIX performs forward modelling of stellar particles into spatially resolved integrated spectra in seconds rather than hours, and computes end‑to‑end gradients via automatic differentiation. This differentiable pipeline enables gradient‑based inverse modelling and optimization workflows, opening the door to simulation‑based inference and machine‑learning applications on IFU data.

# Statement of need

Forward‑modelling tools such as SimSpin [Harborne2020;Harborne2023], MaNGIA [Sarmiento2023], and GalCraft [Wang2024] have advanced our ability to compare simulations with IFU observations, but they typically run on CPUs, require hours per galaxy, and lack built‑in support for inverse modelling. Moreover, many existing codes are monolithic and difficult to extend. `RUBIX` fills this gap by offering:

- High performance: GPU‑accelerated JAX [jax2018github] implementation reduces per‑galaxy runtimes from hours to seconds [Cakir2024], making large mock surveys feasible 
- Differentiability: Automatic gradient computation through the entire pipeline enables gradient‑based parameter inference and integration with modern optimization frameworks 
- Modularity and reproducibility: A clean, linear pipeline of pure functions, configurable via YAML, with comprehensive testing, continuous integration, and documentation ensures ease of extension and reliable, reproducible outputs 

By combining speed, flexibility, and differentiable capabilities, `RUBIX` meets a growing need in the community for scalable, inverse‑capable mock‑observation frameworks. 

# Acknowledgements

We acknowledge the Scientific Software Center at Heidelberg University for code consulting.
This project was made possible by funding from the Carl Zeiss Stiftung.

# References