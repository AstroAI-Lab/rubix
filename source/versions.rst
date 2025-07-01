Code versions
============

`RUBIX` has different code versions. The current version is `0.1`.

Version 0.1
-----------
Forwardmodel IFU cubes of galaxies from cosmological hydrodynamical simulations for stellar particles from IllustrisTNG50 and NIHAO.
This version includes the following features:
- Generate mock IFU flux cubes for stars from IllustrisTNG and NIHAO
- Generate mock photometric images for stars for different filter curves
- Use different stellar population synthesis models (FSPS, Bruzual & Charlot, MaStar)
- Use MUSE as telescope instrument (and some other instruments)
- Simple dust attenuation model
- Computation on multiple CPUs or GPUs via JAX shard_map
- Calculate the gradient with respect to the physical properties of the input particles through the whole pipeline
