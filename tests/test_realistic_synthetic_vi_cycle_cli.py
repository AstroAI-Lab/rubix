from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_realistic_synthetic_vi_cycle.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_realistic_synthetic_vi_cycle", script
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sample_particle_ics_fallback_shapes() -> None:
    mod = _load_module()
    ics, sampler = mod.sample_particle_ics(n_particles=12, seed=0, prefer_agama=False)

    assert sampler == "fallback_disk"
    assert ics.coords_xy.shape == (12, 2)
    assert ics.velocity_xyz.shape == (12, 3)
    assert ics.mass.shape == (12,)
    assert ics.age.shape == (12,)
    assert ics.metallicity.shape == (12,)


def test_coords_to_spaxels_range() -> None:
    mod = _load_module()
    ics, _ = mod.sample_particle_ics(n_particles=20, seed=1, prefer_agama=False)
    spaxels = mod._coords_to_spaxels(ics.coords_xy, nx=8, ny=10)

    assert spaxels.shape == (20, 2)
    assert int(spaxels[:, 0].min()) >= 0
    assert int(spaxels[:, 0].max()) < 8
    assert int(spaxels[:, 1].min()) >= 0
    assert int(spaxels[:, 1].max()) < 10
