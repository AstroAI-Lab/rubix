import jax.numpy as jnp

from rubix.core.ifu import _get_performance_options, _scan_particles


def test_get_performance_options_defaults():
    chunk_size, use_remat = _get_performance_options({})

    assert chunk_size == 0
    assert use_remat is False


def test_get_performance_options_reads_valid_values():
    chunk_size, use_remat = _get_performance_options(
        {"performance": {"particle_chunk_size": 16, "remat_particlewise": True}}
    )

    assert chunk_size == 16
    assert use_remat is True


def test_get_performance_options_rejects_invalid_chunk_size():
    chunk_size, use_remat = _get_performance_options(
        {"performance": {"particle_chunk_size": -5, "remat_particlewise": False}}
    )

    assert chunk_size == 0
    assert use_remat is False


def _simple_step(cube, i):
    update = jnp.full_like(cube, i + 1, dtype=cube.dtype)
    return cube + update, None


def test_scan_particles_chunked_matches_unchunked():
    init_cube = jnp.zeros((3, 4), dtype=jnp.float32)
    nstar = 11

    unchunked = _scan_particles(
        init_cube=init_cube,
        nstar=nstar,
        step_fn=_simple_step,
        chunk_size=0,
    )
    chunked = _scan_particles(
        init_cube=init_cube,
        nstar=nstar,
        step_fn=_simple_step,
        chunk_size=4,
    )

    assert jnp.allclose(unchunked, chunked)


def test_scan_particles_returns_init_for_empty_input():
    init_cube = jnp.ones((2, 2), dtype=jnp.float32)

    result = _scan_particles(
        init_cube=init_cube,
        nstar=0,
        step_fn=_simple_step,
        chunk_size=8,
    )

    assert jnp.allclose(result, init_cube)
