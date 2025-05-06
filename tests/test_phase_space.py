import pytest
import jax
import jax.numpy as jnp
from rubix.core.galactic_dynamics.actions_to_phase_space import (
    actions_to_phase_space,
    map_actions_to_phase_space,
)
import numpy as np
from rubix.core.galactic_dynamics.sampling import sample_df_potential
from rubix.core.galactic_dynamics.distribution_functions import df_total_potential

@pytest.fixture
def params():
    return {
        "R0": 8.0,
        "Rd": 2.5,
        "v0": 220.0,
        "L0": 10.0,
        "tau_m": 10.0,
        "tau1": 0.1,
        "beta": 0.33,
        "t0": 8.0,
        "n_age_bins": 10,
        "sigma_r0": 33.5,
        "sigma_z0": 19.0,
        "Rd_thick": 2.3,
        "L0_thick": 10.0,
        "sigma_r0_thick": 60.0,
        "sigma_z0_thick": 32.0,
        "frac_thick": 0.24,
        "Z_max": 0.03,
        "Z_min": 0.005,
    }

KEY = jax.random.PRNGKey(0)

@ pytest.mark.parametrize("Jr, Jz, Lz", [
    (0.0, 0.0, 0.0),     
    (10.0, 5.0, 500.0),
    (0.1, 0.1, 100.0),
])
def test_single_action_finite(Jr, Jz, Lz, params):
    """
    Test that the action->phase-space mapping returns finite values and correct shape.
    """
    coords = actions_to_phase_space(Jr, Jz, Lz, params, KEY)
    #should return a tuple of 6 floats
    assert isinstance(coords, tuple)
    assert len(coords) == 6
    arr = jnp.stack(coords)
    assert arr.shape == (6,)
    #all entries finite
    assert jnp.isfinite(arr).all()

@ pytest.fixture
def sample_actions():
    #generate 100 random action triplets
    key = jax.random.PRNGKey(1)
    subk1, subk2, subk3 = jax.random.split(key, 3)
    Jr = jax.random.uniform(subk1, (100,), minval=0.0, maxval=50.0)
    Jz = jax.random.uniform(subk2, (100,), minval=0.0, maxval=20.0)
    Lz = jax.random.uniform(subk3, (100,), minval=0.0, maxval=1000.0)
    return jnp.stack([Jr, Jz, Lz], axis=1)


def test_vectorized_mapping(sample_actions, params):
    """
    Test batch mapping does not produce invalid values and correct shape.
    """
    key = jax.random.PRNGKey(2)
    coords = map_actions_to_phase_space(sample_actions, params, key)
    #coords should be an array (N,6)
    assert isinstance(coords, jnp.ndarray)
    assert coords.ndim == 2
    assert coords.shape == (100, 6)
    #no NaNs or infs
    assert jnp.isfinite(coords).all()



@pytest.mark.parametrize("n_samples, envelope_max, corr_thresh", [
    (20000, 1.0, 0.6),
])
def test_Lz_marginal_distribution(n_samples, envelope_max, corr_thresh, params):
    """
    Smoke test: Marginal Lz distribution from the soft acceptance sampling
    vs. rough theory distribution DF(Jr=0,Jz=0,Lz).
    We expect a positive Pearson correlation > corr_thresh.
    """
    key = jax.random.PRNGKey(123)
    cands, _ = sample_df_potential(key, params,
                                   n_candidates=n_samples,
                                   envelope_max=envelope_max,
                                   tau=0.01)
    Lz = np.asarray(cands[:,2])
    #histogram
    bins = 25
    hist, edges = np.histogram(Lz, bins=bins, range=(0, np.max(Lz)), density=True)
    centers = 0.5*(edges[:-1] + edges[1:])
    #theoretical distribution at Jr=0, Jz=0
    theo = np.array([df_total_potential(0.0, 0.0, L, params) for L in centers])
    #normalize
    hist /= hist.max()
    theo /= theo.max()
    #pearson correlation
    corr = np.corrcoef(hist, theo)[0,1]
    assert corr > corr_thresh, f"correlation too low: {corr:.2f}"


#The following tests measure that the maximum is reached for the theoretical and the calculated bin0, i.e. not the correlation as with Lz
@pytest.mark.parametrize("n_samples, envelope_max", [
    (20000, 1.0),
])
def test_Jz_marginal_peak(n_samples, envelope_max, params):
    """
    Smoke test: The marginal Jz distribution from soft-acceptance sampling
    should have its maximum in bin 0 (lowest Jz).
    """
    key = jax.random.PRNGKey(456) 
    cands, _ = sample_df_potential(key, params, 
                                   n_candidates=n_samples,
                                   envelope_max=envelope_max,
                                   tau=0.01) 
    Jz = np.asarray(cands[:,1])
    hist, edges = np.histogram(Jz,
                               bins=25,
                               range=(0, Jz.max()),
                               density=True)
    idx_peak = np.argmax(hist)
    assert idx_peak == 0, f"Jz Distribution peaks in bin {idx_peak}, expected bin 0"

@pytest.mark.parametrize("n_samples, envelope_max", [
    (20000, 1.0),
])
def test_Jr_marginal_peak(n_samples, envelope_max, params):
    """
    Smoke test: The marginal Jr distribution from soft-acceptance sampling
    should have its maximum in bin 0 (lowest Jr).
    """
    key = jax.random.PRNGKey(789)
    cands, _ = sample_df_potential(key, params,
                                   n_candidates=n_samples,
                                   envelope_max=envelope_max,
                                   tau=0.01)
    Jr = np.asarray(cands[:,0])
    hist, edges = np.histogram(Jr,
                               bins=25,
                               range=(0, Jr.max()),
                               density=True)
    idx_peak = np.argmax(hist)
    assert idx_peak == 0, f"Jr Distribution peaks in bin {idx_peak}, expected bin 0"


@pytest.mark.parametrize("Jr,Jz,Lz", [
    (1.0, 0.5, 50.0),
    (10.0, 5.0, 500.0),
])
def test_df_total_potential_gradients(Jr, Jz, Lz, params):
    """
    Test: the gradients of df_total_potential to Jr, Jz and Lz are finite.
    """
    #derive individually
    g_Jr = jax.grad(lambda Jr_: df_total_potential(Jr_, Jz, Lz, params))(Jr)
    g_Jz = jax.grad(lambda Jz_: df_total_potential(Jr, Jz_, Lz, params))(Jz)
    g_Lz = jax.grad(lambda Lz_: df_total_potential(Jr, Jz, Lz_, params))(Lz)

    for val in (g_Jr, g_Jz, g_Lz):
        assert np.isfinite(val), f"Gradient does not contain a finite value: {val}"


from rubix.core.galactic_dynamics.sampling import soft_acceptance, sample_df_potential
from rubix.core.galactic_dynamics.actions_to_phase_space import actions_to_phase_space, map_actions_to_phase_space

@pytest.mark.parametrize("n_vals", [5, 10])
def test_soft_acceptance_gradients(n_vals):
    """
    Test: soft_acceptance(df_vals, rand_vals, envelope_max, tau) is differentiable
    and the derivatives are all finite.
    """
    #dummy data
    df_vals   = jnp.linspace(0.1, 1.0, n_vals)
    rand_vals = jnp.linspace(0.2, 0.8, n_vals)
    envelope_max = 1.0
    tau = 0.05

    #∂/∂df_vals
    g_df = jax.grad(lambda x: jnp.sum(soft_acceptance(x, rand_vals, envelope_max, tau)))(df_vals)
    #∂/∂rand_vals
    g_rand = jax.grad(lambda x: jnp.sum(soft_acceptance(df_vals, x, envelope_max, tau)))(rand_vals)
    #∂/∂envelope_max
    g_env = jax.grad(lambda x: jnp.sum(soft_acceptance(df_vals, rand_vals, x, tau)))(envelope_max)
    #∂/∂tau
    g_tau = jax.grad(lambda x: jnp.sum(soft_acceptance(df_vals, rand_vals, envelope_max, x)))(tau)

    assert jnp.isfinite(g_df).all()
    assert jnp.isfinite(g_rand).all()
    assert jnp.isfinite(g_env), f"env-gradient is not finite: {g_env}"
    assert jnp.isfinite(g_tau), f"tau-gradient is not finite: {g_tau}"

@pytest.mark.parametrize("Jr,Jz,Lz", [
    (2.0, 1.0, 100.0),
    (5.0, 0.5, 300.0),
])
def test_actions_to_phase_space_jacobian(params, Jr, Jz, Lz):
    """
    Test: The Jacobian matrix of the Action to Phase-Space-Mapping is finite.
    """
    key = jax.random.PRNGKey(42)
    def f(cand):
        #cand: [Jr, Jz, Lz]
        return jnp.stack(actions_to_phase_space(cand[0], cand[1], cand[2], params, key))
    J = jax.jacfwd(f)(jnp.array([Jr, Jz, Lz]))
    assert J.shape == (6, 3)
    assert jnp.isfinite(J).all(), "Jacobi-Matrix contains NaN/Inf"