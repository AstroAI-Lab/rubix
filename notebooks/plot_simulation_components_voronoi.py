"""
Component maps of the NIHAO simulation (g8.26e11), Voronoi-binned with exactly
the same bins as the nGIST analysis of the mock MUSE cube.

How the bins are transferred:
  * nGIST stores the binning per spaxel in <prefix>_table.fits (300x300 = 90000 rows,
    row-major, identical to the BINID image in <prefix>_SPATIAL_BINNING_maps.fits).
    BIN_ID >= 0 is the Voronoi bin; BIN_ID < 0 marks spaxels below the S/N
    threshold, which nGIST does not fit (NaN in the KIN/SFH maps) -> masked here too.
  * Each particle is placed in its spaxel exactly the way rubix does it
    (square_spaxel_assignment with symmetric edges, kpc/arcsec from the rubix
    cosmology), then gets that spaxel's BIN_ID.
  * rubix stores the cube as datacube.T with flat index ix + n*iy, so in the FITS
    image (and the nGIST table) rows are particle x and columns are particle y:
    bin_map[ix, iy]. A sanity check compares the simulated and observed velocity
    fields for both orientations and warns if the transposed one fits better.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit

# --- Rubix Imports ---
from rubix.core.cosmology import get_cosmology
from rubix.core.data import convert_to_rubix, prepare_input
from rubix.core.rotation import get_galaxy_rotation

galaxy_id = "g8.26e11"
components = ["None", "ThickDisc", "ThinDisc", "ClassicalBulge", "PseudoBulge"]
component_names = ["All", "Thick disk", "Thin disk", "Classical bulge", "Disky bulge"]
split_file = "/home/annalena/rubix/notebooks/data/gsf/g8.26e11.01024.halo_1.align_with_star.star.gmm_on_jzjcjpjcexyz.scikit_gmm_full_5clusters_white.decontaminated.dat"

# nGIST output of the mock observation (provides the Voronoi bins)
ngist_dir = "/backup/home2/annalena_data/rubix_methods_paper/ngist_826"
ngist_prefix = "NIHAO_idg8.26e11_redMiles_ultraWFMcutcenter2"

z = 0.01
cosmology_name = "PLANCK15"  # same as in the rubix config -> same spaxel size in kpc

Z_sun = 0.0134


# ---------------------------------------------------------------------------
# Voronoi binning from nGIST
# ---------------------------------------------------------------------------
def load_voronoi_binning(ngist_dir, ngist_prefix):
    """
    Read the nGIST spatial binning.

    Returns:
    --------
    binning : dict
        bin_map   : (n_pix, n_pix) int, BIN_ID per spaxel, indexed [ix, iy]; -1 = not binned
        n_bins    : number of Voronoi bins
        nspax     : (n_bins,) number of spaxels per bin
        pixsize   : spaxel size [arcsec]
        n_pix     : spaxels per side
        obs_v     : (n_pix, n_pix) observed LOS velocity (for the orientation check)
    """
    table_path = os.path.join(ngist_dir, f"{ngist_prefix}_table.fits")
    with fits.open(table_path) as h:
        pixsize = float(h[0].header["PIXSIZE"])
        table = h[1].data
        bin_id = np.array(table["BIN_ID"], dtype=np.int64)

    n_pix = int(round(np.sqrt(len(bin_id))))
    bin_map = bin_id.reshape(n_pix, n_pix)
    bin_map = np.where(bin_map >= 0, bin_map, -1)

    n_bins = int(bin_map.max()) + 1
    nspax = np.bincount(bin_map[bin_map >= 0], minlength=n_bins)

    obs_v = fits.getdata(os.path.join(ngist_dir, f"{ngist_prefix}_KIN_maps.fits"), "V")

    print(f"nGIST binning: {n_pix}x{n_pix} spaxels of {pixsize:.3f}\", "
          f"{n_bins} Voronoi bins, {np.mean(bin_map < 0) * 100:.1f}% spaxels unbinned")
    return dict(bin_map=bin_map, n_bins=n_bins, nspax=nspax, pixsize=pixsize,
                n_pix=n_pix, obs_v=obs_v)


def spaxel_edges_kpc(binning, z, cosmology_name):
    """Symmetric spaxel edges in kpc, as rubix' calculate_spatial_bin_edges."""
    cosmo = get_cosmology({"cosmology": {"name": cosmology_name}})
    kpc_per_arcsec = float(cosmo.angular_scale(z))
    pix_kpc = binning["pixsize"] * kpc_per_arcsec
    half = 0.5 * binning["n_pix"] * pix_kpc
    edges = -half + pix_kpc * np.arange(binning["n_pix"] + 1)
    return edges, pix_kpc


def assign_particles_to_bins(positions, binning, edges, pix_kpc, transpose=False):
    """Voronoi bin index of every particle (-1: outside the FoV or unbinned spaxel)."""
    n_pix = binning["n_pix"]
    ix = np.floor((positions[:, 0] - edges[0]) / pix_kpc).astype(np.int64)
    iy = np.floor((positions[:, 1] - edges[0]) / pix_kpc).astype(np.int64)
    inside = (ix >= 0) & (ix < n_pix) & (iy >= 0) & (iy < n_pix)

    bin_map = binning["bin_map"].T if transpose else binning["bin_map"]
    particle_bin = np.full(len(positions), -1, dtype=np.int64)
    particle_bin[inside] = bin_map[ix[inside], iy[inside]]
    return particle_bin


def check_orientation(positions, velocities, masses, binning, edges, pix_kpc, los_axis=2):
    """
    Correlate the mass-weighted mean LOS velocity per bin with the observed nGIST
    velocity for the assumed and the transposed spaxel orientation.
    """
    bin_map = binning["bin_map"]
    valid = bin_map >= 0
    obs_v_bin = np.full(binning["n_bins"], np.nan)
    obs_v_bin[bin_map[valid]] = binning["obs_v"][valid]

    corr = {}
    for transpose in (False, True):
        b = assign_particles_to_bins(positions, binning, edges, pix_kpc, transpose)
        m = b >= 0
        w = np.bincount(b[m], weights=masses[m], minlength=binning["n_bins"])
        wv = np.bincount(b[m], weights=masses[m] * velocities[m, los_axis], minlength=binning["n_bins"])
        sim_v_bin = np.divide(wv, w, where=w > 0, out=np.full_like(wv, np.nan))
        ok = np.isfinite(sim_v_bin) & np.isfinite(obs_v_bin)
        corr[transpose] = np.corrcoef(sim_v_bin[ok], obs_v_bin[ok])[0, 1]

    print(f"Orientation check: corr(sim v, nGIST v) = {corr[False]:+.3f} (as assumed), "
          f"{corr[True]:+.3f} (transposed)")
    if abs(corr[True]) > abs(corr[False]):
        print("  WARNING: transposed orientation matches the observation better -> "
              "set TRANSPOSE_BINS = True")
    if corr[False] < 0 and abs(corr[False]) >= abs(corr[True]):
        print("  WARNING: velocity field is anti-correlated (sign flip / mirrored axis?)")
    return corr


# ---------------------------------------------------------------------------
# Simulation data
# ---------------------------------------------------------------------------
def load_component_data(component):
    config = {
        "pipeline": {"name": "calc_ifu_memory"},
        "data": {
            "name": "NihaoHandler",
            "args": {"particle_type": ["stars"], "save_data_path": "data", "snapshot": "1024"},
            "load_galaxy_args": {"reuse": True, "id": galaxy_id},
            "subset": {"use_subset": False},
        },
        "simulation": {
            "name": "NIHAO",
            "args": {
                "path": f"/home/_data/nihao/nihao_classic/{galaxy_id}/{galaxy_id}.01024",
                "halo_path": f"/home/_data/nihao/nihao_classic/{galaxy_id}/{galaxy_id}.01024.z0.000.AHF_halos",
                "halo_id": 0,
            },
        },
        "telescope": {
            "name": "MUSE_WFM",
            "psf": {"name": "gaussian", "size": 5, "sigma": 0.6},
            "lsf": {"sigma": 0.5},
            "noise": {"signal_to_noise": 100, "noise_distribution": "normal"},
        },
        "galaxy": {
            "dist_z": z,
            "rotation": {"type": "matrix"},
            "component": component,
            "component_file": split_file,
        },
        "cosmology": {"name": cosmology_name},
        "ssp": {
            "template": {"name": "FSPS"},
            "dust": {
                "extinction_model": "Cardelli89",
                "dust_to_gas_ratio": 0.01,
                "dust_to_metals_ratio": 0.4,
                "dust_grain_density": 3.5,
                "Rv": 3.1,
            },
        },
        "output_path": "output",
    }

    convert_to_rubix(config)
    rubixdata = prepare_input(config)
    rotate = get_galaxy_rotation(config)
    rubixdata = rotate(rubixdata)

    star_coordinates = np.array(rubixdata.stars.coords)
    if star_coordinates.ndim == 3: star_coordinates = star_coordinates[0]

    star_velocities = np.array(rubixdata.stars.velocity)
    if star_velocities.ndim == 3: star_velocities = star_velocities[0]

    star_ages = np.array(rubixdata.stars.age)
    if star_ages.ndim == 2: star_ages = star_ages[0]

    star_metallicity = np.array(rubixdata.stars.metallicity)
    if star_metallicity.ndim == 2: star_metallicity = star_metallicity[0]

    star_masses = np.array(rubixdata.stars.mass)
    if star_masses.ndim == 2: star_masses = star_masses[0]

    return star_coordinates, star_velocities, star_ages, star_metallicity, star_masses


# ---------------------------------------------------------------------------
# LOSVD fit
# ---------------------------------------------------------------------------
def gauss_hermite(v, v_rot, v_disp, h3, h4):
    """Gauss-Hermite expansion for the line-of-sight velocity distribution."""
    y = np.asarray((np.asarray(v) - v_rot) / (v_disp))
    return (np.exp(-0.5 * y**2) / (v_disp * np.sqrt(2 * np.pi))) * (
        1
        + h3 * ((2 * np.sqrt(2) * y**3 - 3 * np.sqrt(2) * y) / np.sqrt(6))
        + h4 * ((4 * y**4 - 12 * y**2 + 3) / np.sqrt(24))
    )


def fit_losvd(vz_bin):
    """Fit a Gauss-Hermite LOSVD; returns [v_rot, v_disp, h3, h4] or NaNs."""
    if len(vz_bin) < 10:
        return np.nan, np.nan, np.nan, np.nan
    hist, bins = np.histogram(vz_bin, bins=30, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    p0 = [np.mean(vz_bin), np.std(vz_bin), 0.1, 0.1]
    try:
        popt, _ = curve_fit(gauss_hermite, bin_centers, hist, p0=p0, maxfev=10000)
    except RuntimeError:
        return np.nan, np.nan, np.nan, np.nan
    return popt


# ---------------------------------------------------------------------------
# Voronoi-binned maps
# ---------------------------------------------------------------------------
def compute_component_maps_voronoi(positions, velocities, masses, ages, metallicities,
                                   binning, edges, pix_kpc, los_axis=2, transpose=False):
    """
    Compute all maps for one component on the nGIST Voronoi bins.

    Per bin: stellar mass surface density, Gauss-Hermite v and sigma, mass-weighted
    age and metallicity. The bin values are painted back onto the spaxel grid
    (bin_value[bin_map]), with unbinned spaxels set to NaN, exactly like the nGIST maps.

    Returns:
    --------
    results : dict
        Spaxel images [ix, iy] ('mass', 'los', 'sigma', 'age', 'metallicity'),
        per-bin values ('*_bin'), and the spaxel edges in kpc.
    """
    n_bins = binning["n_bins"]
    bin_map = binning["bin_map"].T if transpose else binning["bin_map"]
    nspax = binning["nspax"]

    particle_bin = assign_particles_to_bins(positions, binning, edges, pix_kpc, transpose)
    sel = particle_bin >= 0
    b = particle_bin[sel]
    m = masses[sel]
    los_vel = velocities[sel, los_axis]

    # mass and mass-weighted age / metallicity per bin
    mass_bin = np.bincount(b, weights=m, minlength=n_bins)
    age_bin = np.bincount(b, weights=m * ages[sel], minlength=n_bins)
    metal_bin = np.bincount(b, weights=m * metallicities[sel], minlength=n_bins)
    has_mass = mass_bin > 0
    age_bin = np.divide(age_bin, mass_bin, where=has_mass, out=np.full(n_bins, np.nan))
    metal_bin = np.divide(metal_bin, mass_bin, where=has_mass, out=np.full(n_bins, np.nan))
    with np.errstate(divide="ignore", invalid="ignore"):
        metal_bin = np.log10(metal_bin / Z_sun)

    # surface density, so bins of different size are comparable [Msun / kpc^2]
    sigma_mass_bin = mass_bin / (nspax * pix_kpc**2)

    # Gauss-Hermite fit of the LOSVD in each bin (particles grouped by bin)
    los_bin = np.full(n_bins, np.nan)
    disp_bin = np.full(n_bins, np.nan)
    order = np.argsort(b, kind="stable")
    b_sorted, v_sorted = b[order], los_vel[order]
    starts = np.searchsorted(b_sorted, np.arange(n_bins), side="left")
    stops = np.searchsorted(b_sorted, np.arange(n_bins), side="right")
    for k in range(n_bins):
        if stops[k] - starts[k] >= 10:
            popt = fit_losvd(v_sorted[starts[k]:stops[k]])
            los_bin[k], disp_bin[k] = popt[0], popt[1]

    def to_image(values):
        img = np.full(bin_map.shape, np.nan)
        valid = bin_map >= 0
        img[valid] = values[bin_map[valid]]
        return img

    return {
        'mass': to_image(sigma_mass_bin),
        'los': to_image(los_bin),
        'sigma': to_image(disp_bin),
        'age': to_image(age_bin),
        'metallicity': to_image(metal_bin),
        'mass_bin': sigma_mass_bin,
        'los_bin': los_bin,
        'sigma_bin': disp_bin,
        'age_bin': age_bin,
        'metallicity_bin': metal_bin,
        'x_edges': edges,
        'y_edges': edges,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_component_grid(components_data, component_names, extent=10, figsize=None,
                        save_path=None):
    """
    Rows: components; columns: mass surface density, LOS velocity, velocity
    dispersion, age, metallicity (all on the Voronoi bins).
    """
    n_components = len(components_data)
    n_cols = 5

    if figsize is None:
        figsize = (16, n_components * 2.5)

    fig, axes = plt.subplots(n_components, n_cols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_2d(axes)

    column_titles = ['Mass density\n[log(M☉ kpc$^{-2}$)]', 'LOS velocity [km/s]',
                     'Velocity dispersion [km/s]', 'Mass-weighted age [Gyr]',
                     'Mass-weighted metallicity\n[log(Z/Z☉)]']
    panels = [
        # key, colormap, vmin, vmax, transform
        ('mass', 'magma', 6.5, 10.5, np.log10),
        ('los', 'seismic', -220, 220, None),
        ('sigma', 'viridis', 0, 200, None),
        ('age', 'turbo', 6, 12, None),
        ('metallicity', 'RdYlBu_r', -0.5, 0.5, None),
    ]

    for title, ax in zip(column_titles, axes[0]):
        ax.text(0.5, 1.08, title, transform=ax.transAxes, fontsize=14, ha='center')

    for row_idx, component_data in enumerate(components_data):
        X, Y = np.meshgrid(component_data['x_edges'], component_data['y_edges'])

        for col_idx, (key, cmap, vmin, vmax, transform) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            img = component_data[key]
            if transform is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    img = transform(img)
            img = np.ma.masked_invalid(img)
            # images are indexed [ix, iy] -> transpose for pcolormesh
            mesh = ax.pcolormesh(X, Y, img.T, cmap=cmap, vmin=vmin, vmax=vmax,
                                 shading='flat', rasterized=True)
            cbar = plt.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
            cbar.ax.tick_params(labelsize=14)

            ax.tick_params(axis='both', which='both', direction='in', length=6, width=1,
                           top=True, right=True, labeltop=False, labelright=False,
                           labelbottom=False, labelleft=False, pad=6)
            ax.set_xlim(-extent, extent)
            ax.set_ylim(-extent, extent)
            ax.set_aspect('equal')

            if row_idx == n_components - 1:
                ax.set_xlabel('x [kpc]', fontsize=14)
                ax.tick_params(axis='x', labelbottom=True, labelsize=14)

        axes[row_idx, 0].set_ylabel(f'{component_names[row_idx]}\ny [kpc]', fontsize=14)
        axes[row_idx, 0].tick_params(axis='y', labelleft=True, labelsize=14)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


TRANSPOSE_BINS = False  # flip only if the orientation check below says so

if __name__ == "__main__":
    binning = load_voronoi_binning(ngist_dir, ngist_prefix)
    edges, pix_kpc = spaxel_edges_kpc(binning, z, cosmology_name)
    print(f"Spaxel size {pix_kpc * 1e3:.1f} pc, FoV ±{edges[-1]:.2f} kpc")

    components_data = []
    component_names_list = []

    for component, name in zip(components, component_names):
        print(f"Loading {component}...")
        try:
            star_coordinates, star_velocities, star_ages, star_metallicity, star_masses = \
                load_component_data(component)

            if component == "None":
                check_orientation(star_coordinates, star_velocities, star_masses,
                                  binning, edges, pix_kpc, los_axis=2)

            comp_data = compute_component_maps_voronoi(
                star_coordinates, star_velocities, star_masses, star_ages, star_metallicity,
                binning, edges, pix_kpc, los_axis=2, transpose=TRANSPOSE_BINS,
            )

            components_data.append(comp_data)
            component_names_list.append(name)
            print(f"  ✓ {component} loaded successfully")
        except Exception as e:
            print(f"  ✗ Error loading {component}: {e}")

    if components_data:
        print(f"\nCreating figure with {len(components_data)} components...")
        output_dir = "../plots_components"
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, f"simulation_component_comparison_{galaxy_id}_voronoi.jpeg")
        plot_component_grid(components_data, component_names_list, extent=edges[-1],
                            save_path=save_path)

        # per-bin values, e.g. for a direct bin-by-bin comparison with the nGIST output
        np.savez(os.path.join(output_dir, f"simulation_components_{galaxy_id}_voronoi_bins.npz"),
                 components=np.array(component_names_list),
                 **{f"{name}_{key}": d[f"{key}_bin"]
                    for name, d in zip(component_names_list, components_data)
                    for key in ('mass', 'los', 'sigma', 'age', 'metallicity')})
        print("Done!")
