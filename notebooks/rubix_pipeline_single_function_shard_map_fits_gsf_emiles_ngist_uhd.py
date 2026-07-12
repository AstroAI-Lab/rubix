import os
import jax
from jax import config
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from rubix.core.pipeline import RubixPipeline 
from rubix.spectra.ifu import convert_luminoisty_to_flux
from rubix.cosmology import PLANCK15
from rubix.core.fits import store_fits


config.update("jax_enable_x64", True)
# Only make GPU 0 and GPU 1 visible to JAX:
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
print(jax.devices())


galaxy_id = os.environ.get("GALAXY_ID")
#component_str = os.environ.get("COMPONENT", None)
angle_str = os.environ.get("ANGLE", None)
split_file = os.environ.get("SPLIT_FILE", "")
sim_path = os.environ.get("SIM_PATH", "/home/_data/nihao/nihao_uhd/g2.79e12_3x6/2.79e12.02000")
halo_path = os.environ.get("HALO_PATH", "/home/_data/nihao/nihao_uhd/g2.79e12_3x6/2.79e12.02000.z0.000.AHF_halos")

print("Notebook gestartet mit:")
print("GALAXY_ID =", galaxy_id)
print("ANGLE =", angle_str)
print("PATH =", sim_path)
print("HALO_PATH =", halo_path)


if angle_str is None or angle_str.strip() == "" or angle_str == "None":
    angle = None
else:
    angle = angle_str



#  os.environ['SPS_HOME'] = '/mnt/storage/annalena_data/sps_fsps'
#os.environ['SPS_HOME'] = '/home/annalena/sps_fsps'
#os.environ['SPS_HOME'] = '/Users/annalena/Documents/GitHub/fsps'
#os.environ['SPS_HOME'] = '/export/home/aschaibl/fsps'
os.environ['SPS_HOME'] = '/home/annalena_data/sps_fsps'


config_NIHAO = {
    "pipeline":{"name": "calc_ifu"},
    
    "logger": {
        "log_level": "DEBUG",
        "log_file_path": None,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    },
    "data": {
        "name": "NihaoHandler",
        "args": {
            "particle_type": ["stars"],
            "save_data_path": "data",
            "snapshot": "1024",
        },
        "load_galaxy_args": {"reuse": True, "id": galaxy_id},
        "subset": {"use_subset": True, "subset_size": 20000},
    },
    "simulation": {
        "name": "NIHAO",
        "args": {
            "path": sim_path,
            "halo_path": halo_path,
            "halo_id": 0,
        },
    },
    "output_path": "output",

    "telescope":
        {"name": "MUSE_WFM",
         "psf": {"name": "gaussian", "size": 300, "sigma": 3.0},
         "lsf": {"sigma": 1.0},
         "noise": {"signal_to_noise": 100,"noise_distribution": "normal"},},
    "cosmology":
        {"name": "PLANCK15"},
        
    "galaxy":
        {"dist_z": 0.01,
         "rotation": {"alpha": float(angle), "beta": 0.0, "gamma": 0.0}, #{"type": "matrix"}, #{"alpha": 0.0, "beta": 0.0, "gamma":0.0},
         "component": None,#None, #["ThinDisk", "ThickDisk", "Halo", "PseudoBulge"], #None,
         "component_file": split_file, 
        },
        
    "ssp": {
        "template": {
            "name": "EMILES_BASTI_BASE_CH_FITS_safe" #"Mastar_CB19_SLOG_1_5" #"FSPS" #"BruzualCharlot2003" #"Mastar_CB19_SLOG_1_5"
        },
        "dust": {
                "extinction_model": "Cardelli89",
                "dust_to_gas_ratio": 0.01,
                "dust_to_metals_ratio": 0.4,
                "dust_grain_density": 3.5,
                "Rv": 3.1,
            },
    },        
}



pipe = RubixPipeline(config_NIHAO)

inputdata = pipe.prepare_data()
rubixdata = pipe.run_sharded(inputdata)


observation_lum_dist = PLANCK15.luminosity_distance_to_z(config_NIHAO["galaxy"]["dist_z"])
observation_z = config_NIHAO["galaxy"]["dist_z"]
pixel_size = 1.0
fluxcube = convert_luminoisty_to_flux(rubixdata, observation_lum_dist, observation_z, pixel_size)
rubixdata = fluxcube/1e-20

store_fits(config_NIHAO, rubixdata, "./output/emiles/")


wave = pipe.telescope.wave_seq
visible_indices = jnp.where((wave >= 4000) & (wave <= 8000))

sharded_visible_spectra = rubixdata[ :, :, visible_indices[0]]
sharded_image = jnp.sum(sharded_visible_spectra, axis=2)
img32 = np.array(sharded_image, dtype=np.float32)

# Plot side by side
plt.figure(figsize=(6, 5))
# Sharded IFU datacube image
plt.imshow(img32, origin="lower", cmap="inferno")
plt.title("Sharded IFU Datacube")
plt.colorbar(label="Flux [erg/s/cm^2]")
plt.tight_layout()
plt.savefig(f"./output/emiles/image_{galaxy_id}_{angle}.jpeg")
plt.show()


spectra_sharded = rubixdata # Spectra of all stars
#print(spectra.shape)

plt.figure(figsize=(10, 5))
plt.title("Rubix Sharded")
plt.xlabel("Wavelength [Angstrom]")
plt.ylabel("Flux [erg/s/cm^2/Angstrom]")
plt.plot(wave, spectra_sharded[150,150,:])
plt.plot(wave, spectra_sharded[150,200,:])
plt.plot(wave, spectra_sharded[200,150,:])

plt.savefig(f"./output/emiles/spectra_{galaxy_id}_{angle}.jpeg")
