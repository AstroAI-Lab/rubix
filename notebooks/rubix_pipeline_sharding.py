import os

#os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=3'

# Specify the number of GPUs to use
os.environ['CUDA_VISIBLE_DEVICES'] = "1,4,5,8,9"

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]   = "false"

#Set the FSPS path to the template files
#  os.environ['SPS_HOME'] = '/mnt/storage/annalena_data/sps_fsps'
#os.environ['SPS_HOME'] = '/home/annalena/sps_fsps'
#os.environ['SPS_HOME'] = '/Users/annalena/Documents/GitHub/fsps'
os.environ['SPS_HOME'] = '/export/home/aschaibl/fsps'

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from rubix.core.pipeline import RubixPipeline 
# Now JAX will list two CpuDevice entries
print(jax.devices())



config = {
    "pipeline":{"name": "calc_ifu"},
    
    "logger": {
        "log_level": "DEBUG",
        "log_file_path": None,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    },
    "data": {
        "name": "IllustrisAPI",
        "args": {
            "api_key": os.environ.get("ILLUSTRIS_API_KEY"),
            "particle_type": ["stars"],
            "simulation": "TNG50-1",
            "snapshot": 99,
            "save_data_path": "data",
        },
        
        "load_galaxy_args": {
        "id": 14,
        "reuse": True,
        },
        
        "subset": {
            "use_subset": True,
            "subset_size": 10000,
        },
    },
    "simulation": {
        "name": "IllustrisTNG",
        "args": {
            "path": "data/galaxy-id-14.hdf5",
        },
    
    },
    "output_path": "output",

    "telescope":
        {"name": "MUSE",
         "psf": {"name": "gaussian", "size": 5, "sigma": 0.6},
         "lsf": {"sigma": 0.5},
         "noise": {"signal_to_noise": 100,"noise_distribution": "normal"},},
    "cosmology":
        {"name": "PLANCK15"},
        
    "galaxy":
        {"dist_z": 0.1,
         "rotation": {"type": "edge-on"},
        },
        
    "ssp": {
        "template": {
            "name": "FSPS"
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

pipe = RubixPipeline(config)
inputdata = pipe.prepare_data()
rubixdata = pipe.run_sharded(inputdata)


#Plotting the spectra
wave = pipe.telescope.wave_seq

plt.figure(figsize=(10, 5))
plt.title("Spectra of a single star")
plt.xlabel("Wavelength (Angstroms)")
plt.ylabel("Luminosity")
#spectra = rubixdata.stars.datacube # Spectra of all stars
spectra = rubixdata
plt.plot(wave, spectra[12,12,:])
plt.plot(wave, spectra[12,14,:])
plt.savefig("./output/rubix_spectra.jpg")
plt.close()

plt.figure(figsize=(6, 5))
# get the indices of the visible wavelengths of 4000-8000 Angstroms
visible_indices = jnp.where((wave >= 4000) & (wave <= 8000))
#visible_spectra = rubixdata.stars.datacube[:, :, visible_indices[0]]
visible_spectra = rubixdata[:, :, visible_indices[0]]
# Sum up all spectra to create an image
image = jnp.sum(visible_spectra, axis = 2)
plt.imshow(image, origin="lower", cmap="inferno")
plt.colorbar()
plt.title("Image of the galaxy")
plt.xlabel("X pixel")
plt.ylabel("Y pixel")
plt.savefig("./output/rubix_image.jpg")
plt.close()


