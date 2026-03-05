import os
import glob
import re
import numpy as np
import h5py
from astropy.io import fits

#do in terminal to link to the correct folder
#ln -s /home/annalena/rubix/rubix/spectra/ssp/templates/EMILES_template.h5 /home/annalena/.conda/envs/rubix/lib/python3.12/site-packages/rubix/spectra/ssp/templates/EMILES_template.h5

#this file converts the MILES SSP templates from their original FITS format to the HDF5 format expected by RUBIX
#input and output paths
template_dir = './templates/EMILES_BASTI_BASE_CH_FITS_safe/'
output_h5_file = './templates/EMILES_template.h5'

# Solar metallicity reference
Z_SUN = 0.0142

# 1. Gather all fits files in the directory
fits_files = glob.glob(os.path.join(template_dir, '*.fits'))
if not fits_files:
    raise FileNotFoundError(f"No FITS files found in {template_dir}")

# 2. Regex to extract Age and Metallicity from the nGIST filename format
# Example: Mun1.30Zp0.00T10.0000_iPp0.00_baseFe.fits
pattern = re.compile(r'Z([pm]\d+\.\d+)T(\d+\.\d+)')

data_dict = {}
ages_set = set()
mets_set = set()

print(f"Found {len(fits_files)} FITS files. Processing...")

for f in fits_files:
    filename = os.path.basename(f)
    match = pattern.search(filename)
    
    if match:
        # Extract and format metallicity
        # Replace 'p' with '+' and 'm' with '-'
        z_str = match.group(1).replace('p', '+').replace('m', '-')
        met_log = float(z_str)
        met = Z_SUN * (10 ** met_log)
        
        # Extract age
        age = float(match.group(2))
        
        ages_set.add(age)
        mets_set.add(met)
        
        # Read the FITS data
        with fits.open(f) as hdul:
            flux = hdul[0].data
            header = hdul[0].header
            
            # Reconstruct wavelength array
            crval = header['CRVAL1']
            # Sometimes CDELT1 is named CD1_1 depending on the FITS writer
            cdelt = header.get('CDELT1', header.get('CD1_1')) 
            wave = crval + np.arange(len(flux)) * cdelt
            
            data_dict[(age, met)] = (wave, flux)

# 3. Sort into ordered arrays
ages = np.sort(list(ages_set))
metallicities = np.sort(list(mets_set))

print(f"Found {len(ages)} unique ages and {len(metallicities)} unique metallicities.")

# 4. Build the master flux grid
# Get the wavelength array from the first template (they should all share the same grid)
sample_wave = data_dict[(ages[0], metallicities[0])][0]

# Create the flux grid. 
# Note: The shape is set to (n_age, n_metallicity, n_wavelength) here. 
# If your RUBIX reader expects (n_metallicity, n_age, n_wavelength), just swap `len(ages)` and `len(metallicities)` here and in the loop!
flux_grid = np.zeros((len(metallicities), len(ages), len(sample_wave)))

for j, m in enumerate(metallicities): # Swap the loop order for clarity
    for i, a in enumerate(ages):
        if (a, m) in data_dict:
            # Note: Index is now [j, i, :] -> [metallicity, age, wavelength]
            flux_grid[j, i, :] = data_dict[(a, m)][1]
        else:
            print(f"Warning: Missing FITS file for Age={a}, Metallicity={m}")

# 5. Write to HDF5 using the EXACT keys found in your BC03lr_old.h5 file
with h5py.File(output_h5_file, 'w') as hf:
    hf.create_dataset('flux', data=flux_grid)
    hf.create_dataset('wavelength', data=sample_wave)
    hf.create_dataset('age', data=ages)
    hf.create_dataset('metallicity', data=metallicities)

print(f"\nSuccess! HDF5 file saved to {output_h5_file}")