import h5py
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact
from jdaviz import Cubeviz
from mpdaf.obj import Cube


def visualize_rubix(filename):
    """
    This function creates an interactive visualization of a given data cube.
    The visualization consists of two plots:
    1. A 2D image slice of the data cube at a given wavelength index.
    2. A spectrum plot for a selected pixel in the image slice, along with the summed spectrum for all pixels within a given radius of the selected pixel.

    Parameters:
        filename (str): The path to the FITS file containing the data cube.

    Returns:
        An interactive visualization of the data cube with sliders for selecting the wavelength index, pixel coordinates, and radius of the aperture.
    """

    cube = Cube(filename=filename)

    # Define your plotting function for both slice and spectrum
    def plot_cube_slice_and_spectrum(wave_index, wave_range, x, y, radius):

        # Extract the image slice over the wavelength range
        if wave_index - wave_range < 0:
            start = 0
        else:
            start = wave_index - wave_range
        image1 = cube[start : wave_index + wave_range, :, :].sum(axis=0)

        # Extract the spectrum for the given pixel
        spectrum = cube[:, y, x]

        # Create a mask for pixels within the specified radius of the selected pixel (x, y)
        y_indices, x_indices = np.indices((cube.shape[1], cube.shape[2]))
        distance_mask = np.sqrt((x_indices - x) ** 2 + (y_indices - y) ** 2) <= radius

        # Get the coordinates of the selected pixels within the radius
        y_coords, x_coords = np.where(distance_mask)

        # Sum the spectra for all the selected pixels
        spectrum_sum = cube.data[:, y_coords, x_coords].sum(axis=1)

        # Extract the overall summed spectrum for the entire field of view
        spectrum_all = cube.data.sum(axis=(1, 2))

        # Create the figure and two axes (subplots)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot the image slice on the first axis (ax1)
        image1.plot(
            ax=ax1,
            colorbar="v",
            title="$\lambda$ = %.1f to %.1f (%s)"
            % (
                cube.wave.coord(start),
                cube.wave.coord(wave_index + wave_range),
                cube.wave.unit,
            ),
        )

        # Add scatter point for the selected pixel
        ax1.scatter(x, y, color="red", marker="o", s=100)

        # Highlight all selected pixels by overlaying the mask with a transparent color
        mask_overlay = np.zeros_like(image1.data)
        mask_overlay[distance_mask] = 1  # Mark selected pixels
        ax1.imshow(mask_overlay, origin="lower", cmap="Blues", alpha=0.1)

        # Set labels for the image plot
        ax1.set_xlabel("X-axis")
        ax1.set_ylabel("Y-axis")

        # Plot the spectrum on the second axis (ax2)
        spectrum.plot(ax=ax2, label=f"Spectrum for Pixel ({x}, {y})", color="blue")

        # Create a second y-axis for the field-of-view spectrum
        ax3 = ax2.twinx()
        ax3.plot(
            cube.wave.coord(),
            spectrum_all,
            label=f"Spectrum for Whole FOV",
            color="black",
        )

        # Plot the summed spectrum for the selected region
        ax2.plot(
            cube.wave.coord(),
            spectrum_sum,
            label=f"Summed Spectrum (Radius = {radius})",
            color="green",
            linestyle="-.",
        )

        # Add a vertical line to indicate the current wavelength
        ax3.vlines(
            cube.wave.coord(wave_index),
            ymin=0,
            ymax=np.max(spectrum_all),
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="$\lambda$ = %.1f (%s)"
            % (cube.wave.coord(wave_index), cube.wave.unit),
        )

        # Highlight a range around the current wavelength
        ax2.axvspan(
            cube.wave.coord(wave_index - wave_range),
            cube.wave.coord(wave_index + wave_range),
            color="red",
            alpha=0.2,
        )

        # Set labels, grid, and legends
        ax2.set_xlabel("Wavelength (%s)" % cube.wave.unit)
        ax2.set_ylabel("Intensity (Pixel and Summed Region)")
        ax2.grid()
        ax3.set_ylabel("Intensity (Whole FOV)")

        # Add legends to both y-axes
        ax2.legend(loc="upper left")
        ax3.legend(loc="upper right")

        # set ax2 and ax3 lower limit to 0

        ax2.set_ylim(bottom=0)
        ax3.set_ylim(bottom=0)

        # Adjust layout for better visualization
        plt.tight_layout()

        # Show the plot
        plt.show()

    # Create an interactive slider for the wave index (for cube slice)
    wave_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=cube.shape[0] - 1,
        step=1,
        description="Waveindex:",
        continuous_update=False,
    )

    # Create an interactive slider for the wavelength range
    wave_ranger = widgets.IntSlider(
        value=10,
        min=0,
        max=cube.shape[0] // 2,
        step=1,
        description="Wavelengthrange:",
        continuous_update=False,
    )

    # Create interactive sliders for X and Y pixel selection
    x_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=cube.shape[2] - 1,
        step=1,
        description="X Pixel:",
        continuous_update=False,
    )

    y_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=cube.shape[1] - 1,
        step=1,
        description="Y Pixel:",
        continuous_update=False,
    )

    # Create an interactive slider for the radius
    radius_slider = widgets.IntSlider(
        value=1,
        min=0,
        max=min(cube.shape[1] // 2, cube.shape[2] // 2) * 2,
        step=1,
        description="Radius:",
        continuous_update=False,
    )

    # Use the interact function to link the sliders with the combined plotting function
    interactive_plot = interact(
        plot_cube_slice_and_spectrum,
        wave_index=wave_slider,
        wave_range=wave_ranger,
        x=x_slider,
        y=y_slider,
        radius=radius_slider,
    )

    return interactive_plot


def visualize_cubeviz(filename):
    """
    This function creates an interactive visualization of a given data cube using Cubeviz.

    Parameters:
        filename (str): The path to the FITS file containing the data cube.

    Returns:
        An interactive visualization of the data cube using Cubeviz.
    """

    def cubeviz(self):
        cubeviz = Cubeviz()
        cubeviz.load_data(filename)
        cubeviz.show()

    return cubeviz(filename)


def stellar_age_histogram(h5_file):
    with h5py.File(h5_file, "r") as f:
        star_ages = f["particles/stars/age"][:]
    plt.figure(figsize=(8, 6))
    plt.hist(star_ages, bins=50, color="darkorange", edgecolor="black", alpha=0.7)
    plt.xlabel("Age [Gyr]")
    plt.ylabel("Number of stars")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def star_coords_2D(h5_file):
    with h5py.File(h5_file, "r") as f:
        star_coords = f["particles/stars/coords"][:]
    x = star_coords[:, 0]
    y = star_coords[:, 1]
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=1, alpha=0.5)
    plt.xlabel("x [kpc]")
    plt.ylabel("y [kpc]")
    plt.grid(True)
    plt.show()


def star_metallicity_histogram(h5_file):
    with h5py.File(h5_file, "r") as f:
        star_metallicity = f["particles/stars/metallicity"][:]
    plt.figure(figsize=(8, 6))
    plt.hist(
        star_metallicity, bins=50, color="forestgreen", edgecolor="black", alpha=0.7
    )
    plt.xlabel("Metallicity")
    plt.ylabel("Number of stars")
    plt.title("Stellar Metallicity Distribution")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
