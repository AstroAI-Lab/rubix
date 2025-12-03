from types import SimpleNamespace
from unittest.mock import MagicMock

import h5py
import numpy as np

from rubix.core import visualisation


class DummyWave:
    def __init__(self):
        self.unit = "Angstrom"

    def coord(self, index=None):
        if index is None:
            return np.array([4000.0, 5000.0])
        return 4000.0 + index * 1000.0


class DummyImage:
    def __init__(self, data):
        self.data = data
        self.plot = MagicMock()


class DummySpectrum:
    def __init__(self, data):
        self.data = data
        self.plot = MagicMock()


class DummyCube:
    def __init__(self):
        self.shape = (4, 3, 3)
        self.data = np.arange(np.prod(self.shape)).reshape(self.shape)
        self.wave = DummyWave()
        self.slice_calls = []

    def __getitem__(self, key):
        self.slice_calls.append(key)
        sliced = self.data[key]
        if sliced.ndim == 3:
            return DummyCubeSlice(sliced)
        return DummySpectrum(sliced)


class DummyCubeSlice:
    def __init__(self, data):
        self._data = data

    def sum(self, axis=0):
        return DummyImage(self._data.sum(axis=axis))


def test_plot_cube_slice_and_spectrum(monkeypatch):
    cube, interact_data, ax1, ax2, ax3 = _prepare_visualize_plot(monkeypatch)
    plot_fn = interact_data["func"]
    plot_fn(wave_index=1, wave_range=1, x=1, y=1, radius=1)

    ax1.scatter.assert_called_once()
    ax1.imshow.assert_called_once()
    ax2.plot.assert_called()
    ax3.plot.assert_called_once()
    ax2.axvspan.assert_called_once()
    ax2.set_xlabel.assert_called_once()
    ax2.set_ylabel.assert_called_once()
    ax2.grid.assert_called_once()
    ax2.legend.assert_called_once()
    ax3.set_ylabel.assert_called_once()
    ax3.legend.assert_called_once()
    ax2.set_ylim.assert_called_with(bottom=0)
    ax3.set_ylim.assert_called_with(bottom=0)
    ax3.vlines.assert_called_once()


def test_plot_cube_slice_and_spectrum_clamps_start(monkeypatch):
    cube, interact_data, _, _, _ = _prepare_visualize_plot(monkeypatch)
    plot_fn = interact_data["func"]

    plot_fn(wave_index=1, wave_range=2, x=1, y=1, radius=1)

    assert cube.slice_calls
    first_slice = cube.slice_calls[0]
    assert isinstance(first_slice, tuple)
    slice_axis = first_slice[0]
    assert isinstance(slice_axis, slice)
    assert slice_axis.start == 0


def _prepare_visualize_plot(monkeypatch):
    cube = DummyCube()
    monkeypatch.setattr(visualisation, "Cube", lambda filename: cube)

    def fake_slider(**kwargs):
        return SimpleNamespace(description=kwargs.get("description", ""))

    monkeypatch.setattr(visualisation.widgets, "IntSlider", fake_slider)

    ax1 = MagicMock()
    ax2 = MagicMock()
    ax3 = MagicMock()
    ax2.twinx.return_value = ax3
    fig = MagicMock()
    monkeypatch.setattr(
        visualisation.plt,
        "subplots",
        lambda *args, **kwargs: (fig, (ax1, ax2)),
    )
    monkeypatch.setattr(visualisation.plt, "tight_layout", MagicMock())
    monkeypatch.setattr(visualisation.plt, "show", MagicMock())

    interact_data = {}

    def fake_interact(func, **kwargs):
        interact_data["func"] = func
        return "widget"

    monkeypatch.setattr(visualisation, "interact", fake_interact)

    visualisation.visualize_rubix("/tmp/cube.fits")

    return cube, interact_data, ax1, ax2, ax3


def _create_star_h5(tmp_path):
    path = tmp_path / "stars.h5"
    with h5py.File(path, "w") as f:
        stars = f.create_group("particles/stars")
        stars.create_dataset("age", data=np.array([1.5, 2.0, 3.0]))
        stars.create_dataset(
            "coords",
            data=np.array(
                [
                    [0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                ]
            ),
        )
        stars.create_dataset("metallicity", data=np.array([0.1, 0.2, 0.3]))
    return path


def test_visualize_rubix_sets_up_interact(monkeypatch):
    cube = MagicMock(shape=(4, 5, 6))
    monkeypatch.setattr(visualisation, "Cube", MagicMock(return_value=cube))

    slider_calls = []

    def fake_int_slider(**kwargs):
        slider = MagicMock()
        slider.description = kwargs.get("description")
        slider_calls.append(kwargs)
        return slider

    monkeypatch.setattr(visualisation.widgets, "IntSlider", fake_int_slider)
    interact_mock = MagicMock(return_value="widget")
    monkeypatch.setattr(visualisation, "interact", interact_mock)

    result = visualisation.visualize_rubix("/tmp/cube.fits")

    visualisation.Cube.assert_called_once_with(filename="/tmp/cube.fits")
    assert result == "widget"
    assert len(slider_calls) == 5
    interact_mock.assert_called_once()
    interact_kwargs = interact_mock.call_args.kwargs
    assert "wave_index" in interact_kwargs
    assert interact_kwargs["wave_index"].description == "Waveindex:"
    assert interact_kwargs["x"].description == "X Pixel:"


def test_visualize_cubeviz_loads_and_shows(monkeypatch):
    cubeviz_mock = MagicMock()
    monkeypatch.setattr(
        visualisation,
        "Cubeviz",
        MagicMock(return_value=cubeviz_mock),
    )

    visualisation.visualize_cubeviz("/tmp/cube.fits")

    visualisation.Cubeviz.assert_called_once()
    cubeviz_mock.load_data.assert_called_once_with("/tmp/cube.fits")
    cubeviz_mock.show.assert_called_once()


def test_stellar_age_histogram_uses_hdf5_data(tmp_path, monkeypatch):
    path = _create_star_h5(tmp_path)
    plt = visualisation.plt
    hist = MagicMock()
    monkeypatch.setattr(plt, "figure", MagicMock())
    monkeypatch.setattr(plt, "hist", hist)
    monkeypatch.setattr(plt, "xlabel", MagicMock())
    monkeypatch.setattr(plt, "ylabel", MagicMock())
    monkeypatch.setattr(plt, "grid", MagicMock())
    monkeypatch.setattr(plt, "tight_layout", MagicMock())
    monkeypatch.setattr(plt, "show", MagicMock())

    visualisation.stellar_age_histogram(str(path))

    hist.assert_called_once()
    np.testing.assert_array_equal(hist.call_args.args[0], np.array([1.5, 2.0, 3.0]))


def test_star_coords_2d_scatter(monkeypatch, tmp_path):
    path = _create_star_h5(tmp_path)
    plt = visualisation.plt
    scatter = MagicMock()
    monkeypatch.setattr(plt, "figure", MagicMock())
    monkeypatch.setattr(plt, "scatter", scatter)
    monkeypatch.setattr(plt, "xlabel", MagicMock())
    monkeypatch.setattr(plt, "ylabel", MagicMock())
    monkeypatch.setattr(plt, "grid", MagicMock())
    monkeypatch.setattr(plt, "show", MagicMock())

    visualisation.star_coords_2D(str(path))

    scatter.assert_called_once()
    x_arg, y_arg = scatter.call_args.args[:2]
    np.testing.assert_array_equal(x_arg, np.array([0.0, 3.0]))
    np.testing.assert_array_equal(y_arg, np.array([1.0, 4.0]))


def test_star_metallicity_histogram_plots_metallicity(monkeypatch, tmp_path):
    path = _create_star_h5(tmp_path)
    plt = visualisation.plt
    hist = MagicMock()
    monkeypatch.setattr(plt, "figure", MagicMock())
    monkeypatch.setattr(plt, "hist", hist)
    monkeypatch.setattr(plt, "xlabel", MagicMock())
    monkeypatch.setattr(plt, "ylabel", MagicMock())
    monkeypatch.setattr(plt, "title", MagicMock())
    monkeypatch.setattr(plt, "grid", MagicMock())
    monkeypatch.setattr(plt, "tight_layout", MagicMock())
    monkeypatch.setattr(plt, "show", MagicMock())

    visualisation.star_metallicity_histogram(str(path))

    hist.assert_called_once()
    np.testing.assert_array_equal(hist.call_args.args[0], np.array([0.1, 0.2, 0.3]))
