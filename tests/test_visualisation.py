from unittest.mock import MagicMock

import h5py
import numpy as np

from rubix.core import visualisation


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
    monkeypatch.setattr(visualisation, "Cubeviz", MagicMock(return_value=cubeviz_mock))

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
