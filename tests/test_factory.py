import logging
from unittest.mock import MagicMock, patch

import pytest

from rubix.galaxy.input_handler.factory import get_input_handler


def test_get_input_handler_illustris():
    config = {"simulation": {"name": "IllustrisTNG", "args": {"path": "value1"}}}

    with patch("rubix.galaxy.input_handler.factory.IllustrisHandler") as mock_handler:
        mock_instance = MagicMock()
        mock_handler.return_value = mock_instance

        result = get_input_handler(config)

        assert result == mock_instance
        mock_handler.assert_called_once_with(path="value1", logger=None)


def test_get_input_handler_unsupported():
    config = {"simulation": {"name": "UnknownSim", "args": {}}}

    with pytest.raises(ValueError) as excinfo:
        get_input_handler(config)

    assert "not supported" in str(excinfo.value)


def test_get_input_handler_pynbody():
    config = {
        "simulation": {
            "name": "NIHAO",
            "args": {"path": "/tmp/nihao", "halo_path": "/tmp/halo"},
        },
        "galaxy": {"dist_z": 0.12},
    }
    logger = logging.getLogger("rubix.tests.factory.pynbody")
    logger.info = MagicMock()

    with patch("rubix.galaxy.input_handler.factory.PynbodyHandler") as mock_handler:
        mock_instance = MagicMock()
        mock_handler.return_value = mock_instance

        result = get_input_handler(config, logger=logger)

        assert result == mock_instance
        mock_handler.assert_called_once_with(
            path="/tmp/nihao",
            halo_path="/tmp/halo",
            dist_z=0.12,
            logger=logger,
        )
        logger.info.assert_any_call("Using PynbodyHandler to load a NIHAO galaxy")
