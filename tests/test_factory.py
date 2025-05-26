from unittest.mock import MagicMock, patch

import pytest

from rubix.galaxy.input_handler.factory import get_input_handler


def test_get_input_handler_illustris():
    config = {"simulation": {"name": "IllustrisTNG", "args": {"path": "value1"}}}

    with patch("rubix.galaxy.input_handler.factory.IllustrisHandler") as mock_handler:
        mock_instance = MagicMock()
        mock_handler.return_value = mock_instance

        result = get_input_handler(config)

        # Check if the mock instance is returned
        assert result == mock_instance
        # Ensure that the constructor is called with the correct arguments
        mock_handler.assert_called_once_with(path="value1", logger=None)


def test_get_input_handler_unsupported():
    config = {"simulation": {"name": "UnknownSim", "args": {}}}

    with pytest.raises(ValueError) as excinfo:
        get_input_handler(config)


def test_get_input_handler_illustris():
    config = {"simulation": {"name": "IllustrisTNG", "args": {"path": "value1"}}}

    with patch("rubix.galaxy.input_handler.factory.IllustrisHandler") as mock_handler:
        mock_instance = MagicMock()
        mock_handler.return_value = mock_instance

        result = get_input_handler(config)

        # Check if the mock instance is returned
        assert result == mock_instance
        # Ensure that the constructor is called with the correct arguments
        mock_handler.assert_called_once_with(path="value1", logger=None)


def test_get_input_handler_unsupported():
    config = {"simulation": {"name": "UnknownSim", "args": {}}}

    with pytest.raises(ValueError) as excinfo:
        get_input_handler(config)

    assert "not supported" in str(excinfo.value)
