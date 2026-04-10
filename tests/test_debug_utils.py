import logging
import os

from ra_sim import debug_utils


def test_enable_numba_logging_defaults_to_warning(monkeypatch) -> None:
    monkeypatch.setenv("RA_SIM_DEBUG", "1")
    monkeypatch.delenv("NUMBA_LOG_LEVEL", raising=False)
    logger = logging.Logger("numba-test-warning")
    original_get_logger = logging.getLogger

    monkeypatch.setattr(
        debug_utils.logging,
        "getLogger",
        lambda name=None: logger if name == "numba" else original_get_logger(name),
    )

    debug_utils.enable_numba_logging()

    assert logger.level == logging.WARNING
    assert len(logger.handlers) == 1
    assert logger.handlers[0].level == logging.NOTSET
    assert logger.handlers[0].formatter is not None
    assert os.environ["NUMBA_LOG_LEVEL"] == "WARNING"


def test_enable_numba_logging_respects_explicit_level(monkeypatch) -> None:
    monkeypatch.setenv("RA_SIM_DEBUG", "1")
    monkeypatch.setenv("NUMBA_LOG_LEVEL", "DEBUG")
    logger = logging.Logger("numba-test-debug")
    original_get_logger = logging.getLogger

    monkeypatch.setattr(
        debug_utils.logging,
        "getLogger",
        lambda name=None: logger if name == "numba" else original_get_logger(name),
    )

    debug_utils.enable_numba_logging()

    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1
    assert os.environ["NUMBA_LOG_LEVEL"] == "DEBUG"
