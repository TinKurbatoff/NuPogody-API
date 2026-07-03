"""Shared pytest configuration.

Silence loguru during tests so the debug logs the core emits on reset don't
clutter test output.
"""

from loguru import logger

logger.disable("nupogodi")
