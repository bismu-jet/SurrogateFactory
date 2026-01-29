"""Shared pytest configuration and fixtures."""

import logging

# Surface all log output during tests so failures are easy to diagnose.
logging.basicConfig(level=logging.DEBUG)
