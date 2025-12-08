"""Pytest configuration and shared fixtures."""

import pytest


def pytest_configure(config):
    """Register custom pytest marks."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')")

