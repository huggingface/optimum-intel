import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Dynamically add the 'gemma4' marker to every parameterized test whose
    name contains 'gemma4' (this also covers 'gemma4_moe')."""
    gemma4_marker = pytest.mark.gemma4
    for item in items:
        if "gemma4" in item.nodeid:
            item.add_marker(gemma4_marker)
