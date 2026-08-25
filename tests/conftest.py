import gc
import warnings
import pytest

@pytest.fixture(autouse=True)
def clean_memory_and_warnings():
    # Filter known third-party background stream teardowns
    warnings.filterwarnings("ignore", category=ResourceWarning)
    yield
    gc.collect()

