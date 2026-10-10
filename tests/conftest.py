import pytest
import torch


@pytest.fixture(scope="session")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
