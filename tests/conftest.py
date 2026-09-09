import json
from pathlib import Path
import pytest

@pytest.fixture
def diagnostic_config():
    return json.loads((Path(__file__).parents[1] / "recipes/diagnostic_cpu.json").read_text())
