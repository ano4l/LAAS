import pytest


@pytest.fixture(autouse=True)
def isolate_analysis_compute(monkeypatch):
    # A developer's enabled cloud backend must never make unit tests incur GPU usage.
    monkeypatch.setenv("ANALYSIS_COMPUTE", "local")
