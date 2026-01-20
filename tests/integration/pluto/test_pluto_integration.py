"""Pluto Integration Smoke Test

Minimal test that verifies basic Pluto SDK connectivity and functionality.

What it tests:
- Authentication works
- Can create a run in the simple_test project
- Can log 1 config parameter (integration_test=True)
- Can log 1 metric (integration_metric=0.123)

What it uploads to Pluto:
- Total: 2 items (1 config + 1 metric)
- Project: simple_test (or PLUTO_PROJECT env var)
- Run name: neptune-exporter-integration-test

This is a health check test - not comprehensive.
"""

import os
import pytest

def test_pluto_integration_smoke():
    api_key = os.getenv("PLUTO_API_KEY")
    project = os.getenv("PLUTO_PROJECT", "simple_test")
    # Don't require API key - pluto.init() will use stored credentials
    # if not api_key:
    #     pytest.skip("Set PLUTO_API_KEY to run Pluto integration tests")

    try:
        import pluto
    except Exception as e:
        pytest.skip(f"Pluto SDK not installed: {e}")

    # Prefer Op-based SDK if available
    if hasattr(pluto, "init"):
        op = pluto.init(project=project, name="neptune-exporter-integration-test")
        op.update_config({"integration_test": True})
        op.log({"integration_metric": 0.123})
        # optional: finish/close if SDK exposes it
        if hasattr(op, "finish"):
            op.finish()
    elif hasattr(pluto, "Client"):
        client = pluto.Client(api_key=api_key)
        run = client.create_run(project=project, name="neptune-exporter-integration-test")
        client.runs.update(run_id=getattr(run, "id", run), metadata={"integration_test": True})
        client.log_metric(run_id=getattr(run, "id", run), key="integration_metric", value=0.123)
    else:
        pytest.skip("Pluto SDK does not expose expected API")