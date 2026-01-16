import os
import pytest

def test_pluto_integration_smoke():
    api_key = os.getenv("PLUTO_API_KEY")
    project = os.getenv("PLUTO_PROJECT", "ryan")
    if not api_key:
        pytest.skip("Set PLUTO_API_KEY to run Pluto integration tests")

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