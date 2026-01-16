import pyarrow as pa
import pandas as pd
from decimal import Decimal
from unittest.mock import Mock
from pathlib import Path

from neptune_exporter.loaders.pluto_loader import PlutoLoader


def make_op_mock():
    op = Mock()
    op.settings = Mock()
    op.settings._op_id = "op-123"
    op.update_config = Mock()
    op.log = Mock()
    return op


def test_sanitize_attribute_name():
    loader = PlutoLoader()
    name = "metrics/accuracy@user#1"
    sanitized = loader._sanitize_attribute_name(name)
    assert "@" not in sanitized and "#" not in sanitized


def test_convert_step_to_int():
    loader = PlutoLoader()
    assert loader._convert_step_to_int(Decimal("1.5"), 1000) == 1500
    assert loader._convert_step_to_int(None, 1000) is None


def test_upload_parameters_calls_client_update(tmp_path, monkeypatch):
    # Prepare small DataFrame with params
    df = pd.DataFrame(
        [
            {
                "attribute_type": "float",
                "attribute_path": "config/lr",
                "float_value": 0.01,
            },
            {
                "attribute_type": "string",
                "attribute_path": "notes/desc",
                "string_value": "test",
            },
        ]
    )

    loader = PlutoLoader()

    # Mock client with expected API
    mock_client = Mock(spec_set=["runs"])
    mock_client.runs = Mock(spec_set=["update"])
    mock_client.runs.update = Mock()

    # Patch _ensure_client to return our mock
    monkeypatch.setattr(loader, "_ensure_client", lambda: mock_client)

    # Call upload parameters
    loader._upload_parameters(df, run_id="run-123")

    # Expect update called once
    assert mock_client.runs.update.call_count == 1
    args, kwargs = mock_client.runs.update.call_args
    assert kwargs.get("run_id") == "run-123" or args[0] == "run-123"


def test_create_run_uses_pluto_init(monkeypatch):
    loader = PlutoLoader()
    # Mock pluto module with init
    fake_pluto = Mock()
    fake_pluto.init = Mock(return_value=make_op_mock())
    monkeypatch.setattr(loader, "_pluto", fake_pluto)

    run_id = loader.create_run(project_id="org/proj", run_name="r1")
    assert str(run_id) == "op-123"


def test_upload_run_data_uses_op_log(monkeypatch, tmp_path):
    loader = PlutoLoader()
    op = make_op_mock()
    loader._ops = {"op-123": op}

    df = pd.DataFrame([
        {"attribute_type": "float_series", "attribute_path": "metrics/a", "float_value": 0.5, "step": 1},
        {"attribute_type": "string_series", "attribute_path": "notes/msg", "string_value": "hi", "step": 1},
    ])

    # create a generator that yields a single pyarrow table
    tbl = pa.Table.from_pandas(df)
    gen = (t for t in [tbl])

    loader.upload_run_data(run_data=gen, run_id="op-123", files_directory=tmp_path, step_multiplier=1)

    # expect op.log called at least once
    assert op.log.call_count >= 1


def test_upload_parameters_string_set(monkeypatch):
    loader = PlutoLoader()
    df = pd.DataFrame(
        [
            {
                "attribute_type": "string_set",
                "attribute_path": "cfg/tags",
                "string_set_value": ["a", "b", "c"],
            }
        ]
    )

    mock_client = Mock(spec_set=["runs"])
    mock_client.runs = Mock(spec_set=["update"])
    mock_client.runs.update = Mock()
    monkeypatch.setattr(loader, "_ensure_client", lambda: mock_client)

    loader._upload_parameters(df, run_id="R1")
    assert mock_client.runs.update.call_count == 1


def test_create_run_with_parent_and_fork(monkeypatch):
    loader = PlutoLoader()
    fake_pluto = Mock()
    fake_pluto.init = Mock(return_value=make_op_mock())
    monkeypatch.setattr(loader, "_pluto", fake_pluto)

    run_id = loader.create_run(project_id="p", run_name="child", parent_run_id="parent-1", fork_step=2.5)
    assert str(run_id) == "op-123"


def test_upload_run_data_op_update_config_called(monkeypatch, tmp_path):
    loader = PlutoLoader()
    op = make_op_mock()
    loader._ops = {"op-123": op}

    df = pd.DataFrame([
        {"attribute_type": "string", "attribute_path": "cfg/name", "string_value": "v"},
    ])
    tbl = pa.Table.from_pandas(df)
    gen = (t for t in [tbl])

    loader.upload_run_data(run_data=gen, run_id="op-123", files_directory=tmp_path, step_multiplier=1)
    # update_config should be called for params
    assert op.update_config.call_count == 1


def test_upload_artifacts_skips_missing_files(monkeypatch, tmp_path):
    loader = PlutoLoader()
    df = pd.DataFrame(
        [{"attribute_type": "file", "attribute_path": "a/file", "file_value": {"path": "missing.txt"}}]
    )
    mock_client = Mock(spec_set=["upload_artifact"])
    mock_client.upload_artifact = Mock()
    monkeypatch.setattr(loader, "_ensure_client", lambda: mock_client)

    # Path.exists returns False
    monkeypatch.setattr(Path, "exists", lambda self: False)

    loader._upload_artifacts(df, run_id="RUN-X", files_base_path=tmp_path, step_multiplier=1)
    assert mock_client.upload_artifact.call_count == 0


def test_create_experiment_with_projects_api(monkeypatch):
    loader = PlutoLoader()
    class C:
        pass

    mock_client = Mock(spec_set=["projects"])
    mock_proj = Mock()
    mock_proj.id = "proj-789"
    # simulate projects.create
    mock_client.projects = Mock(spec_set=["create"])
    mock_client.projects.create = Mock(return_value=mock_proj)

    monkeypatch.setattr(loader, "_ensure_client", lambda: mock_client)

    exp_id = loader.create_experiment("org/proj", "experiment")
    assert str(exp_id) == "proj-789"
    mock_client.projects.create.assert_called_once()


def test_find_run_uses_list_runs(monkeypatch):
    loader = PlutoLoader()

    mock_client = Mock(spec_set=["list_runs"])
    mock_run = Mock()
    mock_run.name = "run-name"
    mock_run.id = "run-999"
    mock_client.list_runs = Mock(return_value=[mock_run])
    monkeypatch.setattr(loader, "_ensure_client", lambda: mock_client)

    found = loader.find_run("proj", "run-name", "exp-1")
    assert str(found) == "run-999"
    mock_client.list_runs.assert_called_once()


def test_upload_metrics_calls_client_log_metric(monkeypatch):
    loader = PlutoLoader()
    df = pd.DataFrame(
        {
            "attribute_path": ["m1", "m1", "m2"],
            "attribute_type": ["float_series", "float_series", "float_series"],
            "step": [1, 2, 1],
            "timestamp": [None, None, None],
            "float_value": [0.1, 0.2, 0.3],
        }
    )

    mock_client = Mock(spec_set=["log_metric"])
    mock_client.log_metric = Mock()
    monkeypatch.setattr(loader, "_ensure_client", lambda: mock_client)

    loader._upload_metrics(df, run_id="RUN-X", step_multiplier=1)

    # Expect at least 3 calls (one per row)
    assert mock_client.log_metric.call_count == 3


def test_upload_artifacts_calls_upload(monkeypatch, tmp_path):
    loader = PlutoLoader()
    df = pd.DataFrame(
        [
            {"attribute_type": "file", "attribute_path": "a/file", "file_value": {"path": "f1.txt"}},
            {"attribute_type": "file_series", "attribute_path": "b/series", "file_value": {"path": "f2.txt"}, "step": 1},
        ]
    )

    mock_client = Mock()
    mock_client.upload_artifact = Mock()
    monkeypatch.setattr(loader, "_ensure_client", lambda: mock_client)

    # patch Path.exists to True so uploads proceed
    monkeypatch.setattr(Path, "exists", lambda self: True)

    loader._upload_artifacts(df, run_id="RUN-X", files_base_path=tmp_path, step_multiplier=1)

    # Expect upload_artifact called for both entries
    assert mock_client.upload_artifact.call_count >= 2


def test_upload_run_data_fallback_uses_client(monkeypatch, tmp_path):
    loader = PlutoLoader()

    # Create a table with one param, one metric, one file
    df = pd.DataFrame(
        [
            {"attribute_type": "string", "attribute_path": "cfg/x", "string_value": "v"},
            {"attribute_type": "float_series", "attribute_path": "m/x", "float_value": 1.2, "step": 1},
            {"attribute_type": "file", "attribute_path": "out/f", "file_value": {"path": "f.txt"}},
        ]
    )

    tbl = pa.Table.from_pandas(df)
    gen = (t for t in [tbl])

    mock_client = Mock(spec_set=["runs", "log_metric", "upload_artifact"])
    mock_client.runs = Mock(spec_set=["update"])
    mock_client.runs.update = Mock()
    mock_client.log_metric = Mock()
    mock_client.upload_artifact = Mock()

    monkeypatch.setattr(loader, "_ensure_client", lambda: mock_client)
    # patch Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: True)

    loader.upload_run_data(run_data=gen, run_id="RUN-X", files_directory=tmp_path, step_multiplier=1)

    assert mock_client.runs.update.call_count == 1
    assert mock_client.log_metric.call_count >= 1
    assert mock_client.upload_artifact.call_count >= 1
