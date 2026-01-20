"""Pluto String Series Integration Test

Tests dual-route logging for string_series (logs) to verify they appear in both
Pluto's Files tab (as Text artifacts) and Logs tab (via stdout/stderr).

What it tests:
- String series collection and separation (stdout vs stderr)
- Text artifact creation (logs/stdout and logs/stderr)
- Print statements for Logs tab visibility
- Proper routing based on attribute path ("error"/"stderr" → stderr, else → stdout)

What it uploads to Pluto:
- 8 log messages total:
  - 5 stdout: Training/epoch messages
  - 3 stderr: Warnings and errors
- 2 Text artifacts: logs/stdout.txt and logs/stderr.txt
- Project: simple_test
- Run name: test_string_series_logs

Note: For large datasets (289k+ logs), Logs tab may fail with 502 errors,
but Text artifacts in Files tab will always work.
"""

import os
import sys
import tempfile
from pathlib import Path
from decimal import Decimal
from typing import Generator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from neptune_exporter.loaders.pluto_loader import PlutoLoader
from neptune_exporter.types import ProjectId, TargetExperimentId, TargetRunId


def create_simple_string_series_parquet() -> Path:
    """Create a minimal parquet file with string_series (logs) data."""
    rows = []
    
    # Add some stdout messages (training logs)
    stdout_messages = [
        "Epoch 1/10: Training started",
        "Epoch 1/10: loss=0.523, accuracy=0.72",
        "Epoch 2/10: loss=0.412, accuracy=0.81",
        "Epoch 3/10: loss=0.301, accuracy=0.88",
        "Training complete!",
    ]
    
    for i, msg in enumerate(stdout_messages):
        rows.append({
            "attribute_type": "string_series",
            "attribute_path": "logs/output",
            "step": Decimal(i),
            "timestamp": i,
            "string_value": msg,
            "float_value": None,
            "int_value": None,
            "bool_value": None,
            "datetime_value": None,
            "file_value": None,
            "string_set_value": None,
            "histogram_value": None,
        })
    
    # Add some stderr messages (warnings/errors)
    stderr_messages = [
        "Warning: GPU memory is 85% full",
        "Error: encountered NaN in batch 42",
        "Warning: learning rate will decrease at epoch 5",
    ]
    
    for i, msg in enumerate(stderr_messages, start=len(stdout_messages)):
        rows.append({
            "attribute_type": "string_series",
            "attribute_path": "logs/errors",  # "error" in path routes to stderr
            "step": Decimal(i),
            "timestamp": i,
            "string_value": msg,
            "float_value": None,
            "int_value": None,
            "bool_value": None,
            "datetime_value": None,
            "file_value": None,
            "string_set_value": None,
            "histogram_value": None,
        })
    
    # Create temporary parquet file
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df)
    
    temp_dir = Path(tempfile.gettempdir()) / "neptune_exporter_test"
    temp_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = temp_dir / "string_series_test.parquet"
    
    pq.write_table(table, str(parquet_path))
    print(f"✅ Created test parquet: {parquet_path}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    return parquet_path


def run_data_generator(parquet_path: Path) -> Generator[pa.Table, None, None]:
    """Generator that yields parquet data."""
    table = pq.read_table(str(parquet_path))
    yield table


def test_pluto_string_series_integration():
    """Send string_series data to actual Pluto 'simple_test' project."""
    
    print("\n" + "="*60)
    print("STRING_SERIES INTEGRATION TEST")
    print("="*60)
    
    # Create test data
    parquet_path = create_simple_string_series_parquet()
    
    # Initialize loader
    loader = PlutoLoader()
    
    # Create experiment (returns project name)
    print("\n📝 Creating Pluto experiment...")
    experiment_id = loader.create_experiment(
        project_id=ProjectId("simple_test"),
        experiment_name="string_series_test"
    )
    print(f"✅ Experiment ID: {experiment_id}")
    
    # Create run
    print("\n📝 Creating Pluto run...")
    run_id = loader.create_run(
        project_id=ProjectId("simple_test"),
        run_name="test_string_series_logs",
        experiment_id=experiment_id,
    )
    print(f"✅ Run ID: {run_id}")
    
    # Upload data
    print("\n📤 Uploading string_series data to Pluto...")
    print("   Expect to see:")
    print("   - 5 stdout messages in terminal output (below)")
    print("   - 3 stderr messages in terminal output (below)")
    print("   - Both 'logs/stdout' and 'logs/stderr' text artifacts in Pluto Files tab")
    print("   - Messages in Pluto Logs tab\n")
    
    try:
        loader.upload_run_data(
            run_data=run_data_generator(parquet_path),
            run_id=run_id,
            files_directory=Path("/tmp"),
            step_multiplier=1,
        )
        print("\n✅ Upload complete!")
        print(f"\n🎯 Check Pluto UI:")
        print(f"   Project: simple_test")
        print(f"   Run: test_string_series_logs")
        print(f"   - Logs tab: should show stdout and stderr messages")
        print(f"   - Files tab: should have 'logs/stdout' and 'logs/stderr' text artifacts")
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_pluto_string_series_integration()
