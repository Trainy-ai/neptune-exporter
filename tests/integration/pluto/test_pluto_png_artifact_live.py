"""Pluto PNG Artifact Integration Test

Tests file artifact upload (specifically PNG images) to verify the Artifact API works
correctly and images display properly in Pluto's Files tab.

What it tests:
- PNG file creation (matplotlib validation plot)
- File artifact creation using pluto.Artifact(path, caption=...)
- File metadata handling (path, size, hash)
- Combined upload: logs + file artifacts
- Proper positional argument usage (not data= keyword)

What it uploads to Pluto:
- 1 PNG artifact: validation/timeseriesid_25_103665.png (generated matplotlib plot)
- 5 validation log messages (stdout)
- 1 Text artifact: logs/stdout.txt with the 5 messages
- Project: simple_test
- Run name: test_validation_png

This test validates the fix for empty PNG files (using positional argument
instead of data= keyword argument).
"""

import json
import tempfile
from pathlib import Path
from decimal import Decimal
from typing import Generator
import hashlib

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from neptune_exporter.loaders.pluto_loader import PlutoLoader
from neptune_exporter.types import ProjectId, TargetRunId

# Use non-interactive backend for headless PNG generation
matplotlib.use('Agg')


def create_validation_png(output_path: Path) -> None:
    """Create a dummy validation time series PNG similar to Neptune validation plots."""
    # Generate synthetic time series data
    import numpy as np
    
    time_steps = np.linspace(0, 1, 50)
    values = 0.5 + 0.3 * np.sin(2 * np.pi * time_steps) + np.random.normal(0, 0.05, 50)
    values = np.clip(values, 0, 1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_steps, values, linewidth=2, color='steelblue')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('[<TimeSeriesID(25_103665)>] - favorita_sales_target_0 (Patch 63, FCD=2047)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    ax.set_xlim([0, 1.0])
    
    # Add text annotation
    ax.text(0.5, 0.5, 'No future data available from FCD 2047', 
            ha='center', va='center', fontsize=12, color='gray', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=100, format='png')
    plt.close()
    
    print(f"✅ Created validation PNG: {output_path}")


def create_file_metadata(file_path: Path, hash_value: str) -> dict:
    """Create Neptune file metadata dict."""
    return {
        "path": str(file_path.name),
        "size": file_path.stat().st_size,
        "hash": hash_value,
    }


def create_parquet_with_png_artifact(png_path: Path) -> Path:
    """Create parquet data with file_series entry pointing to PNG."""
    
    # Get file hash
    with open(png_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    
    rows = []
    
    # Add string_series log entries
    log_messages = [
        "Validation started",
        "Processing batch 1/10",
        "Processing batch 5/10",
        "Processing batch 10/10",
        "Validation complete - generating plots",
    ]
    
    for i, msg in enumerate(log_messages):
        rows.append({
            "attribute_type": "string_series",
            "attribute_path": "logs/validation",
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
    
    # Add file_series entry for the PNG
    file_value_dict = create_file_metadata(png_path, file_hash)
    rows.append({
        "attribute_type": "file_series",
        "attribute_path": "validation/timeseriesid_25_103665",
        "step": Decimal(0),
        "timestamp": 0,
        "string_value": None,
        "float_value": None,
        "int_value": None,
        "bool_value": None,
        "datetime_value": None,
        "file_value": file_value_dict,
        "string_set_value": None,
        "histogram_value": None,
    })
    
    # Create temporary parquet file
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df)
    
    temp_dir = Path(tempfile.gettempdir()) / "neptune_exporter_test"
    temp_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = temp_dir / "validation_png_test.parquet"
    
    pq.write_table(table, str(parquet_path))
    print(f"✅ Created test parquet: {parquet_path}")
    print(f"   Rows: {len(df)}")
    print(f"   Contains: 5 log messages + 1 PNG artifact")
    
    return parquet_path


def run_data_generator(parquet_path: Path) -> Generator[pa.Table, None, None]:
    """Generator that yields parquet data."""
    table = pq.read_table(str(parquet_path))
    yield table


def test_pluto_png_artifact_integration():
    """Send PNG validation plot to actual Pluto 'simple_test' project."""
    
    print("\n" + "="*60)
    print("VALIDATION PNG ARTIFACT INTEGRATION TEST")
    print("="*60)
    
    # Create temp directory for files
    temp_dir = Path(tempfile.gettempdir()) / "neptune_exporter_test"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create PNG file
    png_path = temp_dir / "validation_timeseriesid_25_103665.png"
    create_validation_png(png_path)
    
    # Create parquet data referencing the PNG
    parquet_path = create_parquet_with_png_artifact(png_path)
    
    # Initialize loader
    loader = PlutoLoader()
    
    # Create experiment
    print("\n📝 Creating Pluto experiment...")
    experiment_id = loader.create_experiment(
        project_id=ProjectId("simple_test"),
        experiment_name="validation_png_test"
    )
    print(f"✅ Experiment ID: {experiment_id}")
    
    # Create run
    print("\n📝 Creating Pluto run...")
    run_id = loader.create_run(
        project_id=ProjectId("simple_test"),
        run_name="test_validation_png",
        experiment_id=experiment_id,
    )
    print(f"✅ Run ID: {run_id}")
    
    # Upload data
    print("\n📤 Uploading PNG artifact to Pluto...")
    print("   Expect to see:")
    print("   - 5 validation log messages in terminal output (below)")
    print("   - 1 PNG artifact 'validation/timeseriesid_25_103665' in Pluto Files tab")
    print("   - PNG should display as a time series plot (not empty)\n")
    
    try:
        loader.upload_run_data(
            run_data=run_data_generator(parquet_path),
            run_id=run_id,
            files_directory=temp_dir,
            step_multiplier=1,
        )
        print("\n✅ Upload complete!")
        print(f"\n🎯 Check Pluto UI:")
        print(f"   Project: simple_test")
        print(f"   Run: test_validation_png")
        print(f"   - Files tab: should have 'validation/timeseriesid_25_103665' PNG artifact")
        print(f"   - PNG should show time series plot (NOT empty)")
        print(f"   - Logs tab: should show validation log messages")
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_pluto_png_artifact_integration()
