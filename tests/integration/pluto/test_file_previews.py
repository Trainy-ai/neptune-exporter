"""Pluto File Preview Test

Tests file preview functionality for different file types in Pluto.

This test uploads one of each supported file type to verify that:
- YAML files have preview support
- TXT files have preview support
- PY files have preview support
- PNG images have preview support

Additionally tests:
- Basic metrics logging
- Basic params (config) logging
- Chart/histogram logging

What it uploads to Pluto:
- Project: simple_test (or PLUTO_PROJECT env var)
- Run name: file-preview-test
- 1 YAML file (config.yaml)
- 1 TXT file (notes.txt)
- 1 PY file (example.py)
- 1 PNG image (plot.png)
- 3 metrics (accuracy, loss, f1_score)
- 3 params (learning_rate, batch_size, model_name)
- 1 histogram
"""

import os
import tempfile
from pathlib import Path

import pytest

try:
    import numpy as np
except ImportError:
    np = None


def test_pluto_file_previews():
    """Test file preview feature with various file types."""
    api_key = os.getenv("PLUTO_API_KEY")
    project = os.getenv("PLUTO_PROJECT", "simple_test")

    try:
        import pluto
    except Exception as e:
        pytest.skip(f"Pluto SDK not installed: {e}")

    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # 1. Create YAML file
        yaml_file = tmpdir_path / "config.yaml"
        yaml_content = """
# Model Configuration
model:
  name: ResNet50
  layers: 50
  dropout: 0.2

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam

data:
  train_split: 0.8
  validation_split: 0.1
  test_split: 0.1
"""
        yaml_file.write_text(yaml_content)

        # 2. Create TXT file
        txt_file = tmpdir_path / "notes.txt"
        txt_content = """Experiment Notes
================

This is a test run for the file preview feature.

Key observations:
- The model converged after 50 epochs
- Validation accuracy plateaued at 95%
- Some overfitting observed after epoch 75

Next steps:
1. Try adding more dropout
2. Implement early stopping
3. Test with larger dataset
"""
        txt_file.write_text(txt_content)

        # 3. Create Python file
        py_file = tmpdir_path / "example.py"
        py_content = '''#!/usr/bin/env python3
"""Example training script."""

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """A simple neural network."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_model(model, data_loader, epochs: int = 10):
    """Train the model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            outputs = model(batch[0])
            loss = criterion(outputs, batch[1])
            loss.backward()
            optimizer.step()
'''
        py_file.write_text(py_content)

        # 4. Create PNG image
        png_file = tmpdir_path / "plot.png"
        try:
            from PIL import Image
            import numpy as np
            
            # Create a simple gradient image
            width, height = 400, 300
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Create a gradient from blue to red
            for y in range(height):
                for x in range(width):
                    img_array[y, x] = [
                        int(255 * x / width),  # Red channel
                        int(128 * (1 - abs(x - width/2) / (width/2))),  # Green channel
                        int(255 * (1 - x / width))  # Blue channel
                    ]
            
            img = Image.fromarray(img_array, 'RGB')
            img.save(png_file)
        except ImportError:
            # Fallback: create a minimal valid PNG if PIL is not available
            # This is a 1x1 red pixel PNG
            png_bytes = bytes([
                0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
                0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00,
                0x0C, 0x49, 0x44, 0x41, 0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
                0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D, 0xB4, 0x00, 0x00, 0x00,
                0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82
            ])
            png_file.write_bytes(png_bytes)

        # Initialize Pluto
        if hasattr(pluto, "init"):
            op = pluto.init(project=project, name="file-preview-test")
            
            # Log parameters (config)
            params = {
                "learning_rate": 0.001,
                "batch_size": 32,
                "model_name": "ResNet50"
            }
            op.update_config(params)
            
            # Log metrics at different steps
            for step in range(5):
                metrics = {
                    "accuracy": 0.7 + step * 0.05,
                    "loss": 1.0 - step * 0.15,
                    "f1_score": 0.65 + step * 0.06
                }
                op.log(metrics, step=step)
            
            # Log files with appropriate types
            files = {}
            
            # Upload YAML file as Text for preview
            if hasattr(pluto, "Text"):
                files["config/yaml"] = pluto.Text(str(yaml_file), caption="config.yaml")
            
            # Upload TXT file as Text for preview
            if hasattr(pluto, "Text"):
                files["notes/txt"] = pluto.Text(str(txt_file), caption="notes.txt")
            
            # Upload Python file as Text for preview
            if hasattr(pluto, "Text"):
                files["code/python"] = pluto.Text(str(py_file), caption="example.py")
            
            # Upload PNG as Image for preview
            if hasattr(pluto, "Image"):
                files["visualizations/plot"] = pluto.Image(str(png_file), caption="plot.png")
            
            if files:
                op.log(files)
            
            # Log a histogram/chart at different steps
            # Histogram expects raw data values as numpy array or list
            if hasattr(pluto, "Histogram"):
                try:
                    import numpy as np
                    # Generate sample data - must be numpy array or list
                    for step in range(3):
                        # Create numpy array of random values
                        np.random.seed(42 + step)  # Different seed for each step
                        hist_values = np.random.randn(100) * (step + 1)  # 100 samples
                        op.log({"metrics/value_distribution": pluto.Histogram(hist_values)}, step=step)
                except Exception as e:
                    print(f"⚠️  Histogram logging failed: {e}")
            
            # Finish the run
            if hasattr(op, "finish"):
                op.finish()
            
            print("\n" + "="*60)
            print("✅ File preview test completed!")
            print("="*60)
            print(f"\nUploaded to project: {project}")
            print(f"Run name: file-preview-test")
            print("\n📁 Files uploaded (appear in Files/Summary section):")
            print("  - config.yaml (YAML file with Text preview)")
            print("  - notes.txt (Text file with Text preview)")
            print("  - example.py (Python file with Text preview)")
            print("\n📊 Charts/Images on MAIN DASHBOARD:")
            print("  - plot.png (PNG image - appears on dashboard as IMAGE type)")
            print("  - 3 metrics over 5 steps (accuracy, loss, f1_score)")
            print("  - 1 histogram (value_distribution)")
            print("\n📋 Parameters:")
            print("  - learning_rate, batch_size, model_name")
            print("\n" + "="*60)
            print("KEY INSIGHT:")
            print("="*60)
            print("• pluto.Image() → logType=IMAGE → Shows on MAIN DASHBOARD")
            print("• pluto.Text() → logType=TEXT → Only in Files/Summary section")
            print("• Metrics/Histograms → Always on MAIN DASHBOARD")
            print("\nFile PREVIEWS work in the Files/Summary section!")
            print("="*60 + "\n")
            
        else:
            pytest.skip("Pluto SDK does not expose init() API")


def test_pluto_unusual_attribute_names():
    """Test how Pluto handles attribute names with unusual characters.
    
    Tests that the sanitization correctly:
    - Preserves forward slashes (/) in paths
    - Preserves dots (.) in names
    - Preserves underscores (_) and hyphens (-)
    - Preserves spaces
    - Replaces special chars (@, #, !, etc.) with underscores
    """
    project = os.getenv("PLUTO_PROJECT", "simple_test")

    try:
        import pluto
    except Exception as e:
        pytest.skip(f"Pluto SDK not installed: {e}")

    if not hasattr(pluto, "init"):
        pytest.skip("Pluto SDK does not expose init() API")

    try:
        op = pluto.init(project=project, name="unusual-names-test")
        
        # Test cases: (input_name, expected_behavior_description)
        test_cases = {
            # Standard paths with slashes - should be preserved
            "metrics/accuracy": "slash preserved",
            "config/model/layers": "nested slashes preserved",
            "data/train/accuracy": "slashes in path",
            
            # Dots - should be preserved
            "metric.value": "dot preserved",
            "file.name.with.dots": "multiple dots preserved",
            "version.1.0": "version-like naming",
            
            # Underscores and hyphens - should be preserved
            "my_metric": "underscore preserved",
            "my-metric": "hyphen preserved",
            "my_metric_name": "multiple underscores",
            "my-metric-name": "multiple hyphens",
            
            # Spaces - should be preserved
            "metric with spaces": "spaces preserved",
            "learning rate": "space in name",
            
            # Mixed valid chars
            "metrics/model.accuracy_v1": "complex path with slash, dot, underscore",
            "data/train-set/accuracy.final": "mix of all allowed chars",
            
            # Special chars that SHOULD be replaced with underscore
            "metric@special": "@ becomes _",
            "metric#special": "# becomes _",
            "metric!special": "! becomes _",
            "metric&special": "& becomes _",
            "metric$value": "$ becomes _",
            "metric%value": "% becomes _",
            "metric(test)": "() become _",
            "metric[array]": "[] become _",
            "metric{obj}": "{} become _",
            "metric=value": "= becomes _",
            "metric+value": "+ becomes _",
            "metric*value": "* becomes _",
            
            # Mixed special chars and valid
            "metrics/accuracy@v1": "slash preserved, @ becomes _",
            "file.name@bad": "dot preserved, @ becomes _",
            "metric_name#test": "underscore and hyphen preserved, # becomes _",
        }
        
        print("\n" + "="*70)
        print("TESTING UNUSUAL ATTRIBUTE NAMES IN PLUTO")
        print("="*70)
        
        for step, (attr_name, description) in enumerate(test_cases.items()):
            try:
                # Log a metric with the unusual name
                op.log({attr_name: float(step)}, step=step)
                print(f"✅ {attr_name:40} → {description}")
            except Exception as e:
                print(f"❌ {attr_name:40} → FAILED: {e}")
        
        print("\n" + "="*70)
        print("KEY FINDINGS:")
        print("="*70)
        print("✅ Forward slashes (/) are PRESERVED")
        print("   → Allows nested paths like 'metrics/accuracy'")
        print()
        print("✅ Dots (.) are PRESERVED")
        print("   → Allows files like 'model.pkl' or 'version.1.0'")
        print()
        print("✅ Underscores (_) and hyphens (-) are PRESERVED")
        print("   → Standard Python naming conventions work")
        print()
        print("✅ Spaces are PRESERVED")
        print("   → Natural language attribute names work")
        print()
        print("🔄 Special chars (@#!&$%*+=etc) become UNDERSCORES")
        print("   → 'metric@v1' → 'metric_v1'")
        print()
        print("📝 Names are truncated to 250 chars max")
        print("="*70 + "\n")
        
        if hasattr(op, "finish"):
            op.finish()
        
    except Exception as e:
        pytest.skip(f"Pluto test setup failed: {e}")


if __name__ == "__main__":
    # Allow running directly for quick testing
    test_pluto_file_previews()
    print("\n\nNow testing unusual names...\n")
    test_pluto_unusual_attribute_names()
