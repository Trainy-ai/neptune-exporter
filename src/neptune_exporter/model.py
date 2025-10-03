#
# Copyright (c) 2025, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pyarrow as pa


SCHEMA = pa.schema(
    [
        ("project_id", pa.string()),
        ("run_id", pa.string()),
        ("attribute_path", pa.string()),
        ("attribute_type", pa.string()),
        ("step", pa.decimal128(18, 6)),
        ("timestamp", pa.timestamp("ms", tz="UTC")),
        ("int_value", pa.int64()),
        ("float_value", pa.float64()),
        ("string_value", pa.string()),
        ("bool_value", pa.bool_()),
        ("datetime_value", pa.timestamp("ms", tz="UTC")),
        ("string_set_value", pa.list_(pa.string())),
        (
            "file_value",
            pa.struct(
                [
                    ("path", pa.string()),
                ]
            ),
        ),
        (
            "histogram_value",
            pa.struct(
                [
                    ("type", pa.string()),
                    ("edges", pa.list_(pa.float64())),
                    ("values", pa.list_(pa.float64())),
                ]
            ),
        ),
    ]
)
